"""BALLDONTLIE NFL HTTP transport.

This module is deliberately football-ignorant.  It performs authentication,
OpenAPI-compatible query encoding, bounded retry/backoff, JSON decoding, cursor
pagination and an optional immutable-raw persistence callback.
"""

from __future__ import annotations

import email.utils
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from nflprops.providers.bdl import endpoints

RetrySleep = Callable[[float], None]
RawHook = Callable[..., None]

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class BDLClient:
    """Synchronous BDL transport.

    ``raw_hook`` is invoked after every HTTP response with keyword arguments
    ``path``, ``params``, ``requested_at``, ``received_at``, ``status_code`` and
    ``payload``.  Phase 2 can supply the immutable raw-store implementation without
    coupling this client to a storage backend.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 30,
        max_retries: int = 5,
        per_page: int = 100,
        *,
        raw_hook: RawHook | None = None,
        client: httpx.Client | None = None,
        sleep: RetrySleep = time.sleep,
        jitter_seed: int = 0,
    ) -> None:
        if not api_key:
            raise ValueError("BDL API key is required")
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be in [1, 100]")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.per_page = per_page
        self.raw_hook = raw_hook
        self._sleep = sleep
        self._rng = random.Random(jitter_seed)
        # PHASE 4: cumulative retry count since the last pop_retry_count().
        # Callers doing one logical resource fetch (which may itself involve
        # several paginated get() calls) read this once, immediately after,
        # via pop_retry_count() -- never a second, independent retry loop.
        self._retry_count = 0
        self._headers = {
            "Authorization": api_key,
            "Accept": "application/json",
            "User-Agent": "nflprops/2026.1.0",
        }
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers=self._headers,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def pop_retry_count(self) -> int:
        """Return retries accumulated since the last call, then reset to 0.

        PHASE 4 telemetry only -- this reads the outcome of the single
        existing retry loop in `get()`; it does not add a second one. A
        caller doing one logical resource fetch (possibly several paginated
        `get()` calls) should call this exactly once, immediately after.
        """
        count = self._retry_count
        self._retry_count = 0
        return count

    def __enter__(self) -> BDLClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(UTC)

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None and response.status_code == 429:
            value = response.headers.get("Retry-After")
            if value:
                try:
                    return max(0.0, float(value))
                except ValueError:
                    try:
                        dt = email.utils.parsedate_to_datetime(value)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                        return max(0.0, (dt - self._utcnow()).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
        base = min(30.0, 0.5 * (2**attempt))
        return base + self._rng.uniform(0.0, min(1.0, base * 0.25))

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue one GET request and return a JSON object.

        Transport exceptions and 408/429/5xx transient responses are retried.
        400/401/403/404 are never retried.
        """
        encoded = self._encode_params(path, dict(params or {}))
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            requested_at = self._utcnow()
            response: httpx.Response | None = None
            try:
                response = self._client.get(path, params=encoded, headers=self._headers)
                received_at = self._utcnow()

                payload: Any
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"_non_json_body": response.text}

                if self.raw_hook is not None:
                    self.raw_hook(
                        path=path,
                        params=encoded,
                        requested_at=requested_at,
                        received_at=received_at,
                        status_code=response.status_code,
                        payload=payload,
                    )

                if response.status_code in _RETRYABLE_STATUS:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    self._retry_count += 1
                    self._sleep(self._retry_delay(response, attempt))
                    continue

                response.raise_for_status()
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"BDL response for {path!r} was not a JSON object"
                    )
                return payload

            except httpx.HTTPStatusError:
                # Non-retryable status or exhausted retry budget.
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                self._retry_count += 1
                self._sleep(self._retry_delay(response, attempt))

        assert last_exc is not None
        raise last_exc

    def paginated_get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield rows across BDL cursor pages.

        ``cursor`` is an integer in the current official OpenAPI contract.
        """
        if path in endpoints.NON_PAGINATED:
            raise ValueError(f"{path} is not a cursor-paginated endpoint")

        base = dict(params or {})
        base.setdefault("per_page", self.per_page)
        cursor: int | None = None
        seen: set[int] = set()

        while True:
            page_params = dict(base)
            if cursor is not None:
                page_params["cursor"] = cursor
            response = self.get(path, page_params)
            data = response.get("data", [])
            if not isinstance(data, list):
                raise ValueError(f"BDL paginated response for {path} has non-list data")
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"BDL row for {path} is not an object")
                yield item

            meta = response.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if next_cursor in (None, ""):
                break
            try:
                next_cursor = int(next_cursor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid next_cursor {next_cursor!r} for {path}"
                ) from exc
            if next_cursor in seen:
                raise RuntimeError(f"cursor loop detected for {path}: {next_cursor}")
            seen.add(next_cursor)
            cursor = next_cursor

    def _encode_params(
        self,
        path: str,
        params: Mapping[str, Any],
    ) -> list[tuple[str, Any]]:
        """Encode endpoint-specific BDL form/explode query parameters.

        Returning a list of pairs is intentional: repeated form keys must not be
        collapsed by a dictionary.
        """
        mapping = endpoints.ARRAY_WIRE_NAMES.get(path, {})
        out: list[tuple[str, Any]] = []

        for key, value in params.items():
            if value is None:
                continue
            wire_key = mapping.get(key, key)
            is_array = key in mapping

            if is_array:
                values: Sequence[Any]
                if isinstance(value, (str, bytes)):
                    values = [value]
                elif isinstance(value, Sequence):
                    values = value
                else:
                    values = [value]
                for item in values:
                    if item is not None:
                        out.append((wire_key, item))
                continue

            # Reject accidental list use on scalar parameters.  This catches the
            # season_type array-vs-scalar trap before a network request is made.
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                raise TypeError(
                    f"{key!r} is scalar for BDL endpoint {path!r}; got sequence"
                )
            out.append((wire_key, value))

        return out
