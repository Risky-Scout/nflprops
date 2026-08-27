"""BALLDONTLIE NFL provider adapter.

Transport, permissive raw schemas, provider quirks, strict canonical mapping and
capability facade are implemented in this package.  A real pinned OpenAPI snapshot
is still required before production use.
"""

from nflprops.providers.bdl.client import BDLClient
from nflprops.providers.bdl.provider import BDLProvider

__all__ = ["BDLClient", "BDLProvider"]
