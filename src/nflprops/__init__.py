"""nflprops — NFL player prop prediction system.

Every prop for a game is derived from ONE coherent simulation of that game.
See docs/IMPLEMENTATION_SPEC.md for the normative build contract.
"""

from nflprops.version import MODEL_VERSION, __version__

__all__ = ["MODEL_VERSION", "__version__"]
