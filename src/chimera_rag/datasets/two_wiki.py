"""2WikiMultihopQA dataset loader.

The native format closely mirrors HotpotQA's shape, so we subclass
:class:`HotpotQALoader` to reuse its parsing logic. Documented as a
separate class to keep the public registry name (``two_wiki``)
self-describing and to give future divergence a clear home.
"""

from __future__ import annotations

from chimera_rag.datasets.hotpotqa import HotpotQALoader


class TwoWikiLoader(HotpotQALoader):
    """2WikiMultihopQA loader — same schema as HotpotQA."""


__all__ = ["TwoWikiLoader"]
