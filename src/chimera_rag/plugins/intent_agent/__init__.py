"""IntentAgent (innovation #2) — intent-aware retrieval agent.

Supports both legacy M-R-D-E architecture and new multi-agent architecture.
The active retriever is selected via yaml config:
- intent_agent.strategy_selector / intent_agent.langgraph → legacy
- intent_agent.multi_agent → new multi-agent architecture
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

# New multi-agent architecture registrations
from chimera_rag.plugins.intent_agent import classifier as _classifier  # noqa: F401
from chimera_rag.plugins.intent_agent import orchestrator as _orchestrator  # noqa: F401

# Legacy registrations (kept for backward compatibility)
from chimera_rag.plugins.intent_agent import (  # noqa: F401
    experiential_memory,
    intent_classifier,
    session_memory,
)

# Legacy modules that may fail if old code is partially present
for _mod_name in ("agent", "reasoning", "strategies", "langgraph", "hooks"):
    try:
        __import__(f"chimera_rag.plugins.intent_agent.{_mod_name}")
    except (ImportError, Exception) as _e:
        _logger.debug("optional legacy module %s not loaded: %s", _mod_name, _e)
