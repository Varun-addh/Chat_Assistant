"""Service layer package.

This module keeps a couple of commonly-used re-exports for convenience.
"""

from .core.session_manager import get_session_manager  # re-export
from .chat.llm_service import llm_service  # re-export
