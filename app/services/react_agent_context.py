"""
Simple context service for ReactAgent to pass data to tools
"""
from typing import Optional, Dict, Any
import contextvars
import logging

logger = logging.getLogger(__name__)

# Context variables for async/thread-safe context passing
_user_id_context: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar('user_id', default=None)
_selected_document_ids_context: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar('selected_document_ids', default=None)

def set_current_context(user_id: int, selected_document_ids: Optional[list] = None) -> None:
    """Set context for current async context"""
    _user_id_context.set(user_id)
    _selected_document_ids_context.set(selected_document_ids or [])

def get_current_context() -> Optional[Dict[str, Any]]:
    """Get context for current async context"""
    user_id = _user_id_context.get(None)
    selected_document_ids = _selected_document_ids_context.get([])
    
    if user_id is None:
        return None
    
    return {
        'user_id': user_id,
        'selected_document_ids': selected_document_ids
    }

def clear_current_context() -> None:
    """Clear context for current async context"""
    _user_id_context.set(None)
    _selected_document_ids_context.set([])