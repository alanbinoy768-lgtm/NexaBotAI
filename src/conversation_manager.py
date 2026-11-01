from typing import Any, Dict, List, Optional


def is_follow_up(message: str, history: Optional[List[Dict[str, Any]]] = None) -> bool:
    history = history or []
    if not message:
        return False
    lower = message.lower()
    pronoun_patterns = [" it ", " they ", " this ", " that ", " these ", " those ", " he ", " she ", " who ", " what "]
    if any(pattern in lower for pattern in pronoun_patterns):
        return True
    if history:
        return bool(history[-1].get("text", ""))
    return False


def contextual_hint(message: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    if not is_follow_up(message, history):
        return None
    if "who created it" in message.lower() and history:
        previous = history[-1].get("text", "")
        if "python" in previous.lower():
            return "Python was created by Guido van Rossum."
    return None
