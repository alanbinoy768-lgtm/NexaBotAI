import random
from typing import Any, Dict, List, Optional

from src.config import DEFAULT_CONFIDENCE_THRESHOLD, INTENTS_PATH, LOW_CONFIDENCE_THRESHOLD
from src.knowledge_retriever import retrieve_answer
from src.nlp_engine import load_intent_dataset


def load_response_templates(dataset_path: str = INTENTS_PATH) -> Dict[str, List[str]]:
    dataset = load_intent_dataset(dataset_path)
    return {intent["tag"]: intent.get("responses", []) for intent in dataset.get("intents", [])}


def choose_response(intent: str, responses: List[str], recent_responses: Optional[List[str]] = None) -> str:
    recent_responses = recent_responses or []
    available = [response for response in responses if response not in recent_responses]
    if not available:
        available = responses

    if intent == "greeting":
        preferred = [response for response in available if "Hello" in response or "Hi" in response]
        if preferred:
            return random.choice(preferred)

    return random.choice(available)


def build_contextual_response(user_text: str, last_user_message: Optional[str], intent: str) -> Optional[str]:
    text = (user_text or "").lower()
    if intent == "conversation" and "who created it" in text and last_user_message and "python" in last_user_message.lower():
        return "Python was created by Guido van Rossum."
    if intent == "conversation" and "who created" in text and last_user_message:
        return "I do not have enough context for that specific person or topic. Please tell me what you are referring to."
    return None


def generate_response(user_text: str, intent: str, confidence: float, history: Optional[List[Dict[str, Any]]] = None, current_context: Optional[str] = None, knowledge_answer: Optional[str] = None) -> Dict[str, Any]:
    history = history or []
    last_user_message = None
    if history:
        for item in reversed(history):
            if item.get("sender") == "user":
                last_user_message = item.get("text")
                break

    templates = load_response_templates()
    responses = templates.get(intent, ["I am here to help you."])

    contextual = build_contextual_response(user_text, last_user_message, intent)
    if contextual:
        return {"text": contextual, "intent": intent, "confidence": confidence}

    if intent == "greeting" and "how are you" in user_text.lower():
        return {
            "text": "I am doing well, thanks for asking! What can I help you with?",
            "intent": intent,
            "confidence": confidence,
        }

    if knowledge_answer:
        return {"text": knowledge_answer, "intent": intent, "confidence": confidence}

    if intent == "unknown" or confidence < LOW_CONFIDENCE_THRESHOLD:
        knowledge = retrieve_answer(user_text)
        if knowledge:
            return {
                "text": knowledge["answer"],
                "intent": "general_knowledge",
                "confidence": confidence,
                "knowledge": knowledge,
            }
        return {
            "text": "I could not find a reliable answer in my local knowledge base for that question. Please ask about a topic I know or rephrase it.",
            "intent": "unknown",
            "confidence": confidence,
        }

    if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
        return {
            "text": "I am not certain I understood that correctly. Please rephrase the question or provide a little more detail.",
            "intent": "clarification",
            "confidence": confidence,
        }

    bot_reply = choose_response(intent, responses, [response.get("text") for response in history if response.get("sender") == "bot"])
    return {"text": bot_reply, "intent": intent, "confidence": confidence}
