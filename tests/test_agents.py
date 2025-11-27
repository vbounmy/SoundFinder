import unittest
import numpy as np

# ========= COPIE DES AGENTS =========

def clean_lyrics(user_text: str) -> dict:
    text = user_text.lower()
    useless_phrases = [
        "c'est quoi", "c est quoi", "la chanson", "la musique", "qui dit",
        "tu connais", "trouve", "quelle est", "donne moi", "donne-moi",
        "est ce que", "est-ce que", "le son", "le titre"
    ]
    for phrase in useless_phrases:
        text = text.replace(phrase, "")
    text = " ".join(text.split()).strip()
    return {"lyrics_fragment": text}

def make_message(from_agent, to, type_, payload, context=None):
    return {
        "from_agent": from_agent,
        "to": to,
        "type": type_,
        "payload": payload,
        "context": context or {}
    }

def send_message(msg):
    target = msg["to"]
    return AGENTS[target](msg)

# ==== AGENT 1 ====
def agent_nlp(msg):
    text = msg["payload"]["text"]
    return make_message("agent_nlp", msg["from_agent"], "response",
                        clean_lyrics(text))

# ==== AGENT 2 ====
def agent_transcriber(msg):
    audio = msg["payload"]["audio"]
    if audio is None:
        return make_message("agent_transcriber", msg["from_agent"], "response",
                            {"error": "Pas d'audio", "text": ""})
    return make_message("agent_transcriber", msg["from_agent"], "response",
                        {"error": None, "text": "fake transcription"})

# ==== AGENT 3 ====
def agent_retriever(msg):
    fragment = msg["payload"]["lyrics_fragment"]
    if fragment == "":
        return make_message("agent_retriever", msg["from_agent"], "response",
                            {"genius_results": [], "google_results": []})
    return make_message("agent_retriever", msg["from_agent"], "response",
                        {"genius_results": [{"title": "test"}],
                         "google_results": [{"title": "google"}]})

# ==== AGENT 4 ====
def agent_decider(msg):
    return make_message("agent_decider", msg["from_agent"], "response",
                        {"final_title": "Titre final"})

# ===== REGISTRY =====
AGENTS = {
    "agent_nlp": agent_nlp,
    "agent_transcriber": agent_transcriber,
    "agent_retriever": agent_retriever,
    "agent_decider": agent_decider
}

class TestAgents(unittest.TestCase):

    def test_agent_nlp(self):
        msg = make_message("test", "agent_nlp", "request", {"text": "c'est quoi hello ?"})
        response = send_message(msg)
        self.assertIn("lyrics_fragment", response["payload"])
        self.assertEqual(response["payload"]["lyrics_fragment"], "hello ?")

    def test_agent_transcriber_empty(self):
        msg = make_message("test", "agent_transcriber", "request",
                           {"audio": None, "language": "Français"})
        response = send_message(msg)
        self.assertIn("error", response["payload"])
        self.assertEqual(response["payload"]["text"], "")

    def test_agent_retriever_invalid(self):
        msg = make_message("test", "agent_retriever", "request",
                           {"lyrics_fragment": ""})
        response = send_message(msg)
        self.assertEqual(response["payload"]["genius_results"], [])
        self.assertEqual(response["payload"]["google_results"], [])

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)