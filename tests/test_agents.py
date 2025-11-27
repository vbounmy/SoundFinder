import unittest

from app.backend import (
    agent_nlp,
    search_genius,
    search_google,
    decide_final_title_llm,
    make_youtube_iframe,
    build_spotify_search_url,
)

print("LOADED TEST FILE")

class TestAgents(unittest.TestCase):

    # =====================
    # TEST NLP CLEANING
    # =====================
    def test_agent_nlp(self):
        res = agent_nlp("C'est quoi la chanson qui dit shape of you ?")
        assert isinstance(res, dict)
        assert "shape of you" in res["lyrics_fragment"]

    # =====================
    # TEST GENIUS SEARCH (mock si API désactivée)
    # =====================
    def test_search_genius(self):
        res = search_genius("shape of you")
        # Retour normal = list
        assert isinstance(res, list)

    # =====================
    # TEST GOOGLE SEARCH
    # =====================
    def test_search_google(self):
        results, yt = search_google("shape of you")
        assert isinstance(results, list)
        assert isinstance(yt, str) or yt is None

    # =========================
    # TEST FINAL DECISION LLM
    # =========================
    def test_decide_final_title_llm(self):
        lyrics = "shape of you"
        genius_res = [{"title": "Shape of You", "artist": "Ed Sheeran"}]
        google_res = [{"title": "Shape of You Lyrics", "link": "https://youtube.com"}]

        res = decide_final_title_llm(lyrics, genius_res, google_res)

        assert isinstance(res, dict)
        assert "title" in res
        assert "artist" in res

    # =====================
    # TEST YOUTUBE EMBED
    # =====================
    def test_make_youtube_iframe(self):
        iframe = make_youtube_iframe("https://www.youtube.com/watch?v=JGwWNGJdvx8")
        assert "<iframe" in iframe

    # =====================
    # TEST SPOTIFY URL
    # =====================
    def test_build_spotify_search_url(self):
        url = build_spotify_search_url("Shape of You", "Ed Sheeran")
        assert url.startswith("https://open.spotify.com/search/")
