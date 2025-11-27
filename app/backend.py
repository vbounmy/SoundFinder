import gradio as gr
import speech_recognition as sr
import tempfile
import scipy.io.wavfile
import numpy as np
import requests
import os
import json
import urllib.parse
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

recognizer = sr.Recognizer()


# ============================
# LANGUAGE MAP
# ============================
language_map = {
    "Français": "fr-FR",
    "Anglais": "en-US",
    "Arabe": "ar-DZ",
    "Turc": "tr-TR",
    "Espagnol": "es-ES",
    "Italien": "it-IT"
}


# ---------------------------
# AGENT NLP
# ---------------------------
def agent_nlp(user_text: str) -> dict:
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


# ---------------------------
# AGENT SEARCH GENIUS
# ---------------------------
def search_genius(lyrics_fragment):
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {GENIUS_API_KEY}"}
    params = {"q": lyrics_fragment}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        return {"error": "Erreur Genius"}

    data = response.json()

    results = []
    for hit in data["response"]["hits"][:5]:
        results.append({
            "title": hit["result"]["title"],
            "artist": hit["result"]["primary_artist"]["name"]
        })

    return results


# ---------------------------
# AGENT SEARCH GOOGLE (SERPER)
# ---------------------------
def search_google(lyrics_fragment):
    url = "https://google.serper.dev/search"
    payload = {"q": f"song lyrics {lyrics_fragment}"}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": "Erreur Google"}

    data = response.json()

    results = []
    youtube_link = None
    if "organic" in data:
        for item in data["organic"][:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append({
                "title": title,
                "snippet": snippet,
                "link": link
            })
            #Premier lien YouTube
            if not youtube_link and "youtube.com/watch" in link:
                youtube_link = link

    return results, youtube_link

# ---------------------------
# AGENT FINAL DECISION :  LLM
# ---------------------------

def decide_final_title_llm(lyrics_fragment, genius_results, google_results):
    """ 
    Utilise Mistral AI pour décider du titre excat et de l'artiste d'une chanson à partir d'un fragment de paroles et des résultats de recherche.
    """

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    #Prompt
    prompt = f"""
    Tu es un assistant spécialisé dans l'identification de chansons.
    Paroles transmises : "{lyrics_fragment}"

    Résultats Genius : {json.dumps(genius_results, ensure_ascii=False)}
    Résultats Google : {json.dumps(google_results, ensure_ascii=False)}

    Retourne uniquement un JSON avec le format suivant :
    {{
        "title": "<titre excat>",
        "artist": "<artiste>"
    }}
    Si tu n'es pas sûr, mets :
    {{
        "title": "Aucune correspondance trouvée",
        "artist": "Unknown"
    }}
    """

    #Appel à Mistral
    chat_response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "Tu es un assistant qui identifie des chansons."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    #Récupération du JSON renvoyé
    try:
        final_song = chat_response.choices[0].message.content
        if isinstance(final_song, str):
            final_song = json.loads(final_song)
        title = final_song.get("title", "Aucune correspondance trouvée")
        artist = final_song.get("artist", "Unknown")
    except Exception as e:
        title, artist = "Aucune correspondance trouvée", "Unknown"
    return {"title": title, "artist": artist}

# ---------------------------
# YOUTUBE IFRAME HELPER
# ---------------------------
def make_youtube_iframe(youtube_link: str) -> str:
    """Construit un embed YouTube (iframe) à partir d'un lien."""
    if not youtube_link:
        return "No YouTube link found"

    video_id = None
    if "watch?v=" in youtube_link:
        video_id = youtube_link.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_link:
        video_id = youtube_link.split("youtu.be/")[1].split("?")[0]

    if not video_id:
        return f"[Watch on YouTube]({youtube_link})"

    embed_url = f"https://www.youtube.com/embed/{video_id}"
    return f"""
    <div style="width:100%; display:flex; justify-content:center; border-radius:10px">
        <iframe width="720" height="405"
        src="{embed_url}"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>
    </div>
    """

# ---------------------------
# SPOTIFY SEARCH HELPER
# ---------------------------
def build_spotify_search_url(title: str, artist: str) -> str | None:
    """Construit une URL de recherche Spotify à partir du titre + artiste."""
    if not title or title == "Aucune correspondance trouvée":
        return None
    query = f"{title} {artist}".strip()
    q_encoded = urllib.parse.quote_plus(query)
    return f"https://open.spotify.com/search/{q_encoded}"

# ---------------------------
# PIPELINE
# ---------------------------
def vocal_pipeline(audio, language):

    try:

        if audio is None:
            return "Pas d'audio", "", [], [], "Aucune correspondance trouvée", "", "", ""

        if isinstance(audio, dict):
            sample_rate = audio.get("sample_rate", 44100)
            data = audio.get("data", None)
            if data is None:
                return "Audio invalide", "", [], [], "Aucune correspondance trouvée", "", "", ""
        else:
            sample_rate, data = audio

        data = data.astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            scipy.io.wavfile.write(tmp.name, sample_rate, data)
            temp_wav = tmp.name

        # 1. TRANSCRIPTION
        try:
            with sr.AudioFile(temp_wav) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=language_map[language])

        except Exception as e:
            return f"Erreur transcription : {e}", "", [], [], "Aucune correspondance trouvée", "", "", ""

        # 2. NLP
        lyrics_fragment = agent_nlp(text)["lyrics_fragment"]

        # 3. DOUBLE SEARCH
        genius_results = search_genius(lyrics_fragment)
        google_results, youtube_link = search_google(lyrics_fragment)

        # 4. FINAL DECISION (LLM)
        final_song = decide_final_title_llm(lyrics_fragment, genius_results, google_results)
        final_title = f"""
            <div style='line-height: 1.1;'>
                <span style="font-size: 24px; font-weight: 700;">{final_song['title']}</span><br>
                <span style="font-size: 18px; font-weight: 400; opacity: 0.8;">{final_song['artist']}</span>
            </div>
            """

        # 5. YOUTUBE + SPOTIFY LINKS
        youtube_md = f"[Watch on YouTube ↗]({youtube_link})" if youtube_link else "No YouTube link found"
        youtube_html = make_youtube_iframe(youtube_link) if youtube_link else "No YouTube link found"

        spotify_url = build_spotify_search_url(final_song['title'], final_song['artist'])
        spotify_md = f"[Open in Spotify ↗]({spotify_url})" if spotify_url else "No Spotify link found"

        return text, lyrics_fragment, genius_results, google_results, final_title, youtube_md, youtube_html, spotify_md
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n=== TRACEBACK COMPLET ===\n")
        print(tb)
        return f"Erreur interne : {e}\n\nTraceback :\n{tb}", "", [], [], "Aucune correspondance trouvée", "", "", ""