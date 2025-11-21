import gradio as gr
import speech_recognition as sr
import tempfile
import scipy.io.wavfile
import numpy as np
import requests
import os
import re
import unicodedata
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Language_map
language_map = {
    "Français": "fr-FR",
    "Anglais": "en-US",
    "Arabe": "ar-DZ",
    "Turc": "tr-TR",
    "Espagnol": "es-ES",
    "Italien": "it-IT"
}

def agent_nlp(user_text: str) -> dict:
    """
    Agent NLP : nettoie la phrase de l'utilisateur et extrait la partie utile.
    Exemple : "c'est quoi la chanson qui dit I'm blinded by the lights"
    → extrait : "i'm blinded by the lights"
    """

    # on passe tout en minuscule
    text = user_text.lower()

    # mots qui ne servent à rien
    useless_words = [
        "c'est quoi",
        "c est quoi",
        "la chanson",
        "la musique",
        "qui dit",
        "tu connais",
        "trouve",
        "quelle est",
        "donne moi",
        "donne-moi"
    ]

    # on enlève ces mots de la phrase
    for uw in useless_words:
        text = text.replace(uw, "")

    # on enlève les espaces au début/fin
    text = text.strip()

    return {
        "lyrics_fragment": text
    }

def audio_to_text(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="fr-FR")
        return text
    except Exception as e:
        return f"Erreur : {e}"
    
recognizer = sr.Recognizer()

def transcribe(audio):
    if audio is None:
        return "Aucun fichier reçu."

    # audio = (sample_rate, np_array)
    sample_rate, data = audio
    data = data.astype(np.int16)

    # Sauvegarde en wav temporaire
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        scipy.io.wavfile.write(tmp.name, sample_rate, data)
        temp_wav = tmp.name

    try:
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="fr-FR")
        return text
    except Exception as e:
        return f"Erreur : {e}"

def agent_nlp(user_text: str) -> dict:
    text = user_text.lower()

    useless_words = [
        "c'est quoi",
        "c est quoi",
        "la chanson",
        "la musique",
        "qui dit",
        "tu connais",
        "trouve",
        "quelle est",
        "donne moi",
        "donne-moi",
        "c'est",
        "est ce que"
    ]

    for uw in useless_words:
        text = text.replace(uw, "")

    text = text.strip()

    return {"lyrics_fragment": text}

recognizer = sr.Recognizer()

def transcribe_and_clean(audio):
    if audio is None:
        return "Aucun fichier reçu.", ""

    sample_rate, data = audio
    data = data.astype(np.int16)

    # sauver en wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        scipy.io.wavfile.write(tmp.name, sample_rate, data)
        temp_wav = tmp.name

    # transcription
    try:
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="fr-FR")
    except Exception as e:
        return f"Erreur : {e}", ""

    # NLP
    nlp_output = agent_nlp(text)

    return text, nlp_output["lyrics_fragment"]

def agent_nlp(user_text: str) -> dict:
    """
    Agent NLP amélioré :
    - enlève les phrases inutiles
    - garde uniquement les paroles
    - nettoie la ponctuation et les espaces
    """

    # 1. mettre en minuscule
    text = user_text.lower()

    # 2. enlever les accents (optionnel)
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # 3. liste des mots inutiles à supprimer
    useless_phrases = [
        "c'est quoi",
        "c est quoi",
        "la chanson",
        "la musique",
        "qui dit",
        "qui fait",
        "comment s'appelle",
        "comment s appelle",
        "tu connais",
        "trouve",
        "quelle est",
        "donne moi",
        "donne-moi",
        "c'est",
        "est ce que",
        "est-ce que",
        "le son",
        "le titre",
        "ce son"
    ]

    # 4. supprimer les phrases inutiles
    for phrase in useless_phrases:
        text = text.replace(phrase, "")

    # 5. supprimer les caractères spéciaux
    text = re.sub(r"[^a-zA-Z0-9' ]+", " ", text)

    # 6. nettoyer les espaces
    text = " ".join(text.split()).strip()

    return {"lyrics_fragment": text}

def agent_search(lyrics_fragment):
    """
    Agent Search : utilise Genius API pour trouver les musiques
    correspondant aux paroles extraites par l'agent NLP.
    """

    # URL de recherche
    base_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {GENIUS_API_KEY}"}
    params = {"q": lyrics_fragment}

    # Envoi de la requête
    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code != 200:
        return {
            "error": "Erreur API Genius",
            "status": response.status_code,
            "details": response.text
        }

    data = response.json()

    # Extraire les résultats
    candidates = []

    for hit in data["response"]["hits"]:
        title = hit["result"]["title"]
        artist = hit["result"]["primary_artist"]["name"]
        candidates.append({
            "title": title,
            "artist": artist
        })

    return {"candidates": candidates}

def vocal_to_music(audio):
    # 1) Vérification
    if audio is None:
        return "Aucun fichier reçu.", "", ""

    # audio = (sample_rate, np_array)
    sample_rate, data = audio
    data = data.astype(np.int16)

    # --- Convertir en wav ---
    import tempfile, scipy.io.wavfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        scipy.io.wavfile.write(tmp.name, sample_rate, data)
        temp_wav = tmp.name

    # --- 2) Transcription ---
    try:
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="fr-FR")
    except Exception as e:
        return f"Erreur : {e}", "", ""

    # --- 3) NLP : extraction ---
    nlp_output = agent_nlp(text)
    lyrics = nlp_output["lyrics_fragment"]

    # --- 4) SEARCH : Genius API ---
    search_result = agent_search(lyrics)

    return text, lyrics, search_result

# 🔍 SEARCH GOOGLE (SERPER API)
def search_serper(lyrics_fragment):
    url = "https://google.serper.dev/search"

    payload = {
        "q": f"song lyrics {lyrics_fragment}"
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        return {"error": "Erreur Serper", "status": response.status_code}

    data = response.json()

    google_results = []

    if "organic" in data:
        for item in data["organic"][:5]:  # 5 résultats max
            google_results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")
            })

    return {"google_results": google_results}


# 🎵 SEARCH GENIUS API
def search_genius(lyrics_fragment):
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {GENIUS_API_KEY}"}
    params = {"q": lyrics_fragment}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        return {"error": "Erreur Genius", "status": response.status_code}

    data = response.json()

    genius_results = []

    for hit in data["response"]["hits"][:5]:  # 5 résultats max
        genius_results.append({
            "title": hit["result"]["title"],
            "artist": hit["result"]["primary_artist"]["name"]
        })

    return {"genius_results": genius_results}


# 🌟 AGENT SEARCH : COMBINAISON DES 2 API
def agent_search_double(lyrics_fragment):
    genius = search_genius(lyrics_fragment)
    google = search_serper(lyrics_fragment)

    return {
        "genius": genius,
        "google": google
    }

recognizer = sr.Recognizer()

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
    if "organic" in data:
        for item in data["organic"][:5]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")
            })

    return results


# ---------------------------
# PIPELINE COMPLET (INTERFACE)
# ---------------------------
def vocal_pipeline(audio, language):


    if audio is None:
        return "Pas d'audio", "", [], []

    # Convertir audio -> wav
    sample_rate, data = audio
    data = data.astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        scipy.io.wavfile.write(tmp.name, sample_rate, data)
        temp_wav = tmp.name

    # --- 1. Transcription vocale ---
    try:
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_map[language])

    except Exception as e:
        return f"Erreur transcription : {e}", "", [], []

    # --- 2. NLP ---
    lyrics_fragment = agent_nlp(text)["lyrics_fragment"]

    # --- 3. Double Search ---
    genius_results = search_genius(lyrics_fragment)
    google_results = search_google(lyrics_fragment)

    return text, lyrics_fragment, genius_results, google_results


# ---------------------------
# GRADIO INTERFACE
# ---------------------------
interface = gr.Interface(
    fn=vocal_pipeline,
    inputs=[
        gr.Audio(type="numpy"),
        gr.Dropdown(
            choices=["Français", "Anglais", "Arabe", "Turc", "Espagnol", "Italien"],
            label="Choisis ta langue"
        )
    ],
    outputs=[
        "text",
        "text",
        "json",
        "json"
    ],
    title="🎤 Multilingue Music Finder",
    description="Parle dans n'importe quelle langue."
)

import gradio as gr
from pathlib import Path

# -----------------------------
#  YOUR FUNCTIONS (THE SAME AS BEFORE)
# -----------------------------
def process_audio(audio_file, language):
    """
    Replace by your real pipeline:
    1. Speech-to-text
    2. Extract lyrics
    3. Genius search
    4. Google search
    """

    if audio_file is None:
        return "Aucun audio détecté.", "", "", ""

    # TEST OUTPUT – replace with your real logic
    trans = "Transcription OK"
    lyrics = "Paroles extraites OK"
    genius = "Résultats Genius OK"
    google = "Résultats Google OK"

    return trans, lyrics, genius, google


# -----------------------------
#  CUSTOM CSS (VIOLET + NOIR)
# -----------------------------
custom_css = """
* {
    font-family: "Inter", sans-serif !important;
}

body {
    background: linear-gradient(180deg, #0f001b, #1a002e, #000000);
    color: white !important;
}

#landing-box {
    max-width: 450px;
    margin: auto;
    padding: 20px;
    text-align: center;
}

#hero-img {
    width: 100%;
    border-radius: 20px;
    box-shadow: 0 0 40px rgba(180,0,255,0.4);
    transition: 0.4s;
}

#hero-img:hover {
    transform: scale(1.02);
    box-shadow: 0 0 60px rgba(250,0,255,0.6);
}

#launch-btn {
    background: linear-gradient(90deg, #9b3efc, #ef46c7);
    color: white;
    border-radius: 14px;
    height: 58px;
    font-size: 20px;
    margin-top: 30px;
}

#launch-btn:hover {
    opacity: 0.85;
}

.interface-box {
    max-width: 900px;
    margin: auto;
}
"""


# -----------------------------
#   BUILD THE APP
# -----------------------------
with gr.Blocks(css=custom_css, title="MusicFinder AI") as app:

    # ----------- LANDING PAGE -----------
    with gr.Column(elem_id="landing-box", visible=True) as landing:

        gr.HTML("""
        <h1 style="font-size:45px; font-weight:800; color:#d58cff;">
            🎧 MusicFinder AI
        </h1>
        <p style="font-size:18px; opacity:0.9;">
            Trouvez une musique à partir de votre voix.<br>
            Multilingue • Intelligent • Instantané
        </p>
        """)

        # Image Landing
        gr.Image(
            value="music.png",
            elem_id="hero-img",
            show_label=False,
            interactive=False
        )

        launch_btn = gr.Button("Launch the App", elem_id="launch-btn")

    # ----------- MAIN APP -----------
    with gr.Column(elem_id="main-ui", visible=False) as main:

        gr.HTML("<h2 style='text-align:center; margin-bottom:25px;'>🎤 Analyse vocale</h2>")

        audio_input = gr.Audio(type="filepath", label="Upload / Enregistre audio", interactive=True)

        lang = gr.Dropdown(
            ["Français", "Anglais", "Arabe", "Espagnol", "Turc", "Italien"],
            value="Français",
            label="Choisissez la langue"
        )

        transcription = gr.Textbox(label="Transcription générée…")
        lyrics_box = gr.Textbox(label="Paroles extraites…")
        genius_box = gr.Textbox(label="Résultats Genius…")
        google_box = gr.Textbox(label="Résultats Google…")

        submit_btn = gr.Button("Submit", elem_id="launch-btn")
        clear_btn = gr.Button("Clear")


    # --------------------------
    #  BUTTON LINKING
    # --------------------------

    launch_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None,
        [landing, main]
    )

    submit_btn.click(
        process_audio,
        inputs=[audio_input, lang],
        outputs=[transcription, lyrics_box, genius_box, google_box]
    )

    clear_btn.click(
        lambda: ("", "", "", ""),
        None,
        [transcription, lyrics_box, genius_box, google_box]
    )

recognizer = sr.Recognizer()

# ============================
# 🌍 LANGUAGE MAP
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
    if "organic" in data:
        for item in data["organic"][:5]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")
            })

    return results


# ---------------------------
# PIPELINE COMPLET (ROBUSTE)
# ---------------------------
def vocal_pipeline(audio, language):

    if audio is None:
        return "Pas d'audio", "", [], []

    # ✅ Compatible anciennes & nouvelles versions Gradio
    if isinstance(audio, dict):
        sample_rate = audio["sample_rate"]
        data = audio["data"]
    else:
        sample_rate, data = audio

    data = data.astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        scipy.io.wavfile.write(tmp.name, sample_rate, data)
        temp_wav = tmp.name

    # --- 1. Transcription vocale ---
    try:
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_map[language])

    except Exception as e:
        return f"Erreur transcription : {e}", "", [], []

    # --- 2. NLP ---
    lyrics_fragment = agent_nlp(text)["lyrics_fragment"]

    # --- 3. Double Search ---
    genius_results = search_genius(lyrics_fragment)
    google_results = search_google(lyrics_fragment)

    return text, lyrics_fragment, genius_results, google_results


# ============================
# 🎨 CSS VIOLET/NOIR PREMIUM
# ============================
custom_css = """
body {
    background: radial-gradient(circle at top, #2a0040, #000000) !important;
}
.gradio-container {
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}

#title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #d58cff;
    text-shadow: 0 0 25px rgba(180,0,255,0.7);
    margin-top: 10px;
}
#subtitle {
    text-align: center;
    color: #e9d4ff;
    font-size: 18px;
    margin-bottom: 20px;
}

#hero-img img {
    width: 370px;
    border-radius: 20px;
    box-shadow: 0 0 45px rgba(180,0,255,0.45);
    margin: auto;
    display: block;
}

.neon-box {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(160,80,255,0.9) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 18px rgba(140,0,255,0.25);
}

button {
    font-weight: 800 !important;
}

#submit-btn {
    background: linear-gradient(90deg, #9b3efc, #ff3eb5) !important;
    color: white !important;
    border-radius: 14px !important;
    height: 55px !important;
    font-size: 18px !important;
}
#submit-btn:hover {
    opacity: 0.9;
    transform: scale(1.01);
}

#clear-btn {
    background: #444 !important;
    color: white !important;
    border-radius: 14px !important;
    height: 50px !important;
    font-size: 16px !important;
}

textarea, input, select {
    background: rgba(0,0,0,0.55) !important;
    color: white !important;
    border: 1px solid rgba(160,80,255,0.8) !important;
    border-radius: 10px !important;
}
"""


# ============================
# 🧠 INTERFACE PRO (BLOCKS)
# ============================
with gr.Blocks(css=custom_css) as demo:

    gr.HTML("""
        <div id="title">🎧 MusicFinder AI</div>
        <div id="subtitle">
            Trouvez facilement une musique à partir de votre voix,
            dans n'importe quelle langue.
        </div>
    """)

    gr.Image("music.png", show_label=False, elem_id="hero-img")

    with gr.Column(elem_classes="neon-box"):
        audio_input = gr.Audio(type="numpy", label="🎤 Enregistrez / Upload ton audio")
        lang_choice = gr.Dropdown(
            choices=list(language_map.keys()),
            value="Français",
            label="Choisissez la langue"
        )

    with gr.Accordion("Transcription", open=True, elem_classes="neon-box"):
        out_transcript = gr.Textbox(lines=4)

    with gr.Accordion("Paroles extraites", open=False, elem_classes="neon-box"):
        out_lyrics = gr.Textbox(lines=3)

    with gr.Accordion("Résultats Genius", open=False, elem_classes="neon-box"):
        out_genius = gr.JSON()

    with gr.Accordion("Résultats Google", open=False, elem_classes="neon-box"):
        out_google = gr.JSON()

    submit_btn = gr.Button("Submit", elem_id="submit-btn")
    clear_btn = gr.Button("Clear", elem_id="clear-btn")

    submit_btn.click(
        fn=vocal_pipeline,
        inputs=[audio_input, lang_choice],
        outputs=[out_transcript, out_lyrics, out_genius, out_google]
    )

    clear_btn.click(
        fn=lambda: ("", "", [], []),
        inputs=None,
        outputs=[out_transcript, out_lyrics, out_genius, out_google]
    )


recognizer = sr.Recognizer()

# ============================
# 🌍 LANGUAGE MAP
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
    if "organic" in data:
        for item in data["organic"][:5]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")
            })

    return results


# ---------------------------
# PIPELINE COMPLET (ROBUSTE)
# ---------------------------
def vocal_pipeline(audio, language):

    if audio is None:
        return "Pas d'audio", "", [], []

    # Compatible toutes versions Gradio
    if isinstance(audio, dict):
        sample_rate = audio.get("sample_rate", 44100)
        data = audio.get("data", None)
        if data is None:
            return "Audio invalide", "", [], []
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
        return f"Erreur transcription : {e}", "", [], []

    # 2. NLP
    lyrics_fragment = agent_nlp(text)["lyrics_fragment"]

    # 3. Double Search
    genius_results = search_genius(lyrics_fragment)
    google_results = search_google(lyrics_fragment)

    return text, lyrics_fragment, genius_results, google_results


# ============================
# 🎨 CSS VIOLET/NOIR SANS VIDEO
# ============================
custom_css = """
* {
    font-family: "Inter", sans-serif !important;
}

body {
    background: radial-gradient(circle at top, #1a0030, #000000) !important;
}
.gradio-container {
    background: transparent !important;
    color: white !important;
}
#landing {
    max-width: 820px;
    margin: auto;
    padding: 40px 20px 20px;
    text-align: center;
    background: rgba(0,0,0,0.45);
    border: 1px solid rgba(130,60,255,0.9);
    border-radius: 26px;
    box-shadow: 0 0 35px rgba(140,0,255,0.35);
    backdrop-filter: blur(6px);
}
#logo-row img {
    width: 64px;
    height: 64px;
    filter: drop-shadow(0 0 12px rgba(120,0,255,0.8));
}
#app-title {
    font-size: 52px;
    font-weight: 900;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #49a6ff, #b66bff, #ff4fc6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#enter-btn {
    margin-top: 18px;
    height: 56px !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    border-radius: 16px !important;
    background: linear-gradient(90deg, #2b7cff, #9b3efc, #ff3eb5) !important;
    color: white !important;
}
.neon-box {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(160,80,255,0.9) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 18px rgba(140,0,255,0.25);
    backdrop-filter: blur(6px);
}
textarea, input, select {
    background: rgba(0,0,0,0.65) !important;
    color: white !important;
    border-radius: 10px !important;
}
#submit-btn {
    background: linear-gradient(90deg, #2b7cff, #9b3efc, #ff3eb5) !important;
    color: white !important;
    height: 55px !important;
    border-radius: 14px !important;
    font-size: 18px !important;
    font-weight: 900 !important;
}
#clear-btn, #back-btn {
    background: #444 !important;
    color: white !important;
    height: 48px !important;
    border-radius: 14px !important;
}
"""


# ============================
# 🧠 INTERFACE FINAL
# ============================
with gr.Blocks(css=custom_css) as demo:

    # ====== LANDING PAGE ======
    with gr.Column(visible=True, elem_id="landing") as landing_col:
        gr.HTML("""
        <div id="logo-row">
            <img src="logo.png" />
            <div id="app-title">SoundFinder</div>
        </div>
        """)

        gr.Image("music.png", show_label=False)

        enter_btn = gr.Button("🎧 Enter App", elem_id="enter-btn")

    # ====== APP PAGE ======
    with gr.Column(visible=False) as app_col:

        with gr.Column(elem_classes="neon-box"):
            audio_input = gr.Audio(type="numpy", label="🎤 Enregistrez / Upload ton audio")
            lang_choice = gr.Dropdown(
                choices=list(language_map.keys()),
                value="Français",
                label="Choisissez la langue"
            )

        with gr.Accordion("Transcription", open=True, elem_classes="neon-box"):
            out_transcript = gr.Textbox(lines=4)

        with gr.Accordion("Paroles extraites", open=False, elem_classes="neon-box"):
            out_lyrics = gr.Textbox(lines=3)

        with gr.Accordion("Résultats Genius", open=False, elem_classes="neon-box"):
            out_genius = gr.JSON()

        with gr.Accordion("Résultats Google", open=False, elem_classes="neon-box"):
            out_google = gr.JSON()

        submit_btn = gr.Button("Submit", elem_id="submit-btn")
        clear_btn = gr.Button("Clear", elem_id="clear-btn")
        back_btn  = gr.Button("⬅ Back to Landing", elem_id="back-btn")

        submit_btn.click(
            fn=vocal_pipeline,
            inputs=[audio_input, lang_choice],
            outputs=[out_transcript, out_lyrics, out_genius, out_google]
        )

        clear_btn.click(
            fn=lambda: ("", "", [], []),
            outputs=[out_transcript, out_lyrics, out_genius, out_google]
        )

    # ====== NAVIGATION BUTTONS
    enter_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                    outputs=[landing_col, app_col])

    back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                   outputs=[landing_col, app_col])

demo.launch()

