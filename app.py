import gradio as gr
import speech_recognition as sr
import tempfile
import scipy.io.wavfile
import numpy as np
import requests
import os
import re
import unicodedata
import base64
import json
import urllib.parse
from dotenv import load_dotenv
from pathlib import Path
from mistralai import Mistral

load_dotenv()

GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# GENIUS_API_KEY = "m17Yir9x2IJlr2NIISUD6ge8O2gJ26kVLx72rMajYdGVtm5Jr-1L0shJmww1SChw"
# SERPER_API_KEY = "c9defe3057230b55f73bf0095972b9dd8410cc13"
# MISTRAL_API_KEY ="WBPRJ3uH9OLYBxUWDe8MKncRV5kUiT7I"

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
    <div style="width:100%; display:flex; justify-content:center;">
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
            return "Pas d'audio", "", [], [], "Aucune correspondance trouvée", ""

        if isinstance(audio, dict):
            sample_rate = audio.get("sample_rate", 44100)
            data = audio.get("data", None)
            if data is None:
                return "Audio invalide", "", [], [], "Aucune correspondance trouvée"
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
            return (f"Erreur transcription : {e}", "", [], [], "Aucune correspondance trouvée", "")

        # 2. NLP
        lyrics_fragment = agent_nlp(text)["lyrics_fragment"]

        # 3. DOUBLE SEARCH
        genius_results = search_genius(lyrics_fragment)
        google_results, youtube_link = search_google(lyrics_fragment)

        # 4. FINAL DECISION (LLM)
        final_song = decide_final_title_llm(lyrics_fragment, genius_results, google_results)
        final_title = f"{final_song['title']} by {final_song['artist']}"

        # 5. YOUTUBE + SPOTIFY LINKS
        youtube_md = f"[Watch on YouTube]({youtube_link})" if youtube_link else "No YouTube link found"
        youtube_html = make_youtube_iframe(youtube_link) if youtube_link else "No YouTube link found"

        spotify_url = build_spotify_search_url(final_song['title'], final_song['artist'])
        spotify_md = f"[Open in Spotify]({spotify_url})" if spotify_url else "No Spotify link found"

        return text, lyrics_fragment, genius_results, google_results, final_title, youtube_md, youtube_html, spotify_md
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n=== TRACEBACK COMPLET ===\n")
        print(tb)
        return f"Erreur interne : {e}\n\nTraceback :\n{tb}", "", [], [], "Aucune correspondance trouvée", "", "", ""


# ---------------------------
# CSS + IMAGES
# ---------------------------
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Images in base64
logo_vinyle_base64 = img_to_base64("images/logo_vinyle_cover.png")
vinyle_base64 = img_to_base64("images/vinyle.png")
logo_base64 = img_to_base64("images/logo.png")
youtube_logo_base64 = img_to_base64("images/logo_youtube.png")
spotify_logo_base64 = img_to_base64("images/logo_spotify.png")

# Custom CSS
custom_css = """
* {
    font-family: "Inter", sans-serif !important;
}

body {
    background: radial-gradient(circle at top, #100E30, #000000) !important;
}
.gradio-container {
    background: transparent !important;
    color: white !important;
}

label{
    background: transparent !important;
    color: white !important;
    font-weight: 500;
    font-size: 16px;
}
textarea, input, select {
    background: transparent !important;
    color: white !important;
    border-radius: 10px !important;
}

@keyframes fadeInText {
    from { 
        opacity: 0;
        transform: translateX(220px);
    }
    to { 
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes vinyleEntry {
    0% {
        transform: translate(-60%, -50%);
    }
    100% {
        transform: translate(3%, -50%);
    }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

@keyframes aurora-1 {
  0% {
    top: 0;
    right: 0;
  }

  50% {
    top: 100%;
    right: 75%;
  }

  75% {
    top: 100%;
    right: 25%;
  }

  100% {
    top: 0;
    right: 0;
  }
}

@keyframes aurora-2 {
  0% {
    top: -50%;
    left: 0%;
  }

  60% {
    top: 100%;
    left: 75%;
  }

  85% {
    top: 100%;
    left: 25%;
  }

  100% {
    top: -50%;
    left: 0%;
  }
}

@keyframes aurora-3 {
  0% {
    bottom: 0;
    left: 0;
  }

  40% {
    bottom: 100%;
    left: 75%;
  }

  65% {
    bottom: 40%;
    left: 50%;
  }

  100% {
    bottom: 0;
    left: 0;
  }
}

@keyframes aurora-4 {
  0% {
    bottom: -50%;
    right: 0;
  }

  50% {
    bottom: 0%;
    right: 40%;
  }

  90% {
    bottom: 50%;
    right: 25%;
  }

  100% {
    bottom: -50%;
    right: 0;
  }
}

@keyframes aurora-border {
  0% {
    border-radius: 37% 29% 27% 27% / 28% 25% 41% 37%;
  }

  25% {
    border-radius: 47% 29% 39% 49% / 61% 19% 66% 26%;
  }

  50% {
    border-radius: 57% 23% 47% 72% / 63% 17% 66% 33%;
  }

  75% {
    border-radius: 28% 49% 29% 100% / 93% 20% 64% 25%;
  }

  100% {
    border-radius: 37% 29% 27% 27% / 28% 25% 41% 37%;
  }
}

#landing{
    margin: auto;
    padding: 20px 50px;
    max-width: 820px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    justify-content: center;
    align-items: center;
    background: rgba(0,0,0,1);
    border: 1px solid rgba(79,70,229,0.9);
    border-radius: 26px;
    box-shadow: 0 0 200px rgba(79,70,229,0.5);
    backdrop-filter: blur(6px);
}

.gr-html, 
.gr-html > .wrap {
    isolation: unset !important;
    background: transparent !important;
}
.aurora-container {
    position: relative;
    width: 100%;
    text-align: center;
    padding: 20px 0 10px 0;
    overflow: hidden;
    display: inline-block;
}
.aurora-title {
    font-size: 48px !important;
    font-weight: 900;
    letter-spacing: clamp(-1.75px, -0.25vw, -3.5px);
    position: relative;
    overflow: hidden;
    background: rgba(0,0,0,1) !important;
    margin: 0;
}
.aurora {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    mix-blend-mode: darken !important;
    pointer-events: none;
}
.aurora__item {
    overflow: hidden;
    position: absolute;
    width: 60vw;
    height: 60vw;
    background-color: #00c2ff;
    border-radius: 37% 29% 27% 27% / 28% 25% 41% 37%;
    filter: blur(1rem);
    mix-blend-mode: overlay;
}

.aurora__item:nth-of-type(1) {
  top: -50%;
  animation: aurora-border 6s ease-in-out infinite,
    aurora-1 12s ease-in-out infinite alternate;
}
.aurora__item:nth-of-type(2) {
  background-color: #ffc640;
  right: 0;
  top: 0;
  animation: aurora-border 6s ease-in-out infinite,
    aurora-2 12s ease-in-out infinite alternate;
}
.aurora__item:nth-of-type(3) {
  background-color: #33ff8c;
  left: 0;
  bottom: 0;
  animation: aurora-border 6s ease-in-out infinite,
    aurora-3 8s ease-in-out infinite alternate;
}
.aurora__item:nth-of-type(4) {
  background-color: #e54cff;
  right: 0;
  bottom: -50%;
  animation: aurora-border 6s ease-in-out infinite,
    aurora-4 24s ease-in-out infinite alternate;
}


#landing-container {
    position: relative;
    width: 100%;
    height: 250px;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    overflow: hidden;
}
#landing-text {
    width: 200px;
    text-align: right;
    font-size: 16px !important;
    position: absolute;
    left: 30px;
    color: white;
    right: 0;
    opacity: 0;
    animation: fadeInText 2s forwards 0.5s;
}
#landing-text hr {
    border: 1px solid #FFFFFF;
    width: 60px;
    margin: 10px 0 10px auto;
}
#landing-logo,
#landing-logo .gr-image,
#landing-logo .wrap,
#landing-logo .container {
    width: 190px;
    height: 190px;
    z-index: 2;
    position: relative;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#landing-vinyle-wrapper {
    position: absolute;
    left: 52%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 188px;
    height: 188px;
    animation: vinyleEntry 2s ease forwards 0.5s;
}
#landing-vinyle {
    width: 188px;
    height: 188px;
    border-radius: 50%;
    animation: spin 10s linear infinite;
}

.landing-enter-btn {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
#enter-btn {
    height: 50px !important;
    width: 200px !important;
    font-size: 20px !important;
    font-weight: 300 !important;
    border-radius: 50px !important;
    background: rgb(79,70,229) !important;
    color: white !important;
}
#enter-btn:hover {
    background: transparent !important;
    color: rgb(255,255,255) !important;
    border: 1px solid #4F46E5 !important;
}

#main-page{
    padding: 5px 100px 10px 100px;
    justify-content: center;
    align-items: center;
}

.header-container{
    width: 900px;
    background: rgba(0,0,0,0.45);
    border: 1px solid rgba(79,70,229,0.9);
    border-radius: 50px;
    box-shadow: 0 0 35px rgba(79,70,229,0.5);
    backdrop-filter: blur(6px);
}

.header{
    height: 70px;
    width: 900px;
    padding: 0px 20px;
    display: flex;
    flex-direction: row;
    gap: 15px;
    justify-content: flex-start;
    align-items: center;
}
#app-logo {
    width: 80px;
    height: 80px;
    background: transparent !important;
}
#header-text {
    max-width: 500px;
    color: white;
    font-size: 16px !important;
}

.neon-box {
    margin: 15px;
    padding: 20px;
    background: transparent !important;
    border: 1px solid rgba(79,70,229,0.9) !important;
    border-radius: 25px !important;
    box-shadow: 0 0 18px rgba(79,70,229,0.5);
    backdrop-filter: blur(6px);
}
#submit-btn {
    background: linear-gradient(90deg, #4F46E5, #DE1A9C) !important;
    color: white !important;
    height: 30px !important;
    width: 130px !important;
    border-radius: 50px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}
#submit-btn:hover {
    background: transparent !important;
    color: rgb(255,255,255) !important;
    border: 1px solid #4F46E5 !important;
}

.final-result-container {
    margin: 15px;
    padding: 20px;
    background: transparent !important;
    border: 1px solid rgba(79,70,229,0.9) !important;
    border-radius: 25px !important;
    box-shadow: 0 0 18px rgba(79,70,229,0.5);
    backdrop-filter: blur(6px);
    overflow: hidden;
}

#vinyle-and-title {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 40px !important;
    overflow: hidden;
}
.vinyle-container {
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
}
#vinyle {
    width: 200px;
    height: 200px;
    overflow: hidden;
    background: transparent !important;
    border-radius: 50%;
    animation: spin 10s linear infinite;
}
#final-title-box {
    max-width: 350px !important;
    text-align: center !important;
}

.back_clear_btn{
    display: flex !important;
    flex-direction: row !important;
    justify-content: center !important;
    align-items: center !important;
    align-content: center !important;
    gap: 15px !important;
    width: auto !important;
    max-width: fit-content !important;
}
#back-btn {
    background: #505050 !important;
    color: white !important;
    height: 30px !important;
    width: 130px !important;
    font-size: 16px !important;
    border-radius: 50px !important;
    font-weight: 500 !important;
}
#back-btn:hover {
    background: transparent !important;
    color: rgb(255,255,255) !important;
    border: 1px solid #505050 !important;
}
#clear-btn{
    background: #505050 !important;
    color: white !important;
    height: 30px !important;
    width: 130px !important;
    font-size: 16px !important;
    border-radius: 50px !important;
    font-weight: 500 !important;
}
#clear-btn:hover {
    background: transparent !important;
    color: rgb(255,255,255) !important;
    border: 1px solid #505050 !important;
}
"""


# ---------------------------
# INTERFACE
# ---------------------------


with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet")) as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    # ====== LANDING PAGE ======

    with gr.Column(visible=True, elem_id="landing") as landing_col:

        with gr.Column(visible=True, elem_classes="landing-welcome-container"):
            gr.HTML("""
            <div class="aurora-container">
                <h1 class="aurora-title">
                    Welcome on <b>SoundFinder</b>
                    <div class="aurora">
                        <div class="aurora__item"></div>
                        <div class="aurora__item"></div>
                        <div class="aurora__item"></div>
                        <div class="aurora__item"></div>
                    </div>
                </h1>
            </div>
            """)

        with gr.Column(visible=True, elem_classes="landing-vinyle-container"):
            gr.HTML(f"""
                <div id="landing-container">
                    <div id="landing-text">
                        <p>FIND YOUR <b>SONG</b></p>
                        <hr>
                        <p>JUST A <b>CLICK</b> AWAY</p>
                    </div>
                    <img id="landing-logo" src="data:image/png;base64,{logo_vinyle_base64}" />
                    <div id="landing-vinyle-wrapper">
                        <img id="landing-vinyle" src="data:image/png;base64,{vinyle_base64}" />
                    </div>
                </div>
            """)

        with gr.Column(visible=True, elem_classes="landing-enter-btn"):
            enter_btn = gr.Button("🎧 Enter App →", elem_id="enter-btn")

    # ====== APP PAGE ======
    with gr.Column(visible=False, elem_id="main-page") as app_col:

        with gr.Row(elem_classes="header-container"):
            gr.HTML(f"""
                <div class="header">
                    <div id="app-logo">
                        <img src="data:image/png;base64,{logo_base64}" />
                    </div>
                    <div id=header-text>
                        <p>Find a song easily from your voice, in any language</p>
                    </div>
                </class>
            """)

        with gr.Column(elem_classes="neon-box"):
            audio_input = gr.Audio(type="numpy", label="🎤 Drop your audio or record your voice")

        with gr.Column(elem_classes="neon-box"):
            gr.HTML("""
              <div id=language-text>
                  <p>Choose the language</p>
              </div>
              """)
            lang_choice = gr.Dropdown(
                choices=list(language_map.keys()),
                value="Français",
                show_label=False
            )

        submit_btn = gr.Button("Search", elem_id="submit-btn")

        with gr.Accordion("Transcription", open=True, elem_classes="neon-box", visible=False):
            out_transcript = gr.Textbox(lines=3, show_label=False)

        with gr.Accordion("Extracted lyrics", open=True, elem_classes="neon-box", visible=False):
            out_lyrics = gr.Textbox(lines=3, show_label=False)

        with gr.Accordion("Genius results", open=True, elem_classes="neon-box", visible=False):
            out_genius = gr.JSON()

        with gr.Accordion("Google results", open=True, elem_classes="neon-box", visible=False):
            out_google = gr.JSON()

        with gr.Column(elem_classes="final-result-container"):
            gr.HTML("""
            <div id=final-result-text>
                <p>THE <b>SONG</b> YOU ARE LOOKING FOR</p>
                <p>seems to be ...</p>
            </div>
            """)
            with gr.Row(elem_id="vinyle-and-title"):
                gr.HTML(f"""
                    <div class="vinyle-container">
                        <img id="vinyle" src="data:image/png;base64,{vinyle_base64}" />
                    </div>
                """)
                out_final_title = gr.Textbox(lines=2, show_label=False, elem_id="final-title-box")
            out_youtube = gr.Markdown(label="YouTubes Link")

        with gr.Accordion("", open=True, elem_classes="neon-box"):
           gr.HTML(f"""
                <div style="display:flex;align-items:center;gap:15px;margin-bottom:12px;">
                    <img src="data:image/png;base64,{youtube_logo_base64}"
                        alt="YouTube"
                        style="width:120px;height:auto;object-fit:contain;">
                </div>
            """)
           out_youtube_video = gr.HTML()

        with gr.Accordion("", open=True, elem_classes="neon-box"):
            gr.HTML(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <img src="data:image/jpeg;base64,{spotify_logo_base64}"
                        alt="Spotify"
                        style="width:56px;height:56px;">
                    <h3 style="margin:0;font-size:22px;">Spotify</h3>
                </div>
            """)

            out_spotify = gr.Markdown(label="")
            
        with gr.Column(visible=True, elem_classes="back_clear_btn"):
          back_btn  = gr.Button("← Back", elem_id="back-btn")
          clear_btn = gr.Button("Clear", elem_id="clear-btn")

        submit_btn.click(
            fn=vocal_pipeline,
            inputs=[audio_input, lang_choice],
            outputs=[out_transcript, out_lyrics, out_genius, out_google, out_final_title, out_youtube, out_youtube_video, out_spotify]
        )

        clear_btn.click(
            fn=lambda: ("", "", [], [], "", "", "", ""),
            outputs=[out_transcript, out_lyrics, out_genius, out_google, out_final_title, out_youtube, out_youtube_video, out_spotify]
        )

    # ====== NAVIGATION BUTTONS
    enter_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                    outputs=[landing_col, app_col])

    back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                   outputs=[landing_col, app_col])

demo.launch(share=True)
