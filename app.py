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

# load_dotenv()

# GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")


GENIUS_API_KEY = "m17Yir9x2IJlr2NIISUD6ge8O2gJ26kVLx72rMajYdGVtm5Jr-1L0shJmww1SChw"
SERPER_API_KEY = "c9defe3057230b55f73bf0095972b9dd8410cc13"

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
# AGENT FINAL : DECISION
# ---------------------------
def decide_final_title(genius_results, google_results):
    title_counts = {}

    # Genius (poids 1)
    for item in genius_results or []:
        title = item.get("title")
        if title:
            title = normalize_title(title)
            title_counts[title] = title_counts.get(title, 0) + 1

    # Google (poids 2)
    for item in google_results or []:
        title = item.get("title")
        snippet = item.get("snippet", "")
        if snippet:
            m = re.search(r"(.+?) lyrics", snippet.lower())
            if m: title = m.group(1)
        if title:
            title = normalize_title(title)
            title_counts[title] = title_counts.get(title, 0) + 2

    if not title_counts:
        return "Aucune correspondance trouvée"

    return max(title_counts, key=title_counts.get)


def normalize_title(title):
    title = title.lower()
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    return " ".join(title.split()).strip()



# ---------------------------
# PIPELINE
# ---------------------------
def vocal_pipeline(audio, language):

    if audio is None:
        return "Pas d'audio", "", [], [], "Aucune correspondance trouvée"

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
        return f"Erreur transcription : {e}", "", [], [], "Aucune correspondance trouvée"

    # 2. NLP
    lyrics_fragment = agent_nlp(text)["lyrics_fragment"]

    # 3. DOUBLE SEARCH
    genius_results = search_genius(lyrics_fragment)
    google_results = search_google(lyrics_fragment)

    # 4. FINAL DECISION
    final_title = decide_final_title(genius_results, google_results)

    return text, lyrics_fragment, genius_results, google_results, final_title


# ---------------------------
# CSS
# ---------------------------
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
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

#landing{
  margin: auto;
  padding: 50px;
  max-width: 820px;
  display: flex;
  flex-direction: column;
  gap: 30px;
  justify-content: center;
  align-items: center;
  background: rgba(0,0,0,0.45);
  border: 1px solid rgba(79,70,229,0.9);
  border-radius: 26px;
  box-shadow: 0 0 35px rgba(79,70,229,0.5);
  backdrop-filter: blur(6px);
}
.landing-vinyle {
  display: flex !important;
  flex-direction: row !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 15px !important;
  width: auto !important;
  max-width: fit-content !important;
}
#landing-text {
  width: 200px;
  text-align: right;
  font-size: 20px !important;
}
#app-logo-landing,
#app-logo-landing .gr-image,
#app-logo-landing .wrap,
#app-logo-landing .container {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#app-logo-landing img {
  width: 250px;
  height: 250px;
  border-radius: 10px;
  background: transparent !important
}

#vinyle,
#vinyle .gr-image,
#vinyle .wrap,
#vinyle .container {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#vinyle img {
    width: 250px;
    height: 250px;
    background: transparent !important;
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
  width: 180px !important;
  font-size: 20px !important;
  font-weight: 300 !important;
  border-radius: 50px !important;
  background: rgb(79,70,229) !important;
  color: white !important;
}

.header{
  margin: auto;
  padding: 50px;
  width: auto !important;
  max-width: fit-content !important;
  display: flex;
  flex-direction: row;
  gap: 15px;
  justify-content: center;
  align-items: center;
  background: rgba(0,0,0,0.45);
  border: 1px solid rgba(79,70,229,0.9);
  border-radius: 26px;
  box-shadow: 0 0 35px rgba(79,70,229,0.5);
  backdrop-filter: blur(6px);
}
#app-logo,
#app-logo .gr-image,
#app-logo .wrap,
#app-logo .container {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#app-logo img {
  width: 64px;
  height: 64px;
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
#clear-btn{
  background: #505050 !important;
  color: white !important;
  height: 30px !important;
  width: 130px !important;
  font-size: 16px !important;
  border-radius: 50px !important;
  font-weight: 500 !important;
}
#back-btn {
  background: #505050 !important;
  color: white !important;
  height: 30px !important;
  width: 200px !important;
  font-size: 16px !important;
  border-radius: 50px !important;
  font-weight: 500 !important;
}
"""


# ---------------------------
# INTERFACE
# ---------------------------


with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet")) as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    # ====== LANDING PAGE ======

    with gr.Column(visible=True, elem_id="landing") as landing_col:

      with gr.Column(visible=True, elem_classes="landing-vinyle"):
          gr.HTML("""
            <div id=landing-text>
                <p>FIND YOUR <b>SONG</b></p>
                <p>JUST A <b>CLICK</b> AWAY</p>
            </div>
          """)
          gr.Image("logo.jpeg", elem_id="app-logo-landing", show_label=False)

      with gr.Column(visible=True, elem_classes="landing-enter-btn"):
          enter_btn = gr.Button("🎧 Enter App", elem_id="enter-btn")

    # ====== APP PAGE ======
    with gr.Column(visible=False) as app_col:

        with gr.Column(elem_classes="header"):
          gr.Image("logo.jpeg", elem_id="app-logo", show_label=False)
          gr.HTML("""
            <div id=header-text>
                <p>Find a song easily from your voice,</p>
                <p>in any language</p>
            </div>
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

        submit_btn = gr.Button("Submit", elem_id="submit-btn")

        with gr.Accordion("Transcription", open=True, elem_classes="neon-box"):
            out_transcript = gr.Textbox(lines=3, show_label=False)

        with gr.Accordion("Extracted lyrics", open=True, elem_classes="neon-box"):
            out_lyrics = gr.Textbox(lines=3, show_label=False)

        with gr.Accordion("Genius results", open=True, elem_classes="neon-box"):
            out_genius = gr.JSON()

        with gr.Accordion("Google results", open=True, elem_classes="neon-box"):
            out_google = gr.JSON()

        with gr.Column(elem_classes="neon-box"):
            gr.HTML("""
            <div id=final-result-text>
                <p>THE <b>SONG</b> YOU ARE LOOKING FOR</p>
                <p>seems to be ...</p>
            </div>
            """)
            gr.Image("vinyle png.png", elem_id="vinyle", show_label=False)
            out_final_title = gr.Textbox(lines=2, show_label=False)

        with gr.Column(visible=True, elem_classes="back_clear_btn"):
          back_btn  = gr.Button("Back to Landing", elem_id="back-btn")
          clear_btn = gr.Button("Clear", elem_id="clear-btn")

        submit_btn.click(
            fn=vocal_pipeline,
            inputs=[audio_input, lang_choice],
            outputs=[out_transcript, out_lyrics, out_genius, out_google, out_final_title]
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
