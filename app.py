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

        # 4. FINAL DECISION
        final_song = decide_final_title_llm(lyrics_fragment, genius_results, google_results)
        final_title = f"{final_song['title']} by {final_song['artist']}"
        youtube_md = f"[Watch on YouTube]({youtube_link})" if youtube_link else "No YouTube link found"

        return text, lyrics_fragment, genius_results, google_results, final_title, youtube_md
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n=== TRACEBACK COMPLET ===\n")
        print(tb)
        return f"Erreur interne : {e}\n\nTraceback :\n{tb}", "", [], [], "Aucune correspondance trouvée", ""


# ---------------------------
# CSS
# ---------------------------
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
logo_base64 = img_to_base64("images/logo_vinyle_cover.png")
vinyle_base64 = img_to_base64("images/vinyle.png")

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
        transform: translateX(-50px);
    }
    to { 
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes vinyleEntry {
    0% { 
        opacity: 0; 
        transform: translate(-50%, -50%) scale(0.5);
    }
    50% { 
        opacity: 1; 
        transform: translate(-50%, -50%) scale(1);
    }
    100% { 
        opacity: 1; 
        transform: translate(180%, -50%) scale(1);
    }
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
#landing-welcome{
    font-size: 20px !important;
    height: 50px;
}
#landing-container {
    position: relative;
    width: 600px;
    height: 250px;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
#landing-text {
    width: 200px;
    text-align: right;
    font-size: 16px !important;
    position: absolute;
    left: 0;
    color: white;
    opacity: 0;
    animation: fadeInText 1s forwards 0.5s;
}
#landing-text hr {
    border: 1px solid #4F46E5;
    width: 90px;
    margin: 10px 0;
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
#landing-vinyle {
    width: 188px;
    height: 188px;
    border-radius: 50%;
    position: absolute;
    left: 52%;
    transform: translate(-50%, -50%) scale(0.5);
    opacity: 0;
    z-index: 1;
    animation: vinyleEntry 1.5s ease forwards 0.5s, spin 10s linear infinite;
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

        with gr.Column(visible=True, elem_classes="landing-welcome-container"):
            gr.HTML("""
                <div id="landing-welcome">
                    <div id="landing-text">
                        <p>Welcome!</p>
                    </div>
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
                    <img id="landing-logo" src="data:image/png;base64,{logo_base64}" />
                    <img id="landing-vinyle" src="data:image/png;base64,{vinyle_base64}" />
                </div>
            """)

        with gr.Column(visible=True, elem_classes="landing-enter-btn"):
            enter_btn = gr.Button("🎧 Enter App", elem_id="enter-btn")

    # ====== APP PAGE ======
    with gr.Column(visible=False) as app_col:

        with gr.Column(elem_classes="header"):
            gr.Image("images/logo.png", elem_id="app-logo", show_label=False)
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
            gr.Image("images/vinyle.png", elem_id="vinyle", show_label=False)
            out_final_title = gr.Textbox(lines=2, show_label=False)
            out_youtube = gr.Markdown(label="YouTubes Link")


        with gr.Column(visible=True, elem_classes="back_clear_btn"):
          back_btn  = gr.Button("Back to Landing", elem_id="back-btn")
          clear_btn = gr.Button("Clear", elem_id="clear-btn")

        submit_btn.click(
            fn=vocal_pipeline,
            inputs=[audio_input, lang_choice],
            outputs=[out_transcript, out_lyrics, out_genius, out_google, out_final_title, out_youtube]
        )

        clear_btn.click(
            fn=lambda: ("", "", [], [], "", ""),
            outputs=[out_transcript, out_lyrics, out_genius, out_google, out_final_title, out_youtube]
        )

    # ====== NAVIGATION BUTTONS
    enter_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                    outputs=[landing_col, app_col])

    back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                   outputs=[landing_col, app_col])

demo.launch()
