# ---------------------------
# INTERFACE
# ---------------------------

import gradio as gr
from backend import vocal_pipeline, language_map
import base64

# Function : Image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Application

def create_app():
    # Images in base64
    logo_vinyle_base64 = img_to_base64("../images/logo_vinyle_cover.png")
    vinyle_base64 = img_to_base64("../images/vinyle.png")
    logo_base64 = img_to_base64("../images/logo.png")
    youtube_logo_base64 = img_to_base64("../images/logo_youtube.png")
    spotify_logo_base64 = img_to_base64("../images/logo_spotify.png")

    # Import CSS
    with open("styles.css") as f:
        custom_css = f"<style>{f.read()}</style>"

    # UI
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet")) as demo:
        # gr.HTML(f"<style>{custom_css}</style>")
        gr.HTML(custom_css)

        # ====== LANDING PAGE ======

        with gr.Column(visible=True, elem_id="landing") as landing_col:

            # --- WELCOME MESSAGE ---
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

            # --- VINYLE ANIMATION + TEXT ---
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

            # --- ENTER APP BUTTON ---
            with gr.Column(visible=True, elem_classes="landing-enter-btn"):
                enter_btn = gr.Button("🎧 Enter App →", elem_id="enter-btn")

        # ====== APP PAGE ======
        with gr.Column(visible=False, elem_id="main-page") as app_col:

            # --- HEADER ---
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

            # --- AUDIO INPUT + LANGUAGE SELECTION ---
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

            # --- SUBMIT (search) BUTTON ---
            submit_btn = gr.Button("Search", elem_id="submit-btn")

            # --- OUTPUTS (not visible for the user) ---
            with gr.Accordion("Transcription", open=True, elem_classes="neon-box", visible=False):
                out_transcript = gr.Textbox(lines=3, show_label=False)

            with gr.Accordion("Extracted lyrics", open=True, elem_classes="neon-box", visible=False):
                out_lyrics = gr.Textbox(lines=3, show_label=False)

            with gr.Accordion("Genius results", open=True, elem_classes="neon-box", visible=False):
                out_genius = gr.JSON()

            with gr.Accordion("Google results", open=True, elem_classes="neon-box", visible=False):
                out_google = gr.JSON()

            # --- FINAL RESULT ---
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
                    with gr.Column():
                        out_final_title = gr.HTML(elem_id="final-title-box")
                        out_youtube = gr.Markdown(label="YouTubes Link")

            # --- YOUTUBE IFRAME + SPOTIFY LINK ---
            with gr.Row():
                with gr.Column(elem_classes="youtube-iframe-container"):
                    gr.HTML(f"""
                            <div>
                                <img src="data:image/png;base64,{youtube_logo_base64}"
                                    alt="YouTube"
                                    style="width:100px;height:auto;object-fit:contain">
                            </div>
                        """)
                    out_youtube_video = gr.HTML()

                with gr.Column(elem_classes="spotify-link-container"):
                    gr.HTML(f"""
                        <div>
                            <img src="data:image/jpeg;base64,{spotify_logo_base64}"
                                alt="Spotify"
                                style="width:100px;height:auto;text-align:center">
                        </div>
                    """)
                    out_spotify = gr.Markdown(label="")
        
            # --- BACK + CLEAR BUTTONS ---    
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

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
