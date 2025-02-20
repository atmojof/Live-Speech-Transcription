import streamlit as st
import os
from faster_whisper import WhisperModel

# Title of the app
st.title("🗣️ Live Recording Transcription")

# Sidebar: Instructions, settings, and notes
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1. Record audio using the recorder.
        2. The system will record and transcribe the audio.
        3. The transcription can be summarized for easier understanding.
        """
    )
    st.header("Settings")
    st.markdown("- **Transcription Model**: Faster Whisper (small)")
    st.header("Notes")
    st.markdown(
        "Try saving the transcription and then summarizing it using **ChatGPT** with the following prompt: \n\n"
        "Summarize the following text (UPLOAD YOUR TXT FILE)."
    )

# Initialize session state for the transcription result
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = ""

# Transcription function using Faster Whisper
def transcribe_single_whisper(audio_path, lang="en", model=None):
    """
    Transcribes a single audio file using the Faster Whisper small model.
    """
    if model is None:
        # Load the small model on CPU with float32 precision
        model = WhisperModel("small", device="cpu", compute_type="float32")
    segments, _ = model.transcribe(audio_path, language=lang)
    transcription = ""
    for segment in segments:
        start_time = segment.start
        text = segment.text
        transcription += f"[{int(start_time // 60):02}:{int(start_time % 60):02}] {text}\n"
    return transcription.strip()

# Record audio using the built-in widget (requires a recent version of Streamlit)
audio_file_obj = st.audio_input("Record a voice message")

if audio_file_obj is not None:
    # Play the recorded audio in the frontend
    st.audio(audio_file_obj, format="audio/wav")

    # Save the recorded audio bytes as a WAV file
    audio_file_path = "audio.wav"
    audio_bytes = audio_file_obj.read()
    with open(audio_file_path, "wb") as f:
        f.write(audio_bytes)

    # Transcribe the saved audio using Faster Whisper
    with st.spinner("Transcribing audio, please wait..."):
        st.session_state.transcription_result = transcribe_single_whisper(audio_file_path)
    st.success("Transcription complete!", icon="✅")

    # If no transcription was detected, update the result accordingly
    if not st.session_state.transcription_result:
        st.session_state.transcription_result = "No voice detected! Please try speaking louder."

    # Display the transcription in a text area
    st.subheader("Transcription")
    st.text_area("Transcribed Text", value=st.session_state.transcription_result, height=300)

    # Add a download button for the transcription result
    st.download_button(
        label="Download Transcription",
        data=st.session_state.transcription_result,
        file_name="transcription.txt",
        mime="text/plain",
    )
