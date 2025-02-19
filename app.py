import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
from faster_whisper import WhisperModel
import queue
import threading

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = queue.Queue()
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""

# Load the Whisper model
@st.cache_resource
def load_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

# Audio callback function
def audio_callback(indata, frames, time, status):
    if st.session_state.recording:
        st.session_state.audio_buffer.put(indata.copy())

# Function to process audio and transcribe
def transcribe_audio(model):
    while st.session_state.recording or not st.session_state.audio_buffer.empty():
        try:
            # Get audio chunks from the buffer
            audio_chunk = st.session_state.audio_buffer.get(timeout=1)
            audio_array = np.concatenate(audio_chunk, axis=0)

            # Convert to WAV bytes
            wav_io = io.BytesIO()
            write(wav_io, 16000, (audio_array * 32767).astype(np.int16))  # Scale to int16
            wav_io.seek(0)

            # Transcribe the audio chunk
            segments, _ = model.transcribe(wav_io, beam_size=5)
            for segment in segments:
                st.session_state.transcription += segment.text + " "

        except queue.Empty:
            continue

# Streamlit app
st.title("Live Speech-to-Text with Faster-Whisper")

# Load the model
model = load_model()

# Single toggle button
if st.button("Start Recording" if not st.session_state.recording else "Stop Recording"):
    if not st.session_state.recording:
        # Start recording
        st.session_state.recording = True
        st.session_state.audio_buffer = queue.Queue()  # Clear previous buffer
        st.session_state.transcription = ""  # Clear previous transcription
        st.session_state.stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype='float32',
            callback=audio_callback
        )
        st.session_state.stream.start()

        # Start transcription thread
        threading.Thread(target=transcribe_audio, args=(model,), daemon=True).start()
    else:
        # Stop recording
        st.session_state.recording = False
        st.session_state.stream.stop()
        st.session_state.stream.close()
        del st.session_state.stream

# Display current recording status
if st.session_state.recording:
    st.warning("Recording in progress... Speak now!")
else:
    st.info("Not recording. Click the button to start.")

# Display live transcription
st.subheader("Live Transcription:")
transcription_placeholder = st.empty()
while st.session_state.recording:
    transcription_placeholder.markdown(f"**Transcription:** {st.session_state.transcription}")

# Display final transcription
st.subheader("Final Transcription:")
st.write(st.session_state.transcription)

# Instructions
st.markdown("""
**Instructions:**
1. Click the button to start recording.
2. Speak clearly into your microphone.
3. Click the button again to stop recording.
4. The transcription will appear live as you speak.
""")