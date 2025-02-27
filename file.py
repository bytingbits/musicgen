import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pretty_midi
from typing import Optional, Dict
import collections
import io

# Custom loss function required for loading the model
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

# Load model with custom objects
@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure}
    )

# Note generation functions from notebook
def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs, verbose=0)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    return int(pitch), float(step), float(duration)

def notes_to_midi(notes: pd.DataFrame, instrument_name: str, velocity: int = 20) -> bytes:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end
        )
        instrument.notes.append(midi_note)
        prev_start = start

    pm.instruments.append(instrument)
    midi_bytes = io.BytesIO()
    pm.write(midi_bytes)
    return midi_bytes.getvalue()

# Streamlit UI
st.title("AI Music Composer 🎹")

# Sidebar controls
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 2.0, 1.0)
num_notes = st.sidebar.slider("Number of Notes", 50, 500, 120)
#instrument = st.sidebar.selectbox("Instrument", pretty_midi.constants.INSTRUMENT_MAP.keys())
instrument_names = [inst for inst in pretty_midi.constants.INSTRUMENT_MAP]
instrument = st.sidebar.selectbox("Instrument", instrument_names)
# Load model
model = load_model("full_model.h5")  # Update path if needed

# Initial sequence (from notebook example)
key_order = ['pitch', 'step', 'duration']
sample_notes = np.array([[70, 0.5, 0.5], [72, 0.5, 0.5], [74, 0.5, 0.5]] * 8)  # Example starting sequence

# After model loading in Streamlit app:
seq_length = 25  # Should match notebook's seq_length
vocab_size = 128  # Should match notebook's vocab_size

# Initialize with proper sequence length
sample_notes = np.array([[70, 0.5, 0.5], [72, 0.5, 0.5], [74, 0.5, 0.5]] * 9)[:seq_length]
input_notes = sample_notes / np.array([vocab_size, 1, 1])  # Normalize pitch

if st.button("Generate Music"):
    with st.spinner("Composing your masterpiece..."):
        try:
            # Prepare input sequence
            vocab_size = 128
            input_notes = sample_notes[:25] / np.array([vocab_size, 1, 1])
            
            # Generate notes
            generated_notes = []
            prev_start = 0
            for _ in range(num_notes):
                pitch, step, duration = predict_next_note(input_notes, model, temperature)
                start = prev_start + step
                end = start + duration
                input_note = (pitch, step, duration)
                generated_notes.append((*input_note, start, end))
                input_notes = np.delete(input_notes, 0, axis=0)
                input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
                prev_start = start
                input_notes = np.concatenate([input_notes[1:], np.expand_dims(input_note, 0)])

            # Create DataFrame and MIDI
            generated_df = pd.DataFrame(
                generated_notes, columns=(*key_order, 'start', 'end'))
            midi_bytes = notes_to_midi(generated_df, instrument_name=instrument)

            # Display and download
            st.success("🎵 Composition Complete!")
            st.audio(midi_bytes, format="audio/midi")
            
            st.download_button(
                label="Download MIDI",
                data=midi_bytes,
                file_name="composition.mid",
                mime="audio/midi"
            )
            
        except Exception as e:
            st.error(f"Error generating music: {e}")
