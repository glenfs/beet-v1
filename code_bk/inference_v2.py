import os

import numpy as np
import music21 as m21
from tensorflow import keras

from constants import SEQUENCE_LENGTH, SAVE_MODEL_DIR
from preprocess_satb import get_mapping
from rubato_midi_utils.utils import (
    transpose,
    encode_song,
    encode_beats,
    add_fermata,
    save_melody_4parts
)

# ======================================================
# LOAD MODELS
# ======================================================

model_bass = keras.models.load_model(SAVE_MODEL_DIR + "best_model_bass.h5")
model_alto = keras.models.load_model(SAVE_MODEL_DIR + "best_model_alto.h5")
model_tenor = keras.models.load_model(SAVE_MODEL_DIR + "best_model_tenor.h5")

print("✅ Models loaded")

# ======================================================
# MAPPING
# ======================================================

mappings = get_mapping()
value_to_key = {v: k for k, v in mappings.items()}


# ======================================================
# CHUNKING
# ======================================================

def generate_sequence_64(input_string, beats, fermata):
    notes = input_string.split()
    beats = beats.split()
    fermata = fermata.split()

    chunks = []
    beat_chunks = []
    fermata_chunks = []

    for i in range(0, len(notes), SEQUENCE_LENGTH):

        n = notes[i:i + SEQUENCE_LENGTH]
        b = beats[i:i + SEQUENCE_LENGTH]
        f = fermata[i:i + SEQUENCE_LENGTH]

        if len(n) < SEQUENCE_LENGTH:
            pad = SEQUENCE_LENGTH - len(n)
            n += ["/"] * pad
            b += ["0"] * pad
            f += ["0"] * pad

        chunks.append(n)
        beat_chunks.append(b)
        fermata_chunks.append(f)

    return chunks, beat_chunks, fermata_chunks


# ======================================================
# INFERENCE
# ======================================================

def generate_parts(midi_file):
    song = m21.converter.parse(midi_file)
    soprano_part = song.parts[0]

    soprano_part = transpose(soprano_part)

    encoded_notes = encode_song(soprano_part)
    encoded_beats = encode_beats(soprano_part)
    encoded_fermata = add_fermata(encoded_notes)

    note_chunks, beat_chunks, fermata_chunks = generate_sequence_64(
        encoded_notes,
        encoded_beats,
        encoded_fermata
    )

    all_midi_data = []

    for notes, beats, fermata in zip(note_chunks, beat_chunks, fermata_chunks):
        soprano_ids = np.array([mappings[n] for n in notes]).reshape(1, -1)
        beats_ids = np.array([int(b) for b in beats]).reshape(1, -1)
        fermata_ids = np.array([int(f) for f in fermata]).reshape(1, -1)

        # ========= BASS =========
        # 1. Get raw probabilities from the model
        bass_pred = model_bass.predict(
            [soprano_ids, beats_ids, fermata_ids],
            verbose=0
        )
        # bass_ids = np.argmax(bass_pred[0], axis=-1)

        # Keep Bass solid/grounded with lower temp
        bass_ids = generate_ids_with_temperature(bass_pred, temperature=0.2)

        # ========= ALTO =========
        alto_pred = model_alto.predict(
            [soprano_ids, bass_ids.reshape(1, -1), beats_ids, fermata_ids],
            verbose=0
        )
        # alto_ids = np.argmax(alto_pred[0], axis=-1)

        # Let Alto be a bit more melodic/creative
        alto_ids = generate_ids_with_temperature(alto_pred, temperature=0.2)

        # ========= TENOR =========
        tenor_pred = model_tenor.predict(
            [soprano_ids, bass_ids.reshape(1, -1), alto_ids.reshape(1, -1), beats_ids, fermata_ids],
            verbose=0
        )
        # tenor_ids = np.argmax(tenor_pred[0], axis=-1)
        tenor_ids = generate_ids_with_temperature(tenor_pred, temperature=0.2)

        print("Bass unique:", np.unique(bass_ids))

        soprano_notes = [value_to_key[i] for i in soprano_ids[0]]
        bass_notes = [value_to_key[i] for i in bass_ids]
        alto_notes = [value_to_key[i] for i in alto_ids]
        tenor_notes = [value_to_key[i] for i in tenor_ids]

        all_midi_data.append([
            soprano_notes,
            alto_notes,
            tenor_notes,
            bass_notes
        ])

    return save_melody_4parts(all_midi_data)


import numpy as np


def sample_with_temperature(probabilities, temperature):
    """
    probabilities: the output array from model.predict() (the softmax layer)
    temperature: float. 1.0 is default, < 1.0 is conservative, > 1.0 is creative.
    """
    if temperature <= 0:
        return np.argmax(probabilities)

    # 1. Apply temperature sampling for each note in the sequence
    # 1. Apply temperature to the probabilities
    # We use log to turn probabilities back into 'logits' effectively
    predictions = np.log(probabilities) / temperature

    # 2. Re-apply softmax to get new probabilities
    exp_preds = np.exp(predictions)
    reweighted_probabilities = exp_preds / np.sum(exp_preds)

    # 3. Sample from the new distribution
    choices = range(len(probabilities))
    return np.random.choice(choices, p=reweighted_probabilities)


def generate_ids_with_temperature(model_prediction, temperature=0.85):
    """
    Processes model output through temperature sampling.

    Args:
        model_prediction: The raw output from model.predict()
        temperature: Float (0.1 to 1.5). Controls creativity.

    Returns:
        numpy array of sampled IDs.
    """
    sampled_ids = []

    # model_prediction[0] is the sequence of probability distributions
    # for the first (and usually only) song in the batch
    for note_probs in model_prediction[0]:
        sampled_id = sample_with_temperature(note_probs, temperature)
        sampled_ids.append(sampled_id)

    return np.array(sampled_ids)


# ======================================================
# MAIN
# ======================================================

def main():
    INPUT_DIR = "data/input2"

    print("🎼 Starting SATB batch inference...\n")

    # Supported file extensions
    valid_extensions = (".mid", ".midi", ".mxl", "musicxml")

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)

        # Skip if not a file
        if not os.path.isfile(file_path):
            continue

        # Skip unsupported formats
        if not filename.lower().endswith(valid_extensions):
            continue

        print(f"Processing: {filename}")

        try:
            output_file = generate_parts(file_path)
            print(f"✅ Done! Saved to: {output_file}\n")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}\n")

    print("🎵 Batch processing complete.")


if __name__ == "__main__":
    main()
