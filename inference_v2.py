import os

import numpy as np
import music21 as m21
from tensorflow import keras

from constants import SEQUENCE_LENGTH, SAVE_MODEL_DIR, USE_POS_ENCODING, POS_BINS, USE_MODE_ENCODING, MODE_BINS
from inference_v3_duration import get_mode_id
from preprocess_satb import get_mapping, encode_barlines, encode_normalized_position
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

model_bass = keras.models.load_model(SAVE_MODEL_DIR + "best_model_bass.h5", compile=False)
model_alto = keras.models.load_model(SAVE_MODEL_DIR + "best_model_alto.h5", compile=False)
model_tenor = keras.models.load_model(SAVE_MODEL_DIR + "best_model_tenor.h5", compile=False)

print("✅ Models loaded")

# ======================================================
# MAPPING
# ======================================================

mappings = get_mapping()
value_to_key = {v: k for k, v in mappings.items()}


# ======================================================
# CHUNKING
# ======================================================

def generate_sequence_64(input_string, beats, fermata, barlines, mode=None, pos=None):
    notes = input_string.split()
    beats = beats.split()
    fermata = fermata.split()
    barlines = barlines.split()
    mode = mode.split() if mode is not None else None
    pos = pos.split() if pos is not None else None

    chunks = []
    beat_chunks = []
    fermata_chunks = []
    barline_chunks = []
    mode_chunks = []
    pos_chunks = []

    for i in range(0, len(notes), SEQUENCE_LENGTH):
        n = notes[i:i + SEQUENCE_LENGTH]
        b = beats[i:i + SEQUENCE_LENGTH]
        f = fermata[i:i + SEQUENCE_LENGTH]
        bl = barlines[i:i + SEQUENCE_LENGTH]
        m = mode[i:i + SEQUENCE_LENGTH] if mode is not None else None
        p = pos[i:i + SEQUENCE_LENGTH] if pos is not None else None

        if len(n) < SEQUENCE_LENGTH:
            pad = SEQUENCE_LENGTH - len(n)
            n += ["/"] * pad
            b += ["0"] * pad
            f += ["0"] * pad
            bl += ["0"] * pad
            if m is not None:
                m += ["0"] * pad
            if p is not None:
                p += ["0"] * pad

        chunks.append(n)
        beat_chunks.append(b)
        fermata_chunks.append(f)
        barline_chunks.append(bl)
        if m is not None:
            mode_chunks.append(m)
        if p is not None:
            pos_chunks.append(p)

    if mode is not None and pos is not None:
        return chunks, beat_chunks, fermata_chunks, barline_chunks, mode_chunks, pos_chunks
    if mode is not None:
        return chunks, beat_chunks, fermata_chunks, barline_chunks, mode_chunks
    if pos is not None:
        return chunks, beat_chunks, fermata_chunks, barline_chunks, pos_chunks
    return chunks, beat_chunks, fermata_chunks, barline_chunks

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
    encoded_barlines = encode_barlines(soprano_part)
    encoded_pos = encode_normalized_position(soprano_part) if USE_POS_ENCODING else None

    mode_id = get_mode_id(song)
    encoded_mode = " ".join([str(mode_id)] * len(encoded_notes.split())) if USE_MODE_ENCODING else None

    total_len = len(encoded_notes.split())

    if USE_MODE_ENCODING and USE_POS_ENCODING:
        note_chunks, beat_chunks, fermata_chunks, barline_chunks, mode_chunks, pos_chunks = generate_sequence_64(
            encoded_notes,
            encoded_beats,
            encoded_fermata,
            encoded_barlines,
            encoded_mode,
            encoded_pos
        )
    elif USE_MODE_ENCODING:
        note_chunks, beat_chunks, fermata_chunks, barline_chunks, mode_chunks = generate_sequence_64(
            encoded_notes,
            encoded_beats,
            encoded_fermata,
            encoded_barlines,
            encoded_mode
        )
    elif USE_POS_ENCODING:
        note_chunks, beat_chunks, fermata_chunks, barline_chunks, pos_chunks = generate_sequence_64(
            encoded_notes,
            encoded_beats,
            encoded_fermata,
            encoded_barlines,
            None,
            encoded_pos
        )
    else:
        note_chunks, beat_chunks, fermata_chunks, barline_chunks = generate_sequence_64(
            encoded_notes,
            encoded_beats,
            encoded_fermata,
            encoded_barlines
        )

    all_midi_data = []

    if USE_MODE_ENCODING and USE_POS_ENCODING:
        iterator = zip(note_chunks, beat_chunks, fermata_chunks, barline_chunks, mode_chunks, pos_chunks)
    elif USE_MODE_ENCODING:
        iterator = zip(note_chunks, beat_chunks, fermata_chunks, barline_chunks, mode_chunks)
    elif USE_POS_ENCODING:
        iterator = zip(note_chunks, beat_chunks, fermata_chunks, barline_chunks, pos_chunks)
    else:
        iterator = zip(note_chunks, beat_chunks, fermata_chunks, barline_chunks)

    offset = 0

    for items in iterator:
        if USE_MODE_ENCODING and USE_POS_ENCODING:
            notes, beats, fermata, barlines, mode, pos = items
        elif USE_MODE_ENCODING:
            notes, beats, fermata, barlines, mode = items
        elif USE_POS_ENCODING:
            notes, beats, fermata, barlines, pos = items
        else:
            notes, beats, fermata, barlines = items

        remaining = total_len - offset
        if remaining <= 0:
            break
        chunk_len = SEQUENCE_LENGTH if remaining >= SEQUENCE_LENGTH else remaining

        soprano_ids = np.array([mappings[n] for n in notes]).reshape(1, -1)
        beats_ids = np.array([int(b) for b in beats]).reshape(1, -1)
        fermata_ids = np.array([int(f) for f in fermata]).reshape(1, -1)
        barline_ids = np.array([int(bl) for bl in barlines]).reshape(1, -1)
        if USE_MODE_ENCODING:
            mode_ids = np.array([int(m) for m in mode]).reshape(1, -1)
        else:
            mode_ids = None
        if USE_POS_ENCODING:
            pos_ids = np.array([int(p) for p in pos]).reshape(1, -1)
            # Clamp to valid embedding range [0, POS_BINS - 1] for safety.
            pos_ids = np.clip(pos_ids, 0, POS_BINS - 1)
        else:
            pos_ids = None

        bass_inputs = [soprano_ids, beats_ids, fermata_ids, barline_ids]
        if USE_MODE_ENCODING:
            bass_inputs.append(mode_ids)
        if USE_POS_ENCODING:
            bass_inputs.append(pos_ids)
        bass_pred = model_bass.predict(bass_inputs, verbose=0)
        bass_ids = generate_ids_with_temperature(bass_pred, temperature=-1)

        alto_inputs = [soprano_ids, bass_ids.reshape(1, -1), beats_ids, fermata_ids, barline_ids]
        if USE_MODE_ENCODING:
            alto_inputs.append(mode_ids)
        if USE_POS_ENCODING:
            alto_inputs.append(pos_ids)
        alto_pred = model_alto.predict(alto_inputs, verbose=0)
        alto_ids = generate_ids_with_temperature(alto_pred, temperature=0.3)

        tenor_inputs = [
            soprano_ids,
            bass_ids.reshape(1, -1),
            alto_ids.reshape(1, -1),
            beats_ids,
            fermata_ids,
            barline_ids,
        ]
        if USE_MODE_ENCODING:
            tenor_inputs.append(mode_ids)
        if USE_POS_ENCODING:
            tenor_inputs.append(pos_ids)
        tenor_pred = model_tenor.predict(tenor_inputs, verbose=0)
        tenor_ids = generate_ids_with_temperature(tenor_pred, temperature=0.7)

        print("Bass unique:", np.unique(bass_ids))

        soprano_notes = [value_to_key[i] for i in soprano_ids[0][:chunk_len]]
        bass_notes = [value_to_key[i] for i in bass_ids[:chunk_len]]
        alto_notes = [value_to_key[i] for i in alto_ids[:chunk_len]]
        tenor_notes = [value_to_key[i] for i in tenor_ids[:chunk_len]]

        all_midi_data.append([
            soprano_notes,
            alto_notes,
            tenor_notes,
            bass_notes
        ])
        offset += chunk_len

    return save_melody_4parts(all_midi_data)


def sample_with_temperature(probabilities, temperature):
    """
    probabilities: the output array from model.predict() (the softmax layer)
    temperature: float. 1.0 is default, < 1.0 is conservative, > 1.0 is creative.
    """
    if temperature <= 0:
        return np.argmax(probabilities)

    predictions = np.log(probabilities) / temperature
    exp_preds = np.exp(predictions)
    reweighted_probabilities = exp_preds / np.sum(exp_preds)
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

    valid_extensions = (".mid", ".midi", ".mxl", "musicxml")

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)

        if not os.path.isfile(file_path):
            continue

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


















