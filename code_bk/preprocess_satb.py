import os
import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from music21 import stream, note, chord

from constants import (
    MIDI_DATASET_PATH,
    SAVE_DIR,
    SINGLE_FILE_DATASET,
    MAPPING_PATH,
    SEQUENCE_LENGTH,
)

# ============================================================
# CONFIG
# ============================================================

TIME_STEP = 0.25  # 16th note resolution

ACCEPTABLE_DURATIONS = [
    0.25, 0.5, 0.75,
    1.0, 1.25, 1.5, 1.75,
    2.0, 2.25, 2.5, 2.75,
    3.0, 3.25, 3.5, 3.75,
    4.0, 5.0
]


# ============================================================
# LOADING MIDI
# ============================================================

def load_songs(dataset_path):
    """Load all MIDI files using music21"""
    songs = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".musicxml") or file.endswith(".mid"):
                try:
                    song = m21.converter.parse(os.path.join(root, file))
                    songs.append(song)
                except m21.midi.MidiException as e:
                    print(f"Skipping MIDI (error): {file} → {e}")

    return songs


# ============================================================
# VALIDATION
# ============================================================

def has_acceptable_durations(song):
    """Check if all durations are quantizable"""
    for el in song.flat.notesAndRests:
        if el.duration.quarterLength not in ACCEPTABLE_DURATIONS:
            return False
    return True


def normalize_parts(song):
    """
    Ensure exactly 4 parts:
    - keep first 4 if more
    - skip if fewer than 4
    """
    if len(song.parts) < 4:
        return None
    if len(song.parts) > 4:
        song = stream.Score(song.parts[:4])
    return song


# ============================================================
# TRANSPOSITION
# ============================================================

def transpose_to_c_bk(song):
    """Transpose song to C major / A minor"""
    try:
        key = song.analyze("key")
    except Exception:
        return song

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    else:
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    return song.transpose(interval)


def transpose_to_c(song):
    """
    Transposes song to C maj/A min using robust searching
    """
    try:
        # 1. Search the entire song for any KeySignature objects
        # This is way safer than measures_part0[0][4]
        key_list = song.flatten().getElementsByClass(m21.key.KeySignature)

        if key_list:
            key = key_list[0].asKey()
        else:
            # 2. Fallback to algorithmic analysis if no metadata exists
            key = song.analyze("key")

        # 3. Determine the interval
        if key.mode == "major":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        else:  # Covers minor and other modes like Dorian analyzed as minor
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

        # 4. Transpose
        return song.transpose(interval)

    except Exception as e:
        print(f"❌ Transposition failed: {e}. Returning original song.")
        return song

# ============================================================
# ENCODING
# ============================================================

def encode_part(part):
    """Encode a single voice into time steps"""
    encoded = []

    for event in part.flat.notesAndRests:
        if isinstance(event, note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, note.Rest):
            symbol = "r"
        elif isinstance(event, chord.Chord):
            symbol = event.pitches[0].midi
        else:
            continue

        steps = int(event.duration.quarterLength / TIME_STEP)

        for i in range(steps):
            encoded.append(symbol if i == 0 else "_")

    return encoded


def encode_song(song):
    """
    Encode SATB song:
    soprano ^ alto ^ tenor ^ bass
    """
    voices = []
    for part in song.parts[:4]:
        voices.append(" ".join(map(str, encode_part(part))))

    return "^".join(voices)


# ============================================================
# BEATS & FERMATA
# ============================================================
def encode_beats(song, time_step=0.25):
    beat_song = []
    beat_step = 0  # Initialize beat step
    beat_count = 1  # Start counting beats from 1
    fermata = 0  # Fermata indicator, 0 by default
    last_event_symbol = None  # Track the last event for adding fermata
    measure_number = 0
    part = song.parts[0]

    # Iterate through each measure in the part
    measure_number = 0
    for event in part.flat.notesAndRests:
        # Access the parent measure of the event
        parent_measure = event.getContextByClass('Measure')
        #print('parent_measure.number=' + str(parent_measure.number) + '--current measure_number=' + str(
            #measure_number))
        if parent_measure.number > measure_number:
            beat_count = 1
            measure_number = parent_measure.number

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            # if it's the first time we see a note/rest, let's encode it. Otherwise, it's a carry-over
            if step == 0:
                beat_song.append(beat_count)
            else:
                beat_song.append(beat_count)

        beat_step += steps
        if beat_step >= 4:
            beat_count += 1
            beat_step = 0

    # Cast encoded song to str
    beat_song = " ".join(map(str, beat_song))
    return beat_song




def encode_beats_bk(song):
    beats = []
    beat_count = 1
    beat_step = 0

    part = song.parts[0]

    for event in part.flat.notesAndRests:
        steps = int(event.duration.quarterLength / TIME_STEP)
        for _ in range(steps):
            beats.append(beat_count)

        beat_step += steps
        if beat_step >= 4:
            beat_step = 0
            beat_count += 1

    return " ".join(map(str, beats))


def add_fermata(encoded_voice, fermata=None):
    tokens = encoded_voice.split()
    target_len = len(tokens)

    if fermata is None:
        fermata_list = [0] * target_len
    else:
        if isinstance(fermata, str):
            fermata_list = [int(x) for x in fermata.split()] if fermata else []
        else:
            fermata_list = list(fermata)

        if len(fermata_list) < target_len:
            fermata_list.extend([0] * (target_len - len(fermata_list)))
        elif len(fermata_list) > target_len:
            fermata_list = fermata_list[:target_len]

    last_note = None
    for i, t in enumerate(tokens):
        if t.isdigit():
            last_note = i

    if last_note is not None:
        for i in range(last_note, len(tokens)):
            fermata_list[i] = 1

    return " ".join(map(str, fermata_list))


def add_fermata_v2(song, time_step=TIME_STEP, part_index=0):
    """
    Encode fermata directly from score annotations.
    For any event with a fermata, mark 1 for the duration of that event only.
    """
    def has_fermata(event):
        exprs = getattr(event, "expressions", [])
        arts = getattr(event, "articulations", [])
        if any(isinstance(e, m21.expressions.Fermata) for e in exprs):
            return True
        if any(isinstance(a, m21.articulations.Fermata) for a in arts):
            return True
        return False

    fermata = []
    part = song.parts[part_index]

    for event in part.flat.notesAndRests:
        steps = int(event.duration.quarterLength / time_step)
        value = 1 if has_fermata(event) else 0
        for _ in range(steps):
            fermata.append(value)

    return " ".join(map(str, fermata))


# ============================================================
# PREPROCESS PIPELINE
# ============================================================
# ============================================================
# NEW: AUGMENTATION LOGIC
# ============================================================

def preprocess_with_augmentation(dataset_path):
    """
    Standard preprocessing + transposing every song into
    all 12 possible keys.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    songs = load_songs(dataset_path)
    #songs = songs[:1]
    print(f"Loaded {len(songs)} songs for augmented preprocessing")

    saved = 0

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song):
            continue

        song = normalize_parts(song)
        if song is None:
            continue

        # 1. First, bring it to the "Baseline" (C Major / A Minor)
        baseline_song = transpose_to_c(song)

        # 2. Now, transpose that baseline into all 12 keys
        # We shift from -5 semitones to +6 semitones to cover the full octave
        for interval_steps in range(-5, 7):
            try:
                # Transpose the baseline by the current interval
                aug_song = baseline_song.transpose(interval_steps)

                encoded = encode_song(aug_song)
                beats = encode_beats(aug_song)
                fermata = add_fermata_v2(aug_song)
                fermata = add_fermata(encoded.split("^")[0], fermata)

                final = f"{encoded}^{beats}^{fermata}"

                # Save with a unique name including the transposition offset
                file_name = f"{saved}_trans_{interval_steps}"
                with open(os.path.join(SAVE_DIR, file_name), "w") as f:
                    f.write(final)
            except Exception as e:
                print(f"Error transposing song {i} by {interval_steps}: {e}")

        saved += 1
        if saved % 10 == 0:
            print(f"Processed {saved} songs (x12 keys each)")

    print(f"Saved {saved * 12} total SATB files (Augmentation complete)")

def preprocess(dataset_path):
    os.makedirs(SAVE_DIR, exist_ok=True)

    songs = load_songs(dataset_path)
    print(f"Loaded {len(songs)} songs")

    saved = 0

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song):
            continue

        song = normalize_parts(song)
        if song is None:
            continue

        song = transpose_to_c(song)

        encoded = encode_song(song)
        #print(encoded[:50])
        beats = encode_beats(song)
        fermata = add_fermata_v2(song)

        final = f"{encoded}^{beats}^{fermata}"

        with open(os.path.join(SAVE_DIR, str(saved)), "w") as f:
            f.write(final)

        saved += 1

        if saved % 10 == 0:
            print(f"Processed {saved} songs")

    print(f"Saved {saved} SATB files")


# ============================================================
# DATASET + MAPPING
# ============================================================

def create_single_file_dataset(sequence_length):
    delimiter = "/ " * sequence_length
    full_text = ""

    for file in os.listdir(SAVE_DIR):
        song = load_text(os.path.join(SAVE_DIR, file))
        parts = song.split("^")[:4]
        full_text += " ^ ".join(parts) + " " + delimiter

    with open(SINGLE_FILE_DATASET, "w") as f:
        f.write(full_text.strip())

    return full_text


def create_mapping(songs):
    vocab = sorted(set(songs.split()))
    mapping = {sym: i for i, sym in enumerate(vocab)}

    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=4)


def load_text(path):
    with open(path, "r") as f:
        return f.read()


# ============================================================
# TRAINING SEQUENCES
# ============================================================

def generate_training_sequences(sequence_length):
    songs = load_text(SINGLE_FILE_DATASET)
    mapping = json.load(open(MAPPING_PATH))

    int_songs = [mapping[s] for s in songs.split()]
    inputs, targets = [], []

    for i in range(len(int_songs) - sequence_length):
        inputs.append(int_songs[i:i + sequence_length])
        targets.append(int_songs[i:i + sequence_length])

    vocab_size = len(mapping)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocab_size)
    targets = np.array(targets)

    return inputs, targets


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def get_mapping():
    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    return mappings


def convert_song_to_int(song, mappings):
    choral_int = []
    voices = song.split('^')

    # Process the first 4 elements normally
    for v in voices[:4]:  # Only take the first 4 voices
        int_songs = []
        # Transform songs string to list
        events = v.split()
        # Map songs to int
        for symbol in events:
            int_songs.append(mappings[symbol])
        choral_int.append(int_songs)

    # Process the last 2 elements separately and append them to the end
    for v in voices[4:]:  # Take the last 2 elements
        int_songs = []
        events = v.split()
        for symbol in events:
            int_songs.append(symbol)
        choral_int.append(int_songs)

    return choral_int


def   midi_to_chorale_data(dataset_path, mappings):
    chorales = []

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)

            song = load(file_path)

            # Must return: [S, A, T, B, beats, fermata]
            int_song = convert_song_to_int(song, mappings)

            if len(int_song) == 6:
                chorales.append(int_song)
            else:
                print("Skipping file (wrong parts):", file)

    return chorales


# ============================================================
# MAIN
# ============================================================

def main():
    preprocess(MIDI_DATASET_PATH)
    #preprocess_with_augmentation(MIDI_DATASET_PATH)
    songs = create_single_file_dataset(SEQUENCE_LENGTH)
    create_mapping(songs)

    mappings = get_mapping()
    choral = midi_to_chorale_data(SAVE_DIR, mappings)
    #print(choral)

    print("Preprocessing complete ✅")


import json
import os


# ============================================================
# NEW: MAPPING UPDATE LOGIC
# ============================================================

def update_mapping_for_finetuning(new_midi_songs):
    """
    Checks all tokens in new songs against existing mapping.
    Adds new MIDI notes or symbols while preserving existing running numbers.
    """
    if not os.path.exists(MAPPING_PATH):
        print("Error: Mapping file not found. Run standard preprocess first.")
        return

    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    # Identify the highest running number currently in use
    next_id = max(mapping.values()) + 1
    new_symbols_added = 0

    for song_text in new_midi_songs:
        # tokens are MIDI numbers "60", "r", "_", etc.
        tokens = song_text.replace("^", " ").split()

        for token in tokens:
            if token not in mapping:
                mapping[token] = next_id
                print(f"New symbol found: {token} -> {next_id}")
                next_id += 1
                new_symbols_added += 1

    if new_symbols_added > 0:
        with open(MAPPING_PATH, "w") as f:
            json.dump(mapping, f, indent=4)
        print(f"✅ Mapping updated with {new_symbols_added} new items.")
    else:
        print("✅ No new notes found. Mapping is consistent.")

    return mapping


# ============================================================
# NEW: FINE-TUNING PREPROCESS PIPELINE
# ============================================================

def preprocess_for_finetuning(dataset_path):
    """
    Preprocess logic for new pianist data that preserves and updates
    the existing mapping rather than recreating it.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    songs = load_songs(dataset_path)
    print(f"Loaded {len(songs)} songs for fine-tuning")

    encoded_songs_list = []
    saved = 0

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song):
            continue

        song = normalize_parts(song)
        if song is None:
            continue

        song = transpose_to_c(song)

        encoded = encode_song(song)
        beats = encode_beats(song)
        fermata = add_fermata_v2(song)

        final_text = f"{encoded}^{beats}^{fermata}"
        encoded_songs_list.append(final_text)

        # Save the file with a specific 'ft_' prefix to distinguish from hymns
        with open(os.path.join(SAVE_DIR, f"ft_{saved}"), "w") as f:
            f.write(final_text)

        saved += 1
        if saved % 10 == 0:
            print(f"Processed {saved} fine-tuning songs")

    # Update the mapping with any new notes the pianist played
    #update_mapping_for_finetuning(encoded_songs_list)

    print(f"Fine-tuning preprocessing complete. Saved {saved} files.")


# ============================================================
# MAIN EXECUTIONS
# ============================================================

if __name__ == "__main__":
    main()
