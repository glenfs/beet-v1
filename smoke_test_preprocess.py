import os

from constants import SAVE_DIR, SEQUENCE_LENGTH, USE_POS_ENCODING
from preprocess_satb import (
    load_text,
    create_single_file_dataset,
    create_mapping,
    get_mapping,
    midi_to_chorale_data
)


def main():
    if not os.path.isdir(SAVE_DIR):
        raise SystemExit(f"SAVE_DIR not found: {SAVE_DIR}")

    songs = create_single_file_dataset(SEQUENCE_LENGTH)
    create_mapping(songs)

    mappings = get_mapping()
    chorales = midi_to_chorale_data(SAVE_DIR, mappings)

    if not chorales:
        raise SystemExit("No chorales found. Check SAVE_DIR contents.")

    sample = chorales[0]
    expected_parts = 8 if USE_POS_ENCODING else 7
    if len(sample) != expected_parts:
        raise SystemExit(f"Expected {expected_parts} parts, got {len(sample)}.")

    part_lengths = [len(p) for p in sample]
    if len(set(part_lengths)) != 1:
        raise SystemExit(f"Part length mismatch: {part_lengths}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
