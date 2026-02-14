import os
from preprocess_satb import midi_to_chorale_data, get_mapping
from constants import SAVE_DIR


def analyze_and_save_lengths(output_file="song_lengths.txt"):
    mappings = get_mapping()
    chorale_data = midi_to_chorale_data(SAVE_DIR, mappings)

    lengths = [len(chorale[0]) for chorale in chorale_data]

    with open(output_file, "w") as f:
        f.write("Song Index | Length (Time Steps)\n")
        f.write("-" * 30 + "\n")
        for i, length in enumerate(lengths):
            f.write(f"{i:<11} | {length}\n")

    print(f"Stats: Min={min(lengths)}, Max={max(lengths)}, Avg={sum(lengths) / len(lengths)}")
    print(f"Analysis saved to {output_file}")


if __name__ == "__main__":
    analyze_and_save_lengths()