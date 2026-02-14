import os

import music21 as m21
import uuid
from music21 import converter, stream, metadata, key, note
import xml.etree.ElementTree as ET

PROCESSED_FOLDER = 'processed/'


def transpose(song):
    """Transposes song to C maj/A min

    :param piece (m21 stream): Piece to transpose
    :return transposed_song (m21 stream):
    """

    # get key from the song
    # parts = song.getElementsByClass(m21.stream.Part)
    try:
        measures_part0 = song.getElementsByClass(m21.stream.Measure)
        key = measures_part0[0][4]

        # estimate key using music21
        if not isinstance(key, m21.key.Key):
            key = song.analyze("key")

        # get interval for transposition. E.g., Bmaj -> Cmaj
        if key.mode == "major":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        elif key.mode == "minor":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    except:
        print('An Exception occured , retrying')
        parts = song.getElementsByClass(m21.stream.Part)
        measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
        key = measures_part0[0][4]
        # estimate key using music21
        if not isinstance(key, m21.key.Key):
            key = song.analyze("key")

        # get interval for transposition. E.g., Bmaj -> Cmaj
        if key.mode == "major":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        elif key.mode == "minor":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_beats(song, time_step=0.25):
    beat_song = []
    beat_step = 0  # Initialize beat step
    beat_count = 1  # Start counting beats from 1
    fermata = 0  # Fermata indicator, 0 by default
    last_event_symbol = None  # Track the last event for adding fermata
    measure_number = 0
    part = song

    # Iterate through each measure in the part
    measure_number = 0
    for event in part.flat.notesAndRests:
        # Access the parent measure of the event
        parent_measure = event.getContextByClass('Measure')
        print('parent_measure.number=' + str(parent_measure.number) + '--current measure_number=' + str(
            measure_number))
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

def add_fermata(song, time_step=0.25):
    # Convert the text to a list
    elements = song.split()

    # Initialize the fermata list with zeros
    fermata = [0] * len(elements)

    # Traverse the list and find the last integer
    last_integer_index = None
    for i in range(len(elements)):
        if elements[i].isdigit():
            last_integer_index = i

    # Set fermata for the last integer element
    if last_integer_index is not None:
        fermata[last_integer_index:] = [1] * (len(elements) - last_integer_index)

    # Output the fermata list
    print("Fermata list:", fermata)

    # Cast encoded song to str
    fermata = " ".join(map(str, fermata))
    return fermata


def encode_song(song, time_step=0.25):
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter length
    :return:
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def save_melody_4parts(melody_4parts_list, step_duration=0.25, format="midi"):
    # Generate a random filename
    folder = PROCESSED_FOLDER
    filename = 'gen_mel_' + str(uuid.uuid4()) + '.mid'

    # Create a music21 stream for the whole piece.
    piece_stream = m21.stream.Score()
    key_signature = m21.key.KeySignature(0)  # 0 flats or sharps (C major)
    piece_stream.append(key_signature)

    print(melody_4parts_list)
    merged_list_score = merge_list(melody_4parts_list)

    for part in merged_list_score:
        # Create a stream for each part.
        part_stream = m21.stream.Part()
        part_stream.append(key_signature)

        # Initialize variables for note/rest creation.
        start_symbol = None
        step_counter = 1

        # Parse all elements in the part and create note/rest objects.
        for i, symbol in enumerate(part):
            # Handle case in which we have note/rest.
            if symbol == '/':
                continue
            if symbol == '^':
                break
            if symbol != "_" or i + 1 == len(part):
                # Ensure we are dealing with note/rest beyond the first note.
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter  # 0.25 * 4 = 1

                    # Handle rests.
                    if start_symbol == "r":
                        event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # Handle notes.
                    else:
                        # print("symbol=" + str(symbol))
                        event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        #print("is accidentals?")
                        #print(event.pitch.accidental)
                    part_stream.append(event)
                    # Reset the step counter.
                    step_counter = 1

                start_symbol = symbol
            # Handle case where we have prolongation sign "_".
            else:
                step_counter += 1

        # Append the part stream to the piece stream.
        piece_stream.append(part_stream)

    # Write the music21 stream to a MusicXML file
    xml_filename = filename.replace('.mid', '.musicxml')  # Change extension to .xml
    # print(xml_filename)
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    xml_file_path = os.path.join(PROCESSED_FOLDER, xml_filename)

    # Set metadata and defaults
    piece_stream.metadata = metadata.Metadata()
    piece_stream.metadata.composer = "DeepChoral"
    piece_stream.metadata.title = "Generated Harmony"

    #piece_stream.write('musicxml', xml_file_path)


    # Write the music21 stream to a MIDI file.
    # Remove part names

    for n in piece_stream.recurse().notes:
        try:
            n.pitch.simplifyEnharmonic()
        except:
            pass

    piece_stream.makeAccidentals(
        inPlace=True,
        cautionaryPitchClass=False,
        cautionaryAll=False
    )

    for n in piece_stream.recurse().notes:
        if n.pitch.accidental and n.pitch.accidental.name == 'natural':
            n.pitch.accidental.displayStatus = False

    # Remove trailing empty bars before export.
    # Clean each part individually (very important for multi-part scores)
    cleaned_parts = []
    for part in piece_stream.parts:
        cleaned = remove_trailing_empty_measures(part)
        cleaned_parts.append(cleaned)

    clean_score = stream.Score()
    for p in cleaned_parts:
        clean_score.insert(0, p)

    clean_score.write('musicxml', xml_file_path,makeNotation=True)
    clean_score.write('midi', file_path,makeNotation=True)
    #piece_stream.write(format, file_path,makeNotation=True)
    print('saved midi in' + str(file_path))
    #midi_to_musicxml(file_path,xml_file_path)

    # Load the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # List of new IDs to replace with
    new_ids = ['s', 'a', 't', 'b']

    # Update score-part IDs under part-list
    score_parts = root.findall('.//score-part')
    for i, score_part in enumerate(score_parts):
        score_part.set('id', new_ids[i])

    # Update part IDs under score-partwise
    parts = root.findall('.//part')
    for i, part in enumerate(parts):
        part.set('id', new_ids[i])

    # Save the updated XML content back to the file
    tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
    # Add more conditions as needed for other part IDs

    return xml_file_path


def merge_list(my_list):
    # Example lists
    # list1 = [[1, 3, 4, 56, 60, 13], [1, 3, 4, 56, 60, 13], [1, 3, 4, 56, 60, 13], [1, 3, 4, 56, 60, 13]]
    # list2 = [[1, 2, 4, 56, 60, 13], [1, 4, 4, 56, 60, 13], [1, 4, 4, 56, 60, 13], [1, 6, 4, 56, 60, 13]]
    print('********my_list****')
    print(my_list)
    # Merge lists
    merged_list = []

    # for l1, l2, l3, l4 in zip(my_list[0], my_list[1], my_list[2], my_list[3]):
    # merged_list.append(l1 + l2 + l3 + l4)

    # Iterate over the inner lists using zip(*my_list)
    for sublists in zip(*my_list):
        merged_sublist = []
        # Iterate over elements of sublists and concatenate them
        for elements in sublists:
            merged_sublist.extend(elements)
        merged_list.append(merged_sublist)

    # return merged_list
    # Print the merged list
    # print(merged_list)

    return merged_list


def convert_to_musicxml(score):
    # Load MIDI file
    # score = converter.parse(midi_file)
    # we directly pass score here

    # Convert to MusicXML format
    score.makeAccidentals(
        inPlace=True,
        cautionaryPitchClass=False,
        cautionaryAll=False
    )
    xml_str = score.write('musicxml',makeNotation=True)
    #print(xml_str)

    # Save the MusicXML string to a file
    # Replace 'output.xml' with your desired output file path
    with open('output.musicxml', 'w') as xml_file:
        xml_file.write(xml_str,makeNotation=True)

    print(f"Conversion successful. MusicXML file saved as 'output.xml'.")

def midi_to_musicxml(midi_path, musicxml_path):
    score = converter.parse(midi_path)
    score.write('musicxml', fp=musicxml_path,makeNotation=True)


def remove_trailing_empty_measures(s):
    """
    Removes trailing measures that contain only rests (or are completely empty)
    Works on Score, Part, or flat Stream
    """
    if not s.hasMeasures():
        s = s.makeMeasures(inPlace=False)  # older versions often need this

    # Get all measures
    measures = list(s.getElementsByClass('Measure'))
    if not measures:
        return s

    # Find the last non-empty measure
    last_good_index = -1
    for i, m in enumerate(measures):
        # A measure is "empty" if it has no notes (only rests, clefs, etc.)
        if any(e.isNote or e.isChord for e in m.flat.notesAndRests):
            last_good_index = i

    if last_good_index == -1:
        # everything is empty → keep minimal or clear
        return s

    # Keep only up to the last good measure
    measures_to_keep = measures[:last_good_index + 1]

    # Rebuild the stream (preserves most metadata)
    new_s = s.__class__()  # same class as input (Score/Part/Stream)
    for el in s.getElementsNotOfClass('Measure'):
        new_s.insert(el.offset, el)
    for m in measures_to_keep:
        new_s.insert(m.offset, m)

    return new_s
