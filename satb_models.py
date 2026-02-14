from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional,
    Dropout, Dense, TimeDistributed, Concatenate
)
from tensorflow.keras.models import Model

from constants import USE_POS_ENCODING, POS_BINS, USE_MODE_ENCODING, MODE_BINS


# =========================================
# Shared encoders
# =========================================

def build_note_encoder(x, num_classes, units=384):
    x = Embedding(num_classes, 128, mask_zero=True)(x)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Dropout(0.35)(x)
    return x


def build_rhythm_encoder(beats, fermata, barlines, mode=None, pos=None):
    beat_enc = Embedding(32, 16)(beats)
    ferm_enc = Embedding(3, 8)(fermata)
    bar_enc = Embedding(3, 4)(barlines)
    encs = [beat_enc, ferm_enc, bar_enc]
    if mode is not None:
        mode_enc = Embedding(MODE_BINS, 4)(mode)
        encs.append(mode_enc)
    if pos is not None:
        pos_enc = Embedding(POS_BINS, 4)(pos)
        encs.append(pos_enc)
    return Concatenate()(encs)


# =========================================
# 1. BASS  (S -> B)
# =========================================
def build_bass_model(seq_len, num_classes):
    soprano = Input(shape=(seq_len,))
    beats = Input(shape=(seq_len,))
    fermata = Input(shape=(seq_len,))
    barlines = Input(shape=(seq_len,))
    mode = Input(shape=(seq_len,)) if USE_MODE_ENCODING else None
    pos = Input(shape=(seq_len,)) if USE_POS_ENCODING else None

    s_enc = build_note_encoder(soprano, num_classes)
    r_enc = build_rhythm_encoder(beats, fermata, barlines, mode, pos)

    x = Concatenate()([s_enc, r_enc])
    x = Bidirectional(LSTM(384, return_sequences=True))(x)
    x = Dropout(0.6)(x)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    inputs = [soprano, beats, fermata, barlines]
    if USE_MODE_ENCODING:
        inputs.append(mode)
    if USE_POS_ENCODING:
        inputs.append(pos)
    return Model(inputs, out)


# =========================================
# 2. ALTO  (S + B -> A)
# =========================================
def build_alto_model(seq_len, num_classes):
    soprano = Input(shape=(seq_len,))
    bass = Input(shape=(seq_len,))
    beats = Input(shape=(seq_len,))
    fermata = Input(shape=(seq_len,))
    barlines = Input(shape=(seq_len,))
    mode = Input(shape=(seq_len,)) if USE_MODE_ENCODING else None
    pos = Input(shape=(seq_len,)) if USE_POS_ENCODING else None

    s_enc = build_note_encoder(soprano, num_classes)
    b_enc = build_note_encoder(bass, num_classes)
    r_enc = build_rhythm_encoder(beats, fermata, barlines, mode, pos)

    x = Concatenate()([s_enc, b_enc, r_enc])
    x = Bidirectional(LSTM(384, return_sequences=True))(x)
    x = Dropout(0.6)(x)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    inputs = [soprano, bass, beats, fermata, barlines]
    if USE_MODE_ENCODING:
        inputs.append(mode)
    if USE_POS_ENCODING:
        inputs.append(pos)
    return Model(inputs, out)


# =========================================
# 3. TENOR  (S + B + A -> T)
# =========================================
def build_tenor_model(seq_len, num_classes):
    soprano = Input(shape=(seq_len,))
    bass = Input(shape=(seq_len,))
    alto = Input(shape=(seq_len,))
    beats = Input(shape=(seq_len,))
    fermata = Input(shape=(seq_len,))
    barlines = Input(shape=(seq_len,))
    mode = Input(shape=(seq_len,)) if USE_MODE_ENCODING else None
    pos = Input(shape=(seq_len,)) if USE_POS_ENCODING else None

    s_enc = build_note_encoder(soprano, num_classes)
    b_enc = build_note_encoder(bass, num_classes)
    a_enc = build_note_encoder(alto, num_classes)
    r_enc = build_rhythm_encoder(beats, fermata, barlines, mode, pos)

    x = Concatenate()([s_enc, b_enc, a_enc, r_enc])
    x = Bidirectional(LSTM(384, return_sequences=True))(x)
    x = Dropout(0.6)(x)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    inputs = [soprano, bass, alto, beats, fermata, barlines]
    if USE_MODE_ENCODING:
        inputs.append(mode)
    if USE_POS_ENCODING:
        inputs.append(pos)
    return Model(inputs, out)







