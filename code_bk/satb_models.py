from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional,
    Dropout, Dense, TimeDistributed, Concatenate
)
from tensorflow.keras.models import Model

from preprocess_satb import get_mapping

mappings = get_mapping()
num_classes = len(mappings)


# =========================================
# Shared encoders
# =========================================

def build_note_encoder(x, units=512):
    x = Embedding(num_classes, 128, mask_zero=True)(x)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    return x


def build_rhythm_encoder(beats, fermata):
    beat_enc = Embedding(32, 16)(beats)       # beats 1-4
    ferm_enc = Embedding(3, 8)(fermata)     # 0/1

    return Concatenate()([beat_enc, ferm_enc])


# =========================================
# 1️⃣ BASS  (S -> B)
# =========================================
def build_bass_model(seq_len, num_classes):

    soprano = Input(shape=(seq_len,))
    beats   = Input(shape=(seq_len,))
    fermata = Input(shape=(seq_len,))

    s_enc = build_note_encoder(soprano)
    r_enc = build_rhythm_encoder(beats, fermata)

    x = Concatenate()([s_enc, r_enc])

    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

    return Model([soprano, beats, fermata], out)


# =========================================
# 2️⃣ ALTO  (S + B -> A)
# =========================================
def build_alto_model(seq_len, num_classes):

    soprano = Input(shape=(seq_len,))
    bass    = Input(shape=(seq_len,))
    beats   = Input(shape=(seq_len,))
    fermata = Input(shape=(seq_len,))

    s_enc = build_note_encoder(soprano)
    b_enc = build_note_encoder(bass)
    r_enc = build_rhythm_encoder(beats, fermata)

    x = Concatenate()([s_enc, b_enc, r_enc])

    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

    return Model([soprano, bass, beats, fermata], out)


# =========================================
# 3️⃣ TENOR  (S + B + A -> T)
# =========================================
def build_tenor_model(seq_len, num_classes):

    soprano = Input(shape=(seq_len,))
    bass    = Input(shape=(seq_len,))
    alto    = Input(shape=(seq_len,))
    beats   = Input(shape=(seq_len,))
    fermata = Input(shape=(seq_len,))

    s_enc = build_note_encoder(soprano)
    b_enc = build_note_encoder(bass)
    a_enc = build_note_encoder(alto)
    r_enc = build_rhythm_encoder(beats, fermata)

    x = Concatenate()([s_enc, b_enc, a_enc, r_enc])

    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

    return Model([soprano, bass, alto, beats, fermata], out)
