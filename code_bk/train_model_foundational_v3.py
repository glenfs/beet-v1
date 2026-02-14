import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from satb_models import (
    build_bass_model,
    build_alto_model,
    build_tenor_model
)

from preprocess_satb import midi_to_chorale_data, get_mapping
from constants import SAVE_DIR, SEQUENCE_LENGTH, SAVE_MODEL_DIR, VAL_DIR

# ======================================================
# CONFIG
# ======================================================

BATCH_SIZE = 256
EPOCHS = 100

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# LOAD & WINDOW DATA  (your SAME logic, cleaned)
# ======================================================
def load_val_dataset():

    print("🎼 Loading mappings and chorales...")

    mappings = get_mapping()
    num_classes = len(mappings)

    chorale_data = midi_to_chorale_data(SAVE_DIR, mappings)
    val_data = midi_to_chorale_data(VAL_DIR, mappings)

    S, B, A, T = [], [], [], []
    beats, fermata = [], []

    print("🔹 Windowing sequences val...")
    # for testing only remove this below line after.
    #chorale_data= chorale_data[: 100]

    for chorale in val_data:
        #print(chorale)
        #if len(chorale) < 6:
            #continue

        song_len = len(chorale[0])
        if song_len < SEQUENCE_LENGTH:
            continue

        for i in range(0, song_len - SEQUENCE_LENGTH, 4):

            s = chorale[0][i:i+SEQUENCE_LENGTH]
            a = chorale[1][i:i+SEQUENCE_LENGTH]
            t = chorale[2][i:i+SEQUENCE_LENGTH]
            b = chorale[3][i:i+SEQUENCE_LENGTH]
            bt = chorale[4][i:i+SEQUENCE_LENGTH]
            f = chorale[5][i:i+SEQUENCE_LENGTH]

            if all(len(x) == SEQUENCE_LENGTH for x in [s,a,t,b,bt,f]):
                S.append(s)
                B.append(b)
                A.append(a)
                T.append(t)
                beats.append(bt)
                fermata.append(f)

    print("🔹 Converting to numpy...")

    return (
        np.array(S, dtype="int32"),
        np.array(B, dtype="int32"),
        np.array(A, dtype="int32"),
        np.array(T, dtype="int32"),
        np.array(beats, dtype="int32"),
        np.array(fermata, dtype="int32"),
        num_classes
    )

# ======================================================
# LOAD & WINDOW DATA  (your SAME logic, cleaned)
# ======================================================
def load_dataset():

    print("🎼 Loading mappings and chorales...")

    mappings = get_mapping()
    num_classes = len(mappings)

    chorale_data = midi_to_chorale_data(SAVE_DIR, mappings)
    val_data = midi_to_chorale_data(VAL_DIR, mappings)

    S, B, A, T = [], [], [], []
    beats, fermata = [], []

    print("🔹 Windowing sequences...")
    # for testing only remove this below line after.
    #chorale_data= chorale_data[: 100]

    for chorale in chorale_data:
        #print(chorale)
        #if len(chorale) < 6:
            #continue

        song_len = len(chorale[0])
        if song_len < SEQUENCE_LENGTH:
            continue

        for i in range(0, song_len - SEQUENCE_LENGTH, 4):

            s = chorale[0][i:i+SEQUENCE_LENGTH]
            a = chorale[1][i:i+SEQUENCE_LENGTH]
            t = chorale[2][i:i+SEQUENCE_LENGTH]
            b = chorale[3][i:i+SEQUENCE_LENGTH]
            bt = chorale[4][i:i+SEQUENCE_LENGTH]
            f = chorale[5][i:i+SEQUENCE_LENGTH]

            if all(len(x) == SEQUENCE_LENGTH for x in [s,a,t,b,bt,f]):
                S.append(s)
                B.append(b)
                A.append(a)
                T.append(t)
                beats.append(bt)
                fermata.append(f)

    print("🔹 Converting to numpy...")

    return (
        np.array(S, dtype="int32"),
        np.array(B, dtype="int32"),
        np.array(A, dtype="int32"),
        np.array(T, dtype="int32"),
        np.array(beats, dtype="int32"),
        np.array(fermata, dtype="int32"),
        num_classes
    )


# ======================================================
# TRAIN HELPERS
# ======================================================
def compile_and_train(model, inputs, labels, inputs_v, labels_v,save_name):
    # 1. Define Callbacks
    callbacks = [
        # Stops training if 'val_loss' doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        # Saves the best version of the model based on validation performance
        ModelCheckpoint(
            filepath=SAVE_MODEL_DIR + "best_" + save_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    model.compile(
        optimizer=Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        inputs,
        labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        #validation_split=0.1,  # Uses 10% of data to check for overfitting
        validation_data=(inputs_v, labels_v),  # <--- ADD THIS
        callbacks=callbacks
        #callbacks=[
            #EarlyStopping(patience=8, restore_best_weights=True),
            #ModelCheckpoint(SAVE_DIR + save_name, save_best_only=True)
        #]
    )

    model.save(SAVE_MODEL_DIR + save_name)


# ======================================================
# MAIN
# ======================================================
def main():

    S, B, A, T, beats, fermata, num_classes = load_dataset()
    Sv, Bv, Av, Tv, beatsv, fermatav, num_classesv = load_val_dataset()

    print("\n==============================")
    print("Training Bass model (S → B)")
    print("==============================")

    bass_model = build_bass_model(SEQUENCE_LENGTH, num_classes)
    compile_and_train(
        bass_model,
        [S, beats, fermata],
        B,
        [Sv, beatsv, fermatav],
        Bv,
        "model_bass.h5"
    )


    print("\n==============================")
    print("Training Alto model (S+B → A)")
    print("==============================")

    alto_model = build_alto_model(SEQUENCE_LENGTH, num_classes)
    compile_and_train(
        alto_model,
        [S, B, beats, fermata],
        A,
        [Sv, Bv, beatsv, fermatav],
        Av,
        "model_alto.h5"
    )


    print("\n==============================")
    print("Training Tenor model (S+B+A → T)")
    print("==============================")

    tenor_model = build_tenor_model(SEQUENCE_LENGTH, num_classes)
    compile_and_train(
        tenor_model,
        [S, B, A, beats, fermata],
        T,
        [Sv, Bv, Av, beatsv, fermatav],
        Tv,
        "model_tenor.h5"
    )


    print("\n✅ All models trained and saved!")


if __name__ == "__main__":
    main()
