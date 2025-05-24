import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --- Configuration ---
MIDI_DATA_PATH = "midi_songs/" # Path to your MIDI files
OUTPUT_FOLDER = "output_music/"
MODEL_FILE = os.path.join(OUTPUT_FOLDER, "music_generator_model.h5")
NOTES_FILE = os.path.join(OUTPUT_FOLDER, "notes_vocab.pkl")

SEQUENCE_LENGTH = 50  # Length of input sequences
EPOCHS = 50          # Number of epochs to train for
BATCH_SIZE = 64
NUM_NOTES_TO_GENERATE = 200 # How many notes/chords to generate

# --- Helper Functions ---

def get_notes_from_midi_files(data_path):
    """
    Parses MIDI files and extracts notes and chords.
    Returns a list of all notes/chords as strings.
    """
    notes = []
    for file in glob.glob(data_path + "*.mid"):
        try:
            midi = converter.parse(file)
            print(f"Parsing {file}")
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                notes_to_parse = parts.parts[0].recurse() # Get notes from the first instrument
            else:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    return notes

def prepare_sequences(notes, n_vocab, sequence_length):
    """
    Prepare sequences for the LSTM model.
    X: input sequences
    y: corresponding output notes
    """
    # Map notes to integers
    pitchnames = sorted(list(set(notes)))
    note_to_int = dict((note_val, number) for number, note_val in enumerate(pitchnames))

    network_input_list = []
    network_output_list = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input_list.append([note_to_int[char] for char in sequence_in])
        network_output_list.append(note_to_int[sequence_out])

    n_patterns = len(network_input_list)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.array(network_input_list) # Shape: (n_patterns, sequence_length)
    # Normalize input (optional, but can help for some architectures, though embedding handles it well)
    # network_input = network_input / float(n_vocab) # Not strictly necessary if using Embedding

    network_output = to_categorical(network_output_list, num_classes=n_vocab)

    return (network_input, network_output, pitchnames, note_to_int)

def create_network(seq_length, n_vocab):
    """ Create the Keras model """
    model = Sequential()
    # Embedding layer for better representation of discrete note inputs
    model.add(Embedding(n_vocab, 100, input_shape=(seq_length,))) # n_vocab, embedding_dim, input_shape=(sequence_length,)
    model.add(LSTM(
        256, # Number of LSTM units
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256)) # Last LSTM layer does not return sequences
    model.add(BatchNormalization()) # Optional: helps stabilize training
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_network(model, network_input, network_output, model_file):
    """ Train the neural network """
    filepath = model_file # Save model checkpoints
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10, # Stop if loss doesn't improve for 10 epochs
        verbose=1,
        restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stopping]

    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    print(f"Training complete. Model saved to {model_file}")

def generate_music(model, pitchnames, note_to_int, n_vocab, sequence_length, num_notes_to_generate):
    """ Generate notes from the trained model """
    # Invert the note_to_int dictionary to get int_to_note
    int_to_note = dict((number, note_val) for number, note_val in enumerate(pitchnames))

    # Pick a random sequence from the input as a starting point for the generation
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start].tolist() # Get it as a list of integers
    prediction_output = []

    print("\n--- Generating Music ---")
    # Generate notes
    for note_index in range(num_notes_to_generate):
        prediction_input = np.reshape(pattern, (1, len(pattern))) # Shape: (1, sequence_length)
        # prediction_input = prediction_input / float(n_vocab) # Normalize if you did during prep

        prediction = model.predict(prediction_input, verbose=0)

        # Sample from the output distribution (argmax takes the most likely, could also sample)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        print(f"Generated: {result}")

        # Update pattern: remove first element, add new prediction
        pattern.append(index) # Append the integer index
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, output_file="output.mid"):
    """ Convert the output from the prediction to a MIDI file """
    offset = 0
    output_notes = []

    # Create note and chord objects based on the values generated by the model
    for pattern_item in prediction_output:
        # Pattern is a chord
        if ('.' in pattern_item) or pattern_item.isdigit(): # Check if it's a chord (e.g., "60.64.67")
            notes_in_chord = pattern_item.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern_item)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so notes do not stack
        offset += 0.5 # You can experiment with different offset increments

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"MIDI file saved as {output_file}")

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load and Preprocess Data
    print("--- Loading and Preprocessing Data ---")
    notes_corpus = get_notes_from_midi_files(MIDI_DATA_PATH)
    if not notes_corpus:
        print(f"No notes found. Please check your MIDI files in '{MIDI_DATA_PATH}'.")
        exit()

    n_vocab = len(set(notes_corpus))
    print(f"Vocabulary size: {n_vocab}")

    network_input, network_output, pitchnames, note_to_int = prepare_sequences(
        notes_corpus, n_vocab, SEQUENCE_LENGTH
    )

    # Save pitchnames and note_to_int for later generation if model is already trained
    with open(NOTES_FILE, 'wb') as f:
        pickle.dump({'pitchnames': pitchnames, 'note_to_int': note_to_int, 'n_vocab': n_vocab}, f)
    print(f"Vocabulary and mappings saved to {NOTES_FILE}")


    # 2. Train Model or Load Existing
    if os.path.exists(MODEL_FILE):
        print(f"\n--- Loading existing model from {MODEL_FILE} ---")
        try:
            model = load_model(MODEL_FILE)
            print("Model loaded successfully.")
            # Load vocab/mappings that were used to train this model
            with open(NOTES_FILE, 'rb') as f:
                data = pickle.load(f)
                pitchnames = data['pitchnames']
                note_to_int = data['note_to_int']
                n_vocab_loaded = data['n_vocab']
                if n_vocab_loaded != n_vocab: # Check if current dataset vocab matches saved
                    print(f"Warning: Loaded vocabulary size ({n_vocab_loaded}) differs from current dataset ({n_vocab}).")
                    print("This might lead to issues if the model was trained on a different vocabulary.")
                    print("Consider retraining if you changed your MIDI dataset significantly.")
                n_vocab = n_vocab_loaded # Use the vocab size the model was trained with
        except Exception as e:
            print(f"Error loading model: {e}. Will retrain.")
            model = create_network(SEQUENCE_LENGTH, n_vocab)
            train_network(model, network_input, network_output, MODEL_FILE)
    else:
        print("\n--- Creating and Training New Model ---")
        model = create_network(SEQUENCE_LENGTH, n_vocab)
        train_network(model, network_input, network_output, MODEL_FILE)

    # 3. Generate Music
    generated_sequence = generate_music(
        model, pitchnames, note_to_int, n_vocab, SEQUENCE_LENGTH, NUM_NOTES_TO_GENERATE
    )

    # 4. Create MIDI Output
    output_midi_file = os.path.join(OUTPUT_FOLDER, f"generated_music_seq{SEQUENCE_LENGTH}_epoch{EPOCHS}.mid")
    create_midi(generated_sequence, output_midi_file)

    print("\n--- Process Finished ---")