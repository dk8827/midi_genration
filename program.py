import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, articulations, dynamics
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import re # For cleaning instrument names

# --- Custom Callback for MIDI Generation per Epoch ---
class GenerateMidiCallback(keras.callbacks.Callback):
    def __init__(self, output_folder, sequence_length, num_notes_to_generate,
                 pitchnames, event_to_int, n_vocab, seed_input_sequences, base_filename="generated_epoch"):
        super(GenerateMidiCallback, self).__init__()
        self.output_folder = output_folder
        self.sequence_length = sequence_length
        self.num_notes_to_generate = num_notes_to_generate
        self.pitchnames = pitchnames
        self.event_to_int = event_to_int
        self.n_vocab = n_vocab
        self.seed_input_sequences = seed_input_sequences # This should be a numpy array of possible seed patterns
        self.base_filename = base_filename
        os.makedirs(self.output_folder, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} finished. Generating sample MIDI...")

        if self.seed_input_sequences is None or len(self.seed_input_sequences) == 0:
            print("Callback: No seed input sequences available, cannot generate MIDI.")
            return

        # Generate music using the current model state
        generated_sequence = generate_music(
            self.model,
            self.pitchnames,
            self.event_to_int,
            self.n_vocab,
            self.sequence_length,
            self.num_notes_to_generate,
            self.seed_input_sequences # Pass the prepared seed sequences
        )

        # Create MIDI file
        output_midi_file = os.path.join(self.output_folder, f"{self.base_filename}_{epoch+1}.mid")
        create_midi(generated_sequence, output_midi_file)
        print(f"Callback: Sample MIDI saved to {output_midi_file}")

# --- Configuration ---
MIDI_DATA_PATH = "midi_songs/" # Path to your MIDI files
OUTPUT_FOLDER = "output_music/"
MODEL_FILE = os.path.join(OUTPUT_FOLDER, "music_generator_instrument_model.h5")
NOTES_FILE = os.path.join(OUTPUT_FOLDER, "notes_instrument_vocab.pkl")
PROCESSED_MIDI_EVENTS_FILE = os.path.join(OUTPUT_FOLDER, "processed_midi_events.pkl")

SEQUENCE_LENGTH = 50
EPOCHS = 50 # Can be increased for better results
BATCH_SIZE = 64
NUM_NOTES_TO_GENERATE = 200
DEFAULT_INSTRUMENT_NAME = "Piano" # Fallback instrument

# --- Helper Functions ---

INSTRUMENT_MAP = {
    # General MIDI Mappings (a selection, you can expand this)
    # Key: music21 instrument class name (or common name), Value: music21 object
    "Piano": instrument.Piano(),
    "Harpsichord": instrument.Harpsichord(),
    "Clavinet": instrument.ElectricPiano(),
    "Celesta": instrument.Celesta(),
    "Glockenspiel": instrument.Glockenspiel(),
    "Music Box": instrument.Celesta(),
    "Vibraphone": instrument.Vibraphone(),
    "Marimba": instrument.Marimba(),
    "Xylophone": instrument.Xylophone(),
    "Tubular Bells": instrument.TubularBells(),
    "Dulcimer": instrument.Dulcimer(),
    "Electric Guitar (jazz)": instrument.ElectricGuitar(),
    "Electric Guitar (clean)": instrument.ElectricGuitar(),
    "Electric Guitar (muted)": instrument.ElectricGuitar(),
    "Overdriven Guitar": instrument.ElectricGuitar(),
    "Distortion Guitar": instrument.ElectricGuitar(),
    "Acoustic Guitar (steel)": instrument.AcousticGuitar(),
    "Acoustic Guitar (nylon)": instrument.AcousticGuitar(),
    "Violin": instrument.Violin(),
    "Viola": instrument.Viola(),
    "Cello": instrument.Violoncello(),
    "Contrabass": instrument.Contrabass(),
    "StringEnsemble1": instrument.StringInstrument(), # Generic string
    "StringEnsemble2": instrument.StringInstrument(),
    "SynthStrings1": instrument.ElectricPiano(), # Was Synthesizer
    "SynthStrings2": instrument.ElectricPiano(), # Was Synthesizer
    "Choir Aahs": instrument.Choir(),
    "Voice Oohs": instrument.Vocalist(), # Was Voice
    "Synth Voice": instrument.Vocalist(), # Was SynthVoice
    "Orchestra Hit": instrument.Sampler(), # Was Ensemble
    "Trumpet": instrument.Trumpet(),
    "Trombone": instrument.Trombone(),
    "Tuba": instrument.Tuba(),
    "French Horn": instrument.Horn(), # Was FrenchHorn
    "BrassSection": instrument.BrassInstrument(), # Generic brass
    "Soprano Sax": instrument.SopranoSaxophone(), # Was Saxophone
    "Alto Sax": instrument.AltoSaxophone(),
    "Tenor Sax": instrument.TenorSaxophone(),
    "Baritone Sax": instrument.BaritoneSaxophone(),
    "Oboe": instrument.Oboe(),
    "English Horn": instrument.EnglishHorn(),
    "Bassoon": instrument.Bassoon(),
    "Clarinet": instrument.Clarinet(),
    "Piccolo": instrument.Piccolo(),
    "Flute": instrument.Flute(),
    "Recorder": instrument.Recorder(),
    "Pan Flute": instrument.PanFlute(),
    "Blown Bottle": instrument.Shakuhachi(), # Closest
    "Shakuhachi": instrument.Shakuhachi(),
    "Whistle": instrument.Whistle(),
    "Ocarina": instrument.Ocarina(),
    # Percussion - Music21 handles percussion specially, often in PercussionChord
    # For simplicity, we might map common drum names to a generic Percussion object
    # or try to learn specific drum sounds if your MIDI files are very detailed.
    "Drums": instrument.Percussion(),
    "Acoustic Bass Drum": instrument.BassDrum(), # Was Percussion
    "Bass Drum 1": instrument.BassDrum(), # Was Percussion
    "Side Stick": instrument.Percussion(), # SnareDrum has 'side' modifier, but Percussion is simpler
    "Acoustic Snare": instrument.SnareDrum(), # Was Percussion
    "Hand Clap": instrument.Percussion(), # No specific HandClap class
    "Electric Snare": instrument.SnareDrum(), # Was Percussion, SnareDrum can be electric
    "Low Floor Tom": instrument.TomTom(), # Was Percussion
    "Closed Hi Hat": instrument.HiHatCymbal(), # Was Percussion
    "High Floor Tom": instrument.TomTom(), # Was Percussion
    "Pedal Hi Hat": instrument.HiHatCymbal(), # Was Percussion
    "Low Tom": instrument.TomTom(), # Was Percussion
    "Open Hi Hat": instrument.HiHatCymbal(), # Was Percussion
    "Low-Mid Tom": instrument.TomTom(), # Was Percussion
    "Hi-Mid Tom": instrument.TomTom(), # Was Percussion
    "Crash Cymbal 1": instrument.CrashCymbals(), # Was Percussion
    "High Tom": instrument.TomTom(), # Was Percussion
    "Ride Cymbal 1": instrument.RideCymbals(), # Was Percussion
    "Chinese Cymbal": instrument.Cymbals(), # Was Percussion (generic Cymbals)
    "Ride Bell": instrument.RideCymbals(), # RideCymbals often have a bell sound
    "Tambourine": instrument.Tambourine(), # Was Percussion
    "Splash Cymbal": instrument.SplashCymbals(), # Was Percussion
    "Cowbell": instrument.Cowbell(), # Was Percussion
    "Crash Cymbal 2": instrument.CrashCymbals(), # Was Percussion
    "Vibraslap": instrument.Vibraslap(), # Was Percussion
    "Ride Cymbal 2": instrument.RideCymbals(), # Was Percussion
    "Hi Bongo": instrument.BongoDrums(), # Was Percussion
    "Low Bongo": instrument.BongoDrums(), # Was Percussion
    "Mute Hi Conga": instrument.CongaDrum(), # Was Percussion
    "Open Hi Conga": instrument.CongaDrum(), # Was Percussion
    "Low Conga": instrument.CongaDrum(), # Was Percussion
    "High Timbale": instrument.Timbales(), # Was Percussion
    "Low Timbale": instrument.Timbales(), # Was Percussion
    "High Agogo": instrument.Agogo(), # Was Percussion
    "Low Agogo": instrument.Agogo(), # Was Percussion
    "Cabasa": instrument.Percussion(), # No specific Cabasa, use generic
    "Maracas": instrument.Maracas(), # Was Percussion
    "Short Whistle": instrument.Whistle(), # Re-use
    "Long Whistle": instrument.Whistle(),  # Re-use
    "Short Guiro": instrument.Percussion(), # No specific Guiro
    "Long Guiro": instrument.Percussion(),  # No specific Guiro
    "Claves": instrument.Percussion(), # No specific Claves
    "Hi Wood Block": instrument.Woodblock(), # Was Percussion
    "Low Wood Block": instrument.Woodblock(), # Was Percussion
    "Mute Cuica": instrument.Percussion(), # No specific Cuica
    "Open Cuica": instrument.Percussion(),  # No specific Cuica
    "Mute Triangle": instrument.Triangle(), # Was Percussion
    "Open Triangle": instrument.Triangle(), # Was Percussion
}

def get_instrument_from_name(name_str):
    """Attempts to get a music21 instrument object from its name."""
    if name_str in INSTRUMENT_MAP:
        # Create a new instance to avoid sharing state across notes/parts
        return INSTRUMENT_MAP[name_str].__class__() # Returns a new instance of the same class

    # Try music21's built-in string parsing
    try:
        # Clean up name slightly for better matching
        cleaned_name = re.sub(r'\d+', '', name_str).strip() # Remove numbers
        instr_obj = instrument.fromString(cleaned_name)
        if instr_obj:
            return instr_obj
    except: # music21.instrument.InstrumentException or other errors
        pass

    # Fallback for common patterns music21 might not catch directly
    name_lower = name_str.lower()
    if "piano" in name_lower: return instrument.Piano()
    if "guitar" in name_lower: return instrument.AcousticGuitar()
    if "violin" in name_lower: return instrument.Violin()
    if "flute" in name_lower: return instrument.Flute()
    if "drum" in name_lower or "percussion" in name_lower: return instrument.Percussion()
    if "sax" in name_lower: return instrument.Saxophone()
    if "trumpet" in name_lower: return instrument.Trumpet()
    if "bass" in name_lower: return instrument.ElectricBass()

    print(f"Warning: Could not map instrument name '{name_str}'. Defaulting to Piano.")
    return instrument.Piano()


def get_instrument_name(m21_instrument):
    """Gets a simplified, consistent name for a music21 instrument object."""
    if m21_instrument:
        if hasattr(m21_instrument, 'instrumentName') and m21_instrument.instrumentName:
            name = m21_instrument.instrumentName
        elif hasattr(m21_instrument, 'bestName') and callable(m21_instrument.bestName):
             name = m21_instrument.bestName()
        else:
            name = m21_instrument.__class__.__name__ # E.g., "Piano", "Violin"

        # Basic standardization (you can expand this)
        if "guitar" in name.lower(): return "Guitar"
        if "piano" in name.lower(): return "Piano"
        if "violin" in name.lower(): return "Violin"
        if "flute" in name.lower(): return "Flute"
        if "sax" in name.lower(): return "Saxophone"
        if "drum" in name.lower() or "percussion" in name.lower(): return "Drums"
        if "bass" in name.lower() and "drum" not in name.lower(): return "Bass" # avoid "Bass Drum" becoming "Bass"
        if "voice" in name.lower() or "choir" in name.lower(): return "Voice"
        if "synth" in name.lower(): return "Synth"

        # Remove numbers and extra spaces for more general categories
        name = re.sub(r'\d+', '', name).strip()
        name = re.sub(r'\s+', ' ', name) # Consolidate multiple spaces

        # Check against our map keys for a more canonical name
        for k in INSTRUMENT_MAP.keys():
            if k.lower() == name.lower():
                return k
        return name
    return DEFAULT_INSTRUMENT_NAME

def get_notes_from_midi_files(data_path):
    """
    Parses MIDI files and extracts notes, chords, and rests,
    prefixed with their instrument names.
    Returns a list of all "INSTRUMENT:EVENT" strings.
    """
    notes_with_instruments = []
    for file_path in glob.glob(data_path + "*.mid"):
        try:
            midi = converter.parse(file_path)
            print(f"Parsing {file_path}")
            
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                for part in parts:
                    instr = part.getInstrument()
                    instr_name = get_instrument_name(instr)
                    # print(f"  Found instrument: {instr_name} (from {instr}) in part {part.id if part.id else 'N/A'}")

                    for element in part.recurse().notesAndRests: # notesAndRests to include rests
                        event_str = None
                        if isinstance(element, note.Note):
                            event_str = str(element.pitch)
                        elif isinstance(element, chord.Chord):
                            event_str = '.'.join(str(n) for n in element.normalOrder)
                        elif isinstance(element, note.Rest):
                            event_str = "Rest"
                        
                        if event_str:
                            notes_with_instruments.append(f"{instr_name}:{event_str}")
            else:  # file has notes in a flat structure
                # Try to get instrument from the flat stream if possible, otherwise default
                current_instrument_name = DEFAULT_INSTRUMENT_NAME
                # Check if there's an instrument object at the beginning of the flat stream
                instr_at_start = midi.flat.getElementsByClass(instrument.Instrument).first()
                if instr_at_start:
                    current_instrument_name = get_instrument_name(instr_at_start)
                
                for element in midi.flat.notesAndRests:
                    event_str = None
                    # Check if instrument changes mid-stream (less common in flat files but possible)
                    el_instr = element.getInstrument(returnDefault=False)
                    if el_instr:
                         current_instrument_name = get_instrument_name(el_instr)

                    if isinstance(element, note.Note):
                        event_str = str(element.pitch)
                    elif isinstance(element, chord.Chord):
                        event_str = '.'.join(str(n) for n in element.normalOrder)
                    elif isinstance(element, note.Rest):
                        event_str = "Rest"

                    if event_str:
                        notes_with_instruments.append(f"{current_instrument_name}:{event_str}")
            
            if not parts and not midi.flat.notesAndRests:
                 print(f"    No notes or parts found in {file_path}")


        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    print(f"Total instrumented events extracted: {len(notes_with_instruments)}")
    if not notes_with_instruments:
        print("Warning: No notes were extracted. Check your MIDI files and parsing logic.")
    # else:
        # print("Sample extracted events:", notes_with_instruments[:10])
    return notes_with_instruments

def prepare_sequences(notes_with_instruments, n_vocab, sequence_length):
    """
    Prepare sequences for the LSTM model.
    X: input sequences
    y: corresponding output notes
    """
    # Map "INSTRUMENT:EVENT" strings to integers
    pitchnames = sorted(list(set(notes_with_instruments)))
    # print("Vocabulary (first 20):", pitchnames[:20])
    # print("Total vocabulary items:", len(pitchnames))

    event_to_int = dict((event, number) for number, event in enumerate(pitchnames))

    network_input_list = []
    network_output_list = []

    for i in range(0, len(notes_with_instruments) - sequence_length, 1):
        sequence_in = notes_with_instruments[i:i + sequence_length]
        sequence_out = notes_with_instruments[i + sequence_length]
        network_input_list.append([event_to_int[char] for char in sequence_in])
        network_output_list.append(event_to_int[sequence_out])

    n_patterns = len(network_input_list)
    if n_patterns == 0:
        raise ValueError("No patterns generated. Increase data or decrease sequence_length.")

    network_input = np.array(network_input_list)
    network_output = to_categorical(network_output_list, num_classes=n_vocab)

    return (network_input, network_output, pitchnames, event_to_int)

def create_network(seq_length, n_vocab):
    """ Create the Keras model """
    model = Sequential()
    model.add(Embedding(n_vocab, 256, input_shape=(seq_length,))) # Increased embedding dim
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512)) # Last LSTM layer does not return sequences
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu')) # Dense layer before output
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_network(model, network_input, network_output, model_file, custom_callbacks=None):
    """ Train the neural network """
    filepath = model_file
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1, # Changed to 1 for more feedback
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='loss', # Could also use 'val_loss' if using validation_split
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stopping]
    if custom_callbacks:
        if isinstance(custom_callbacks, list):
            callbacks_list.extend(custom_callbacks)
        else:
            callbacks_list.append(custom_callbacks)

    # Consider adding validation_split if dataset is large enough
    # history = model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE,
    #                     callbacks=callbacks_list, validation_split=0.1)
    history = model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    print(f"Training complete. Model saved to {model_file}")
    return history

def generate_music(model, pitchnames, event_to_int, n_vocab, sequence_length, num_events_to_generate, network_input_for_start):
    """ Generate "INSTRUMENT:EVENT" tokens from the trained model """
    int_to_event = dict((number, event) for number, event in enumerate(pitchnames))

    start = np.random.randint(0, len(network_input_for_start) - 1)
    pattern = network_input_for_start[start].tolist() # This is already a list of ints
    prediction_output = []

    print("\n--- Generating Music ---")
    for event_index in range(num_events_to_generate):
        prediction_input = np.reshape(pattern, (1, len(pattern)))
        # No normalization needed here as Embedding layer handles integer inputs

        prediction = model.predict(prediction_input, verbose=0)[0] # Get the single prediction array

        # --- Temperature Sampling (Optional - for more variety) ---
        # temperature = 1.0 # Higher temp = more randomness, lower = more conservative
        # prediction = np.log(prediction + 1e-7) / temperature # Add epsilon to avoid log(0)
        # exp_preds = np.exp(prediction)
        # prediction = exp_preds / np.sum(exp_preds)
        # index = np.random.choice(len(prediction), p=prediction)
        # --- Argmax Sampling (takes the most likely) ---
        index = np.argmax(prediction)
        
        result = int_to_event[index]
        prediction_output.append(result)
        print(f"Generated ({event_index+1}/{num_events_to_generate}): {result}")

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, output_file="output_instrumental.mid"):
    """ Convert the output "INSTRUMENT:EVENT" tokens to a multi-instrument MIDI file """
    
    score = stream.Score()
    parts = {} # To hold stream.Part for each instrument
    part_offsets = {} # To keep track of current offset for each part

    current_global_offset = 0.0 # A simple way to advance time

    for i, pattern_item in enumerate(prediction_output):
        try:
            if ':' not in pattern_item:
                print(f"Skipping malformed item (no ':'): {pattern_item}")
                continue

            instrument_name_str, event_str = pattern_item.split(':', 1)

            # Get or create the part for this instrument
            if instrument_name_str not in parts:
                m21_instr_obj = get_instrument_from_name(instrument_name_str)
                new_part = stream.Part(id=instrument_name_str)
                new_part.insert(0, m21_instr_obj) # Add instrument at the beginning of the part
                parts[instrument_name_str] = new_part
                part_offsets[instrument_name_str] = 0.0
                score.insert(0, new_part) # Insert part into the score

            current_part = parts[instrument_name_str]
            # part_offset = part_offsets[instrument_name_str] # Get current offset for this part

            m21_event = None
            duration = 0.5 # Default duration, can be learned or varied

            if event_str == "Rest":
                m21_event = note.Rest()
                m21_event.duration.quarterLength = duration
            elif ('.' in event_str) or event_str.isdigit(): # Chord
                notes_in_chord_pitches = event_str.split('.')
                chord_notes = []
                for p in notes_in_chord_pitches:
                    try:
                        n = note.Note(int(p)) # music21 normalOrder is MIDI pitch numbers
                        chord_notes.append(n)
                    except Exception as e_chord_note:
                        print(f"Warning: Could not parse pitch '{p}' in chord '{event_str}'. Skipping. Error: {e_chord_note}")
                        continue
                if chord_notes:
                    m21_event = chord.Chord(chord_notes)
                    m21_event.duration.quarterLength = duration
            else: # Single note
                try:
                    m21_event = note.Note(event_str)
                    m21_event.duration.quarterLength = duration
                except Exception as e_note:
                    print(f"Warning: Could not parse note '{event_str}'. Skipping. Error: {e_note}")
                    continue
            
            if m21_event:
                # Instead of per-part offset, let's use a global offset for simplicity for now
                # The model implicitly learns timing between instrument events.
                # To place it in the correct part at the global offset:
                current_part.insert(current_global_offset, m21_event)
                # part_offsets[instrument_name_str] += m21_event.duration.quarterLength # Advance this part's offset

            # Advance global offset based on the event that just occurred.
            # This means events are sequential in the generated list,
            # but will be placed in their respective instrument tracks.
            current_global_offset += duration # Simple increment, can be more sophisticated

        except Exception as e:
            print(f"Error processing generated item '{pattern_item}' at index {i}: {e}")
            continue
            
    if not score.elements:
        print("No valid music elements were added to the score. MIDI will be empty.")
        return

    try:
        score.write('midi', fp=output_file)
        print(f"Multi-instrument MIDI file saved as {output_file}")
    except Exception as e:
        print(f"Error writing MIDI file: {e}")
        # score.show('text') # For debugging what's in the score

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load and Preprocess Data
    print("--- Loading and Preprocessing Data ---")
    if os.path.exists(PROCESSED_MIDI_EVENTS_FILE):
        print(f"Loading pre-processed MIDI events from {PROCESSED_MIDI_EVENTS_FILE}...")
        with open(PROCESSED_MIDI_EVENTS_FILE, 'rb') as f:
            notes_corpus = pickle.load(f)
        print("Pre-processed MIDI events loaded.")
    else:
        print(f"No pre-processed data found. Parsing MIDI files from {MIDI_DATA_PATH}...")
        notes_corpus = get_notes_from_midi_files(MIDI_DATA_PATH)
        if notes_corpus:
            with open(PROCESSED_MIDI_EVENTS_FILE, 'wb') as f:
                pickle.dump(notes_corpus, f)
            print(f"Parsed MIDI events saved to {PROCESSED_MIDI_EVENTS_FILE}")
        else:
            print(f"No notes found after parsing. Pre-processed file not created.")

    if not notes_corpus:
        print(f"No notes available (either from parsing or loading). Please check your MIDI files in '{MIDI_DATA_PATH}'.")
        exit()

    n_vocab_current = len(set(notes_corpus))
    print(f"Current dataset vocabulary size: {n_vocab_current}")

    # Try to load existing vocab to ensure consistency if model exists
    n_vocab_to_use = n_vocab_current
    pitchnames_to_use = sorted(list(set(notes_corpus)))
    event_to_int_to_use = {event: i for i, event in enumerate(pitchnames_to_use)}

    if os.path.exists(MODEL_FILE) and os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, 'rb') as f:
                data = pickle.load(f)
                pitchnames_loaded = data['pitchnames']
                event_to_int_loaded = data['event_to_int']
                n_vocab_loaded = data['n_vocab']
            
            print(f"Loaded vocabulary size from file: {n_vocab_loaded}")
            if n_vocab_loaded != n_vocab_current:
                 print(f"Warning: Vocabulary size mismatch! Loaded: {n_vocab_loaded}, Current Dataset: {n_vocab_current}.")
                 print("Using loaded vocabulary for generation with the existing model.")
            n_vocab_to_use = n_vocab_loaded
            pitchnames_to_use = pitchnames_loaded
            event_to_int_to_use = event_to_int_loaded
        except Exception as e:
            print(f"Error loading {NOTES_FILE}, will use current dataset's vocab: {e}")
            # Fallback to current dataset's vocab if loading fails
            with open(NOTES_FILE, 'wb') as f:
                pickle.dump({'pitchnames': pitchnames_to_use, 'event_to_int': event_to_int_to_use, 'n_vocab': n_vocab_to_use}, f)
            print(f"New vocabulary and mappings saved to {NOTES_FILE}")


    # Prepare sequences with the chosen vocabulary (either loaded or from current data)
    # For training, we always use the current data's full vocabulary.
    # For generation with a pre-trained model, we use the vocabulary it was trained on.
    
    # Sequences for training (always from current full data)
    training_pitchnames = sorted(list(set(notes_corpus)))
    training_n_vocab = len(training_pitchnames)
    training_event_to_int = {event: i for i, event in enumerate(training_pitchnames)}
    
    network_input, network_output, _, _ = prepare_sequences(
        notes_corpus, training_n_vocab, SEQUENCE_LENGTH
    )

    # Save current training vocabulary if we are about to train a new model
    if not os.path.exists(MODEL_FILE) or not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, 'wb') as f:
            pickle.dump({'pitchnames': training_pitchnames, 'event_to_int': training_event_to_int, 'n_vocab': training_n_vocab}, f)
        print(f"Vocabulary for new model saved to {NOTES_FILE}")
        n_vocab_to_use = training_n_vocab
        pitchnames_to_use = training_pitchnames
        event_to_int_to_use = training_event_to_int


    # 2. Train Model or Load Existing
    if os.path.exists(MODEL_FILE):
        print(f"\n--- Loading existing model from {MODEL_FILE} ---")
        try:
            model = load_model(MODEL_FILE)
            print("Model loaded successfully.")
            # Ensure n_vocab_to_use, pitchnames_to_use, event_to_int_to_use are from the loaded NOTES_FILE
            # This should have been handled by the logic above that loads NOTES_FILE
        except Exception as e:
            print(f"Error loading model: {e}. Will retrain with current data.")
            model = create_network(SEQUENCE_LENGTH, training_n_vocab) # Train with current data vocab
            
            # Prepare seed for callback before training
            callback_seed_sequences = None
            possible_starts_for_callback = []
            for i in range(len(notes_corpus) - SEQUENCE_LENGTH):
                seq = notes_corpus[i:i+SEQUENCE_LENGTH]
                try:
                    mapped_seq = [training_event_to_int[s] for s in seq] # Use training vocab
                    possible_starts_for_callback.append(mapped_seq)
                except KeyError:
                    pass 
            if possible_starts_for_callback:
                callback_seed_sequences = np.array(possible_starts_for_callback)
            else:
                print("Warning: Could not create seed for MIDI generation callback during retraining.")

            midi_callback = GenerateMidiCallback(
                output_folder=OUTPUT_FOLDER,
                sequence_length=SEQUENCE_LENGTH,
                num_notes_to_generate=NUM_NOTES_TO_GENERATE // 4, # Generate shorter samples during epoch ends
                pitchnames=training_pitchnames,
                event_to_int=training_event_to_int,
                n_vocab=training_n_vocab,
                seed_input_sequences=callback_seed_sequences
            )
            train_network(model, network_input, network_output, MODEL_FILE, custom_callbacks=[midi_callback])
            # After training, the model is based on training_vocab, so set generation params accordingly
            n_vocab_to_use = training_n_vocab
            pitchnames_to_use = training_pitchnames
            event_to_int_to_use = training_event_to_int
            with open(NOTES_FILE, 'wb') as f: # Save the vocab the new model was trained on
                pickle.dump({'pitchnames': pitchnames_to_use, 'event_to_int': event_to_int_to_use, 'n_vocab': n_vocab_to_use}, f)
    else:
        print("\n--- Creating and Training New Model ---")
        model = create_network(SEQUENCE_LENGTH, training_n_vocab) # Train with current data vocab

        # Prepare seed for callback before training
        callback_seed_sequences = None
        possible_starts_for_callback = []
        for i in range(len(notes_corpus) - SEQUENCE_LENGTH):
            seq = notes_corpus[i:i+SEQUENCE_LENGTH]
            try:
                mapped_seq = [training_event_to_int[s] for s in seq] # Use training vocab
                possible_starts_for_callback.append(mapped_seq)
            except KeyError:
                pass 
        if possible_starts_for_callback:
            callback_seed_sequences = np.array(possible_starts_for_callback)
        else:
            print("Warning: Could not create seed for MIDI generation callback during new model training.")

        midi_callback = GenerateMidiCallback(
            output_folder=OUTPUT_FOLDER,
            sequence_length=SEQUENCE_LENGTH,
            num_notes_to_generate=NUM_NOTES_TO_GENERATE // 4, # Generate shorter samples
            pitchnames=training_pitchnames,
            event_to_int=training_event_to_int,
            n_vocab=training_n_vocab,
            seed_input_sequences=callback_seed_sequences
        )
        train_network(model, network_input, network_output, MODEL_FILE, custom_callbacks=[midi_callback])
        # After training, the model is based on training_vocab, so set generation params accordingly
        n_vocab_to_use = training_n_vocab
        pitchnames_to_use = training_pitchnames
        event_to_int_to_use = training_event_to_int
        with open(NOTES_FILE, 'wb') as f: # Save the vocab the new model was trained on
            pickle.dump({'pitchnames': pitchnames_to_use, 'event_to_int': event_to_int_to_use, 'n_vocab': n_vocab_to_use}, f)


    # 3. Generate Music
    # We need a sample of network_input that matches the vocabulary the model was trained on (n_vocab_to_use)
    # If the current dataset changed significantly, network_input (from current full data) might not be suitable
    # as a seed if its mapping doesn't align with event_to_int_to_use.
    # A safer seed comes from re-mapping a portion of the original corpus using the *loaded* event_to_int_to_use.
    
    # For simplicity, we'll use the network_input generated from the current dataset for seeding.
    # This is generally fine if the current dataset is what the model was trained on, or similar.
    # If you load a model trained on vastly different data, the seed might be suboptimal.
    # A more robust seeding would involve picking from `notes_corpus` and then mapping using `event_to_int_to_use`.
    
    # Let's create a seed pattern based on the `pitchnames_to_use` and `event_to_int_to_use`
    # which are either from the loaded vocab file or from the fresh training.
    
    # Pick a random starting sequence from the original notes_corpus and map it using event_to_int_to_use
    possible_starts = []
    for i in range(len(notes_corpus) - SEQUENCE_LENGTH):
        seq = notes_corpus[i:i+SEQUENCE_LENGTH]
        try:
            mapped_seq = [event_to_int_to_use[s] for s in seq]
            possible_starts.append(mapped_seq)
        except KeyError:
            pass # This sequence contains an event not in the model's vocabulary, skip it for seeding.
    
    if not possible_starts:
        print("Error: Could not create a valid seed sequence from current data using the model's vocabulary.")
        print("This can happen if the current dataset is very different from the one the model was trained on.")
        print("Consider retraining the model with the current dataset.")
        # As a last resort, create a dummy seed (less ideal)
        # Or, if network_input is available and its vocab matches n_vocab_to_use
        if len(network_input) > 0:
             print("Falling back to using a random sequence from the current dataset's network_input for seeding final generation.")
             seed_network_input_array = network_input
        else:
            print("Cannot generate final music, no valid seed.")
            exit()
    else:
        # This is better: uses actual data mapped to the model's vocabulary
        seed_network_input_array = np.array(possible_starts)


    generated_sequence = generate_music(
        model,
        pitchnames_to_use,      # Vocabulary (list of strings) the model was trained on
        event_to_int_to_use,    # Mapping for this vocabulary
        n_vocab_to_use,         # Size of this vocabulary
        SEQUENCE_LENGTH,
        NUM_NOTES_TO_GENERATE,
        seed_network_input_array if possible_starts else seed_network_input # The seed sequences
    )

    # 4. Create MIDI Output
    output_midi_file = os.path.join(OUTPUT_FOLDER, f"generated_instrumental_music_s{SEQUENCE_LENGTH}_e{EPOCHS}.mid")
    create_midi(generated_sequence, output_midi_file)

    print("\n--- Process Finished ---")