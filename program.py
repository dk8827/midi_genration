import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, articulations, dynamics, tempo, meter
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
            # Attempt to create a very basic seed if none provided
            if self.pitchnames:
                print("Callback: Attempting basic seed for MIDI generation...")
                default_seed_event_int = np.random.randint(0, self.n_vocab)
                pattern = [default_seed_event_int] * self.sequence_length
                self.seed_input_sequences = np.array([pattern]) # Make it a 2D array
            else:
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
        if generated_sequence: # Only create MIDI if generation was successful
            output_midi_file = os.path.join(self.output_folder, f"{self.base_filename}_{epoch+1}.mid")
            create_midi(generated_sequence, output_midi_file)
            print(f"Callback: Sample MIDI saved to {output_midi_file}")
        else:
            print("Callback: MIDI generation failed, no sequence produced.")

# --- Configuration ---
MIDI_DATA_PATH = "midi_songs/" # Path to your MIDI files
OUTPUT_FOLDER = "output_music/"
MODEL_FILE = os.path.join(OUTPUT_FOLDER, "music_generator_instrument_model.h5")
NOTES_FILE = os.path.join(OUTPUT_FOLDER, "notes_instrument_vocab.pkl")
PROCESSED_MIDI_EVENTS_FILE = os.path.join(OUTPUT_FOLDER, "processed_midi_events.pkl")

SEQUENCE_LENGTH = 25
EPOCHS = 50 # Can be increased for better results
BATCH_SIZE = 64
NUM_NOTES_TO_GENERATE = 200
DEFAULT_INSTRUMENT_NAME = "Piano" # Fallback instrument

# --- Helper Functions ---

INSTRUMENT_MAP = {
    # General MIDI Mappings (a selection, you can expand this)
    # Key: music21 instrument class name (or common name), Value: music21 object
    "Piano": instrument.Piano(),
    "Electric Piano": instrument.ElectricPiano(),
    "Harpsichord": instrument.Harpsichord(),
    "Clavinet": instrument.ElectricPiano(), # Replaced Clavinet
    "Celesta": instrument.Celesta(),
    "Glockenspiel": instrument.Glockenspiel(),
    "Music Box": instrument.Celesta(), # Often similar to Celesta
    "Vibraphone": instrument.Vibraphone(),
    "Marimba": instrument.Marimba(),
    "Xylophone": instrument.Xylophone(),
    "Tubular Bells": instrument.TubularBells(),
    "Dulcimer": instrument.Dulcimer(),
    "Electric Guitar": instrument.ElectricGuitar(), # Generic Electric Guitar
    "Acoustic Guitar": instrument.AcousticGuitar(), # Generic Acoustic Guitar
    "Violin": instrument.Violin(),
    "Viola": instrument.Viola(),
    "Cello": instrument.Violoncello(), # music21 uses Violoncello
    "Contrabass": instrument.Contrabass(),
    "Electric Bass": instrument.ElectricBass(),
    "StringEnsemble": instrument.StringInstrument(), # Generic string ensemble
    "SynthStrings": instrument.ElectricPiano(), # Replaced Synthesizer
    "Voice": instrument.Choir(), # Or Vocalist
    "Synth Voice": instrument.Choir(), # Replaced SynthVoice
    "Orchestra Hit": instrument.Sampler(), # Ensemble can work too
    "Trumpet": instrument.Trumpet(),
    "Trombone": instrument.Trombone(),
    "Tuba": instrument.Tuba(),
    "French Horn": instrument.Horn(),
    "BrassSection": instrument.BrassInstrument(), # Generic brass
    "Soprano Sax": instrument.SopranoSaxophone(),
    "Alto Sax": instrument.AltoSaxophone(),
    "Tenor Sax": instrument.TenorSaxophone(),
    "Baritone Sax": instrument.BaritoneSaxophone(),
    "Saxophone": instrument.Saxophone(), # General Saxophone
    "Oboe": instrument.Oboe(),
    "English Horn": instrument.EnglishHorn(),
    "Bassoon": instrument.Bassoon(),
    "Clarinet": instrument.Clarinet(),
    "Piccolo": instrument.Piccolo(),
    "Flute": instrument.Flute(),
    "Recorder": instrument.Recorder(),
    "Pan Flute": instrument.PanFlute(),
    "Shakuhachi": instrument.Shakuhachi(),
    "Whistle": instrument.Whistle(),
    "Ocarina": instrument.Ocarina(),
    "Synth": instrument.ElectricPiano(), # Replaced Synthesizer
    # Percussion - Music21 handles percussion specially, often in PercussionChord
    "Drums": instrument.Percussion(), # General percussion group
    "Acoustic Bass Drum": instrument.BassDrum(),
    "Bass Drum 1": instrument.BassDrum(),
    "Side Stick": instrument.SnareDrum(), # Snare can have side stick sound
    "Acoustic Snare": instrument.SnareDrum(),
    "Hand Clap": instrument.SnareDrum(), # Replaced HandClap
    "Electric Snare": instrument.SnareDrum(),
    "Low Floor Tom": instrument.TomTom(),
    "Closed Hi Hat": instrument.HiHatCymbal(),
    "High Floor Tom": instrument.TomTom(),
    "Pedal Hi Hat": instrument.HiHatCymbal(),
    "Low Tom": instrument.TomTom(),
    "Open Hi Hat": instrument.HiHatCymbal(),
    "Low-Mid Tom": instrument.TomTom(),
    "Hi-Mid Tom": instrument.TomTom(),
    "Crash Cymbal 1": instrument.CrashCymbals(),
    "High Tom": instrument.TomTom(),
    "Ride Cymbal 1": instrument.RideCymbals(),
    "Chinese Cymbal": instrument.Cymbals(),
    "Ride Bell": instrument.RideCymbals(),
    "Tambourine": instrument.Tambourine(),
    "Splash Cymbal": instrument.SplashCymbals(),
    "Cowbell": instrument.Cowbell(),
    "Crash Cymbal 2": instrument.CrashCymbals(),
    "Vibraslap": instrument.Vibraslap(),
    "Ride Cymbal 2": instrument.RideCymbals(),
    "Hi Bongo": instrument.BongoDrums(),
    "Low Bongo": instrument.BongoDrums(),
    "Mute Hi Conga": instrument.CongaDrum(),
    "Open Hi Conga": instrument.CongaDrum(),
    "Low Conga": instrument.CongaDrum(),
    "High Timbale": instrument.Timbales(),
    "Low Timbale": instrument.Timbales(),
    "High Agogo": instrument.Agogo(),
    "Low Agogo": instrument.Agogo(),
    "Cabasa": instrument.Maracas(), # Replaced Cabasa
    "Maracas": instrument.Maracas(),
    "Short Whistle": instrument.Whistle(),
    "Long Whistle": instrument.Whistle(),
    "Short Guiro": instrument.Woodblock(), # Replaced Guiro
    "Long Guiro": instrument.Woodblock(), # Replaced Guiro
    "Claves": instrument.Woodblock(), # Replaced Claves
    "Hi Wood Block": instrument.Woodblock(),
    "Low Wood Block": instrument.Woodblock(),
    "Mute Cuica": instrument.TomTom(), # Replaced Cuica
    "Open Cuica": instrument.TomTom(), # Replaced Cuica
    "Mute Triangle": instrument.Triangle(),
    "Open Triangle": instrument.Triangle(),
}

def get_instrument_from_name(name_str):
    """Attempts to get a music21 instrument object from its name."""
    if name_str in INSTRUMENT_MAP:
        # Create a new instance to avoid sharing state across notes/parts
        return INSTRUMENT_MAP[name_str].__class__()

    # Try music21's built-in string parsing (less reliable for specific program names)
    try:
        cleaned_name = re.sub(r'\d+', '', name_str).strip() # Remove numbers
        instr_obj = instrument.fromString(cleaned_name)
        if instr_obj:
            # Check if it's a generic Instrument class, if so, try our map for something more specific
            if instr_obj.__class__ == instrument.Instrument:
                pass # Fall through to more specific mapping
            else:
                return instr_obj
    except:
        pass

    # Fallback for common patterns music21 might not catch directly from program names
    name_lower = name_str.lower()
    if "piano" in name_lower: return instrument.Piano()
    if "guitar" in name_lower: return instrument.AcousticGuitar() # Default to acoustic
    if "violin" in name_lower: return instrument.Violin()
    if "flute" in name_lower: return instrument.Flute()
    if "drum" in name_lower or "percussion" in name_lower: return instrument.Percussion()
    if "sax" in name_lower: return instrument.Saxophone()
    if "trumpet" in name_lower: return instrument.Trumpet()
    if "bass" in name_lower and "drum" not in name_lower : return instrument.ElectricBass() # Default to electric bass

    print(f"Warning: Could not map instrument name '{name_str}' reliably. Defaulting to Piano.")
    return instrument.Piano()


def get_instrument_name(m21_instrument):
    """Gets a simplified, consistent name for a music21 instrument object."""
    if not m21_instrument:
        return DEFAULT_INSTRUMENT_NAME

    original_name_str = None
    if hasattr(m21_instrument, 'instrumentName') and m21_instrument.instrumentName is not None:
        original_name_str = str(m21_instrument.instrumentName)
    elif hasattr(m21_instrument, 'bestName') and callable(m21_instrument.bestName):
        name_from_best = m21_instrument.bestName()
        if name_from_best is not None:
            original_name_str = str(name_from_best)
    
    if not original_name_str: 
        original_name_str = str(m21_instrument.__class__.__name__)
    
    if not original_name_str:
        return DEFAULT_INSTRUMENT_NAME

    temp_name = re.sub(r'\s*\d+\s*$', '', original_name_str).strip()
    temp_name_lower = temp_name.lower()

    for k_map in INSTRUMENT_MAP.keys():
        if k_map.lower() == temp_name_lower:
            return k_map 

    name_l = original_name_str.lower() 

    if "harpsichord" in name_l: return "Harpsichord"
    if "celesta" in name_l: return "Celesta"
    if "glockenspiel" in name_l: return "Glockenspiel"
    if "vibraphone" in name_l: return "Vibraphone"
    if "marimba" in name_l: return "Marimba"
    if "xylophone" in name_l: return "Xylophone"
    if "tubular bells" in name_l: return "Tubular Bells"
    if "dulcimer" in name_l: return "Dulcimer"
    if "clavinet" in name_l: return "Clavinet"
    if "electric grand piano" in name_l or "electric piano" in name_l or "rhodes" in name_l or "wurly" in name_l: return "Electric Piano"
    if "piano" in name_l: return "Piano"
    if "electric guitar" in name_l or "jazz gtr" in name_l or "clean gtr" in name_l or "mute gtr" in name_l or "dist gtr" in name_l: return "Electric Guitar"
    if "acoustic guitar" in name_l or "nylon" in name_l or "steel" in name_l : return "Acoustic Guitar"
    if "guitar" in name_l: return "Guitar"
    if "violin" in name_l: return "Violin"
    if "viola" in name_l: return "Viola"
    if "cello" in name_l or "violoncello" in name_l: return "Cello"
    if ("electric bass" in name_l or "finger" in name_l or "pick" in name_l or "fretless" in name_l) and "bass" in name_l : return "Electric Bass"
    if "acoustic bass" in name_l or "contrabass" in name_l or "double bass" in name_l : return "Contrabass"
    if "bass" in name_l and "drum" not in name_l: return "Bass"
    if "flute" in name_l: return "Flute"
    if "piccolo" in name_l: return "Piccolo"
    if "recorder" in name_l: return "Recorder"
    if "pan flute" in name_l: return "Pan Flute"
    if "ocarina" in name_l: return "Ocarina"
    if "shakuhachi" in name_l: return "Shakuhachi"
    if "whistle" in name_l: return "Whistle"
    if "clarinet" in name_l: return "Clarinet"
    if "oboe" in name_l: return "Oboe"
    if "bassoon" in name_l: return "Bassoon"
    if "english horn" in name_l: return "English Horn"
    if "soprano sax" in name_l: return "Soprano Sax"
    if "alto sax" in name_l: return "Alto Sax"
    if "tenor sax" in name_l: return "Tenor Sax"
    if "baritone sax" in name_l: return "Baritone Sax"
    if "sax" in name_l: return "Saxophone"
    if "trumpet" in name_l: return "Trumpet"
    if "trombone" in name_l: return "Trombone"
    if "tuba" in name_l: return "Tuba"
    if "french horn" in name_l or ("horn" in name_l and "english" not in name_l): return "French Horn"
    if "brass section" in name_l or "brass ensemble" in name_l: return "BrassSection"
    if "string ensemble" in name_l or "strings" in name_l and "synth" not in name_l: return "StringEnsemble"
    if "voice" in name_l or "choir" in name_l or "vocal" in name_l or "aah" in name_l or "ooh" in name_l and "synth" not in name_l: return "Voice"
    if "synth voice" in name_l or "synth choir" in name_l : return "Synth Voice"
    if "synth lead" in name_l or "synth pad" in name_l or "synth brass" in name_l or "synth strings" in name_l or "polysynth" in name_l or "fx " in name_l: return "Synth"
    if "synth" in name_l : return "Synth"
    
    for k_map_perc in INSTRUMENT_MAP.keys():
        is_percussion_map_key = any(p_term in k_map_perc.lower() for p_term in ["drum", "cymbal", "tom", "hi-hat", "snare", "kick", "percussion", "bongo", "conga", "timbale", "cowbell", "tambourine", "claves", "wood", "agogo", "guiro", "maracas", "triangle"])
        if is_percussion_map_key and k_map_perc.lower() in name_l:
            return k_map_perc

    if any(p_term in name_l for p_term in ["drum", "percussion", "cymbal", "tom", "hat", "snare", "kick", "conga", "bongo", "timbale", "agogo", "woodblock", "claves", "guiro", "maracas", "triangle"]):
        return "Drums"

    final_name = re.sub(r'\d+', '', original_name_str).strip()
    final_name = re.sub(r'\s+', ' ', final_name) 
    final_name = final_name.replace("Instrument", "").strip()
    final_name = ''.join(char for char in final_name if char.isalnum() or char.isspace() or char in ['-', '_', '(', ')'])
    final_name = final_name.strip()

    if not final_name or final_name.lower() == "instrument" or len(final_name) < 2:
        class_name = m21_instrument.__class__.__name__
        # Avoid returning "Instrument" if possible
        if class_name and class_name != "Instrument" and class_name != "UnpitchedPercussion" and class_name != "PitchedPercussion":
            # Basic class name might be better than default
            # Try to map class name to our map
            for k_map, v_map_obj in INSTRUMENT_MAP.items():
                if v_map_obj.__class__.__name__ == class_name:
                    return k_map
            return class_name # e.g. Piano, Violin
        return DEFAULT_INSTRUMENT_NAME
        
    return final_name


def get_notes_from_midi_files(data_path):
    notes_with_instruments = []
    for file_path in glob.glob(data_path + "*.mid"):
        try:
            midi = converter.parse(file_path)
            print(f"Parsing {file_path}")
            
            parts = instrument.partitionByInstrument(midi)
            if parts:
                for part in parts:
                    instr = part.getInstrument()
                    # Handle cases where getInstrument might return None or a generic object
                    if instr is None and len(part.getElementsByClass(instrument.Instrument)) > 0 :
                        instr = part.getElementsByClass(instrument.Instrument)[0]

                    instr_name = get_instrument_name(instr)
                    # print(f"  Found instrument: {instr_name} (from {instr}) in part {part.id if part.id else 'N/A'}")

                    for element in part.recurse().notesAndRests:
                        event_str = None
                        if isinstance(element, note.Note):
                            event_str = str(element.pitch)
                        elif isinstance(element, chord.Chord):
                            event_str = '.'.join(str(n) for n in element.normalOrder)
                        elif isinstance(element, note.Rest):
                            event_str = "Rest"
                        
                        if event_str:
                            notes_with_instruments.append(f"{instr_name}:{event_str}")
            else: 
                current_instrument_name = DEFAULT_INSTRUMENT_NAME
                instr_at_start = midi.flat.getElementsByClass(instrument.Instrument).first()
                if instr_at_start:
                    current_instrument_name = get_instrument_name(instr_at_start)
                
                for element in midi.flat.notesAndRests:
                    event_str = None
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
            
            if not notes_with_instruments and not (parts or midi.flat.notesAndRests): # Check if any notes were actually added from THIS file
                 print(f"    No notes or parts found or successfully processed in {file_path}")

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for parsing errors
    
    print(f"Total instrumented events extracted: {len(notes_with_instruments)}")
    if not notes_with_instruments:
        print("Warning: No notes were extracted. Check your MIDI files and parsing logic.")
    return notes_with_instruments

def prepare_sequences(notes_with_instruments, n_vocab, sequence_length):
    pitchnames = sorted(list(set(notes_with_instruments)))
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
        raise ValueError("No patterns generated. Increase data, decrease sequence_length, or check data processing.")

    network_input = np.array(network_input_list)
    network_output = to_categorical(network_output_list, num_classes=n_vocab)

    return (network_input, network_output, pitchnames, event_to_int)

def create_network(seq_length, n_vocab):
    model = Sequential()
    model.add(Embedding(n_vocab, 256, input_shape=(seq_length,))) # Increased embedding for larger vocab
    model.add(LSTM(256, return_sequences=True)) 
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256)) 
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_network(model, network_input, network_output, model_file, custom_callbacks=None):
    filepath = model_file
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10, # Increased patience slightly
        verbose=1,
        restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stopping]
    if custom_callbacks:
        if isinstance(custom_callbacks, list):
            callbacks_list.extend(custom_callbacks)
        else:
            callbacks_list.append(custom_callbacks)

    history = model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    print(f"Training complete. Model saved to {model_file}")
    return history

def generate_music(model, pitchnames, event_to_int, n_vocab, sequence_length, num_events_to_generate, network_input_for_start):
    int_to_event = dict((number, event) for number, event in enumerate(pitchnames))

    if network_input_for_start is None or len(network_input_for_start) == 0:
        print("Error: No seed patterns available for generation.")
        if not pitchnames or n_vocab == 0:
            print("No pitchnames/vocab available for default seed. Cannot generate.")
            return []
        print("Generating a default random seed (less ideal)...")
        default_seed_event_int = np.random.randint(0, n_vocab)
        pattern = [default_seed_event_int] * sequence_length
    else:
        start_index = np.random.randint(0, len(network_input_for_start))
        pattern = network_input_for_start[start_index].tolist()

    prediction_output = []
    print(f"\n--- Generating Music (Seed: {pattern[:5]}...)---")

    for event_index in range(num_events_to_generate):
        prediction_input = np.reshape(pattern, (1, len(pattern)))
        prediction_probs = model.predict(prediction_input, verbose=0)[0]

        temperature = 0.85 
        prediction_probs = np.asarray(prediction_probs).astype('float64')
        
        # Prevent taking log of 0
        prediction_probs_safe = np.log(prediction_probs + 1e-8) / temperature
        exp_preds = np.exp(prediction_probs_safe - np.max(prediction_probs_safe)) # Stability trick: subtract max before exp

        if np.isinf(exp_preds).any() or np.isnan(exp_preds).any() or np.sum(exp_preds) == 0:
            # print(f"Warning: exp_preds resulted in inf/nan or sum zero at event {event_index}. Using argmax.")
            index = np.argmax(model.predict(prediction_input, verbose=0)[0])
        else:
            prediction_probs_norm = exp_preds / np.sum(exp_preds)
            # Ensure probabilities sum to 1 due to potential floating point inaccuracies
            prediction_probs_norm = prediction_probs_norm / np.sum(prediction_probs_norm) 
            
            try:
                index = np.random.choice(len(prediction_probs_norm), p=prediction_probs_norm)
            except ValueError as e: 
                # print(f"Warning: np.random.choice failed. Sum(p): {np.sum(prediction_probs_norm)}. Error: {e}. Using argmax.")
                index = np.argmax(model.predict(prediction_input, verbose=0)[0])
        
        result = int_to_event[index]
        prediction_output.append(result)
        # print(f"Generated ({event_index+1}/{num_events_to_generate}): {result}") # Can be very verbose

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    
    print(f"Finished generating {num_events_to_generate} events.")
    return prediction_output

def create_midi(prediction_output, output_file="output_instrumental.mid"):
    score = stream.Score()
    ts = meter.TimeSignature('4/4')
    score.insert(0, ts)
    tm = tempo.MetronomeMark(number=120) # Default tempo
    score.insert(0, tm)

    parts = {} 
    part_offsets = {} 

    current_global_offset = 0.0
    last_event_duration = 0.5 # Keep track of last duration to advance offset

    for i, pattern_item in enumerate(prediction_output):
        try:
            if ':' not in pattern_item:
                print(f"Skipping malformed item (no ':'): {pattern_item}")
                continue

            instrument_name_str, event_str = pattern_item.split(':', 1)

            if instrument_name_str not in parts:
                m21_instr_obj = get_instrument_from_name(instrument_name_str)
                new_part = stream.Part(id=instrument_name_str)
                new_part.insert(0, m21_instr_obj)
                parts[instrument_name_str] = new_part
                # part_offsets[instrument_name_str] = 0.0 # We'll use global offset primarily
                score.insert(0, new_part)

            current_part = parts[instrument_name_str]
            
            m21_event = None
            duration = 0.5 # Default duration; could be learned or varied

            if event_str == "Rest":
                m21_event = note.Rest()
                m21_event.duration.quarterLength = duration
            elif ('.' in event_str) or event_str.isdigit(): 
                notes_in_chord_pitches = event_str.split('.')
                chord_notes_obj = []
                for p_str in notes_in_chord_pitches:
                    try:
                        # Chords from normalOrder are pitch numbers (MIDI note numbers)
                        n_obj = note.Note(int(p_str)) 
                        chord_notes_obj.append(n_obj)
                    except Exception as e_chord_note:
                        # print(f"Warning: Could not parse pitch '{p_str}' in chord '{event_str}'. Skipping. Error: {e_chord_note}")
                        continue
                if chord_notes_obj:
                    m21_event = chord.Chord(chord_notes_obj)
                    m21_event.duration.quarterLength = duration
            else: 
                try:
                    m21_event = note.Note(event_str) # event_str should be like "C#4"
                    m21_event.duration.quarterLength = duration
                except Exception as e_note:
                    # print(f"Warning: Could not parse note '{event_str}'. Skipping. Error: {e_note}")
                    continue
            
            if m21_event:
                current_part.insert(current_global_offset, m21_event)
                last_event_duration = m21_event.duration.quarterLength
            else: # If event wasn't created (e.g. parse error), use default last duration
                last_event_duration = 0.5 
            
            current_global_offset += last_event_duration # Advance by the duration of the *just added* event

        except Exception as e:
            print(f"Error processing generated item '{pattern_item}' at index {i}: {e}")
            continue
            
    if not score.elements or not any(isinstance(el, stream.Part) and len(el.notesAndRests) > 0 for el in score.elements):
        print("No valid music elements were added to the score. MIDI will be empty or contain only metadata.")
        return

    try:
        # Clean up empty parts before writing
        for part_to_remove in [p for p in score.getElementsByClass(stream.Part) if not p.notesAndRests]:
            score.remove(part_to_remove)
            print(f"Removed empty part: {part_to_remove.id}")

        if not score.getElementsByClass(stream.Part):
             print("No parts with notes remaining. MIDI will be empty.")
             return

        score.write('midi', fp=output_file)
        print(f"Multi-instrument MIDI file saved as {output_file}")
    except Exception as e:
        print(f"Error writing MIDI file: {e}")
        # score.show('text')


if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("--- Loading and Preprocessing Data ---")
    if os.path.exists(PROCESSED_MIDI_EVENTS_FILE):
        print(f"Loading pre-processed MIDI events from {PROCESSED_MIDI_EVENTS_FILE}...")
        try:
            with open(PROCESSED_MIDI_EVENTS_FILE, 'rb') as f:
                notes_corpus = pickle.load(f)
            print("Pre-processed MIDI events loaded.")
            if not notes_corpus:
                 print("Loaded pre-processed data is empty. Reparsing.")
                 notes_corpus = None # Force re-parse
        except EOFError:
            print("EOFError loading pre-processed data. File might be corrupted. Reparsing.")
            notes_corpus = None # Force re-parse


    if not os.path.exists(PROCESSED_MIDI_EVENTS_FILE) or notes_corpus is None: # Added notes_corpus is None check
        print(f"No valid pre-processed data found. Parsing MIDI files from {MIDI_DATA_PATH}...")
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

    # --- Vocabulary Handling ---
    # Base vocabulary on the current notes_corpus
    current_pitchnames = sorted(list(set(notes_corpus)))
    current_n_vocab = len(current_pitchnames)
    current_event_to_int = {event: i for i, event in enumerate(current_pitchnames)}
    print(f"Current dataset vocabulary size: {current_n_vocab}")

    # These will hold the vocabulary to be used by the model (either loaded or current)
    pitchnames_for_model = current_pitchnames
    n_vocab_for_model = current_n_vocab
    event_to_int_for_model = current_event_to_int

    # If a model and its vocab file exist, prefer the loaded vocabulary for consistency
    # especially if we are just generating, not retraining.
    if os.path.exists(MODEL_FILE) and os.path.exists(NOTES_FILE):
        print(f"Model file {MODEL_FILE} and notes file {NOTES_FILE} exist. Attempting to load vocabulary.")
        try:
            with open(NOTES_FILE, 'rb') as f:
                data = pickle.load(f)
                loaded_pitchnames = data['pitchnames']
                loaded_event_to_int = data['event_to_int']
                loaded_n_vocab = data['n_vocab']
            
            print(f"Loaded vocabulary size from file: {loaded_n_vocab}")
            if loaded_n_vocab != current_n_vocab:
                 print(f"Warning: Vocabulary size mismatch! Loaded: {loaded_n_vocab}, Current Dataset: {current_n_vocab}.")
                 print("Using loaded vocabulary for model operations.")
            
            pitchnames_for_model = loaded_pitchnames
            n_vocab_for_model = loaded_n_vocab
            event_to_int_for_model = loaded_event_to_int
        except Exception as e:
            print(f"Error loading {NOTES_FILE}, will use current dataset's vocab: {e}")
            # If loading fails, stick with current_*. We'll save this later if we train.
            pass # pitchnames_for_model etc. already set to current_*

    # Prepare sequences for training using the current dataset's full vocabulary
    # This is always done, as even if we load an old model, we might decide to retrain it.
    network_input, network_output, _, _ = prepare_sequences(
        notes_corpus, current_n_vocab, SEQUENCE_LENGTH
    )

    # Determine if retraining is needed or if a new model will be trained
    train_new_model_flag = True # Assume new model unless existing one loads successfully
    if os.path.exists(MODEL_FILE):
        print(f"\n--- Loading existing model from {MODEL_FILE} ---")
        try:
            model = load_model(MODEL_FILE)
            print("Model loaded successfully.")
            # Critical: Ensure n_vocab_for_model matches the loaded model's output layer
            model_output_vocab_size = model.layers[-1].output_shape[-1]
            if model_output_vocab_size != n_vocab_for_model:
                print(f"CRITICAL MISMATCH: Loaded model expects vocab of size {model_output_vocab_size}, but loaded notes file implies {n_vocab_for_model}.")
                print("This likely means the notes file is out of sync with the model. Consider deleting both and retraining.")
                print("Attempting to proceed with vocab from notes file, but generation might fail or be incorrect.")
                # OR: Force retraining if mismatch
                # raise Exception("Model and vocab mismatch, cannot proceed safely.")
            train_new_model_flag = False # Model loaded, no need to train new unless forced
        except Exception as e:
            print(f"Error loading model: {e}. Will train a new model with current data.")
            train_new_model_flag = True # Force training new model

    if train_new_model_flag:
        print("\n--- Creating and Training New Model ---")
        # If training new, the model's vocabulary IS the current dataset's vocabulary
        pitchnames_for_model = current_pitchnames
        n_vocab_for_model = current_n_vocab
        event_to_int_for_model = current_event_to_int
        
        # Save this new vocabulary as it's what the new model will be trained on
        with open(NOTES_FILE, 'wb') as f:
            pickle.dump({'pitchnames': pitchnames_for_model, 
                         'event_to_int': event_to_int_for_model, 
                         'n_vocab': n_vocab_for_model}, f)
        print(f"Vocabulary for new model (size {n_vocab_for_model}) saved to {NOTES_FILE}")

        model = create_network(SEQUENCE_LENGTH, n_vocab_for_model)
        
        # Prepare seed for callback using the vocabulary the model is being trained with
        callback_seed_sequences_list = []
        for i in range(len(notes_corpus) - SEQUENCE_LENGTH):
            seq = notes_corpus[i:i+SEQUENCE_LENGTH]
            try:
                # Map using the vocabulary for this new model
                mapped_seq = [event_to_int_for_model[s] for s in seq]
                callback_seed_sequences_list.append(mapped_seq)
            except KeyError: # Should not happen if event_to_int_for_model is from current_notes_corpus
                pass 
        
        callback_seed_sequences_np = None
        if callback_seed_sequences_list:
            callback_seed_sequences_np = np.array(callback_seed_sequences_list)
        else:
            print("Warning: Could not create seed for MIDI generation callback during new model training.")

        midi_callback = GenerateMidiCallback(
            output_folder=OUTPUT_FOLDER,
            sequence_length=SEQUENCE_LENGTH,
            num_notes_to_generate=max(50, NUM_NOTES_TO_GENERATE // 4), # Generate at least 50, or 1/4
            pitchnames=pitchnames_for_model,
            event_to_int=event_to_int_for_model,
            n_vocab=n_vocab_for_model,
            seed_input_sequences=callback_seed_sequences_np
        )
        train_network(model, network_input, network_output, MODEL_FILE, custom_callbacks=[midi_callback])

    # --- Generate Music ---
    # Prepare seed for final generation using the model's vocabulary (pitchnames_for_model, etc.)
    
    # We need network_input_for_start mapped with event_to_int_for_model
    # `network_input` was prepared with `current_event_to_int`.
    # If `event_to_int_for_model` is different (e.g. from a loaded model), we must remap.
    
    final_seed_sequences_list = []
    if event_to_int_for_model == current_event_to_int:
        # If vocabularies match, we can directly use network_input (which is already integer mapped)
        if len(network_input) > 0:
             final_seed_sequences_list = network_input.tolist() # network_input is already a list of lists of ints
        else:
            print("Warning: network_input is empty, cannot derive seed sequences directly.")
    else:
        # Vocabularies differ, so remap from original notes_corpus using event_to_int_for_model
        print("Remapping notes_corpus for final generation seed due to vocabulary difference.")
        for i in range(len(notes_corpus) - SEQUENCE_LENGTH):
            seq = notes_corpus[i:i+SEQUENCE_LENGTH]
            try:
                mapped_seq = [event_to_int_for_model[s] for s in seq]
                final_seed_sequences_list.append(mapped_seq)
            except KeyError:
                # This sequence contains an event not in the model's vocabulary, skip it for seeding.
                pass
    
    final_seed_sequences_np = None
    if final_seed_sequences_list:
        final_seed_sequences_np = np.array(final_seed_sequences_list)
    else:
        print("Error: Could not create any valid seed sequences for final generation.")
        print("This can happen if the current dataset is very different from the one the model was trained on,")
        print("and no sequences from the current data can be mapped to the model's vocabulary.")
        print("Attempting a very basic default seed for generation, or generation might fail.")
        # `generate_music` has its own fallback if final_seed_sequences_np is None or empty

    generated_sequence = generate_music(
        model,
        pitchnames_for_model,
        event_to_int_for_model,
        n_vocab_for_model,
        SEQUENCE_LENGTH,
        NUM_NOTES_TO_GENERATE,
        final_seed_sequences_np 
    )

    if generated_sequence:
        output_midi_file = os.path.join(OUTPUT_FOLDER, f"generated_instrumental_music_s{SEQUENCE_LENGTH}_e{EPOCHS}_t085.mid")
        create_midi(generated_sequence, output_midi_file)
    else:
        print("Final music generation failed, no sequence produced.")

    print("\n--- Process Finished ---")