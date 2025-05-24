import os
from music21 import instrument

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

# --- Instrument Mapping ---
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