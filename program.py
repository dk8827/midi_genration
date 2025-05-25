import pickle
import numpy as np
import os

# Import configurations
from config import (
    MIDI_DATA_PATH, OUTPUT_FOLDER, MODEL_FILE, NOTES_FILE, 
    PROCESSED_MIDI_EVENTS_FILE, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, 
    NUM_NOTES_TO_GENERATE, DEFAULT_INSTRUMENT_NAME, TIME_RESOLUTION
)

# Import classes
from midi_processor import MidiProcessor
from music_generation_model import MusicGenerationModel
from midi_file_writer import MidiFileWriter
from generate_midi_callback import GenerateMidiCallback

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    processor = MidiProcessor(data_path=MIDI_DATA_PATH, default_instrument_name=DEFAULT_INSTRUMENT_NAME, time_resolution=TIME_RESOLUTION)
    
    # Calculate median tempo from the dataset
    median_dataset_tempo = processor.get_median_tempo_from_dataset()
    
    writer = MidiFileWriter(midi_processor=processor, 
                            default_instrument_name=DEFAULT_INSTRUMENT_NAME,
                            target_tempo=median_dataset_tempo)

    print("--- Loading and Preprocessing Data ---")
    notes_corpus = None
    if os.path.exists(PROCESSED_MIDI_EVENTS_FILE):
        print(f"Loading pre-processed MIDI events from {PROCESSED_MIDI_EVENTS_FILE}...")
        try:
            with open(PROCESSED_MIDI_EVENTS_FILE, 'rb') as f:
                notes_corpus = pickle.load(f)
            print("Pre-processed MIDI events loaded.")
            if not notes_corpus:
                 print("Loaded pre-processed data is empty. Reparsing.")
                 notes_corpus = None
        except EOFError:
            print("EOFError loading pre-processed data. File might be corrupted. Reparsing.")
            notes_corpus = None
        except Exception as e: # Catch other potential pickle errors
            print(f"Error loading pre-processed data: {e}. Reparsing.")
            notes_corpus = None

    if notes_corpus is None:
        print(f"Parsing MIDI files from {MIDI_DATA_PATH}...")
        notes_corpus = processor.get_notes_from_midi_files()
        if notes_corpus:
            try:
                with open(PROCESSED_MIDI_EVENTS_FILE, 'wb') as f:
                    pickle.dump(notes_corpus, f)
                print(f"Parsed MIDI events saved to {PROCESSED_MIDI_EVENTS_FILE}")
            except Exception as e:
                print(f"Error saving parsed MIDI events: {e}")
        else:
            print(f"No notes found after parsing. Pre-processed file '{PROCESSED_MIDI_EVENTS_FILE}' not created/updated.")

    if not notes_corpus:
        print(f"CRITICAL: No notes available. Please check your MIDI files in '{MIDI_DATA_PATH}'. Exiting.")
        exit()

    # Analyze the time-aware encoding
    processor.analyze_time_aware_encoding(notes_corpus)

    current_pitchnames = sorted(list(set(notes_corpus)))
    current_n_vocab = len(current_pitchnames)
    current_event_to_int = {event: i for i, event in enumerate(current_pitchnames)}
    print(f"Current dataset vocabulary size: {current_n_vocab}")

    pitchnames_for_model = current_pitchnames
    n_vocab_for_model = current_n_vocab
    event_to_int_for_model = current_event_to_int

    music_model_instance = None 

    if os.path.exists(MODEL_FILE) and os.path.exists(NOTES_FILE):
        print(f"Found existing model ({MODEL_FILE}) and notes file ({NOTES_FILE}). Attempting to load.")
        try:
            with open(NOTES_FILE, 'rb') as f:
                data = pickle.load(f)
                loaded_pitchnames = data['pitchnames']
                loaded_event_to_int = data['event_to_int']
                loaded_n_vocab = data['n_vocab']
            print(f"Loaded vocabulary details from notes file: size {loaded_n_vocab}")
            
            music_model_instance = MusicGenerationModel(SEQUENCE_LENGTH, loaded_n_vocab)
            music_model_instance.load_model(MODEL_FILE) 

            if music_model_instance.n_vocab != loaded_n_vocab:
                 print(f"Warning: Vocabulary size mismatch! Model expected {music_model_instance.n_vocab} (from Keras model), notes file had {loaded_n_vocab}.")
                 print("Using Keras model's actual vocabulary size. Ensure notes file is consistent.")
            
            pitchnames_for_model = loaded_pitchnames
            n_vocab_for_model = music_model_instance.n_vocab 
            event_to_int_for_model = loaded_event_to_int 

            if len(loaded_event_to_int) != loaded_n_vocab:
                 print(f"Warning: Loaded event_to_int map has {len(loaded_event_to_int)} items, but notes file n_vocab is {loaded_n_vocab}.")
            if n_vocab_for_model > 0 and max(loaded_event_to_int.values()) >= n_vocab_for_model:
                print(f"CRITICAL Warning: Loaded event_to_int map contains indices >= model's n_vocab ({n_vocab_for_model}). This will likely cause errors.")
                print("Consider retraining or fixing the vocabulary files. Attempting to rebuild event_to_int_for_model based on loaded_pitchnames and model's n_vocab.")
                event_to_int_for_model = {event: i for i, event in enumerate(loaded_pitchnames) if i < n_vocab_for_model}
                if len(event_to_int_for_model) != n_vocab_for_model and len(loaded_pitchnames) >= n_vocab_for_model:
                    print("Warning: Rebuilt event_to_int_for_model size does not match n_vocab_for_model. This could indicate issues.")

            print("Model and vocabulary loaded.")
        except Exception as e:
            print(f"Error loading existing model/notes file: {e}. Training a new model.")
            music_model_instance = None
    
    # Prepare sequences for training using the current dataset's full vocabulary
    # This is always done, as even if we load an old model, we might decide to retrain it with current data (though not implemented here)
    # Or, it's used for training a new model.
    network_input, network_output, _, _ = processor.prepare_sequences(
        notes_corpus, current_n_vocab, SEQUENCE_LENGTH # Use vocab derived from current full notes_corpus
    )
    
    train_new_model_flag = music_model_instance is None

    if train_new_model_flag:
        print("\n--- Creating and Training New Model ---")
        pitchnames_for_model = current_pitchnames
        n_vocab_for_model = current_n_vocab
        event_to_int_for_model = current_event_to_int
        
        music_model_instance = MusicGenerationModel(SEQUENCE_LENGTH, n_vocab_for_model)
        
        try:
            with open(NOTES_FILE, 'wb') as f:
                pickle.dump({'pitchnames': pitchnames_for_model, 
                             'event_to_int': event_to_int_for_model, 
                             'n_vocab': n_vocab_for_model}, f)
            print(f"Vocabulary for new model (size {n_vocab_for_model}) saved to {NOTES_FILE}")
        except Exception as e:
            print(f"Error saving vocabulary for new model: {e}")

        # For callback seed, use network_input which is already int-mapped based on current_vocab
        # Since this is a new model, current_vocab IS the model's vocab.
        callback_seed_sequences_np = None
        if network_input is not None and network_input.size > 0:
            callback_seed_sequences_np = np.array(network_input.tolist()) # network_input is already integer mapped
        else:
            print("Warning: network_input is empty, cannot create diverse seeds for MIDI generation callback.")
            # Fallback for callback seed if network_input is empty
            if pitchnames_for_model and n_vocab_for_model > 0:
                default_seed_int = np.random.randint(0, n_vocab_for_model)
                pattern = [default_seed_int] * SEQUENCE_LENGTH
                callback_seed_sequences_np = np.array([pattern])
                print("Using a single random seed for callback due to empty network_input.")

        if callback_seed_sequences_np is not None and len(callback_seed_sequences_np) > 0:
            midi_callback = GenerateMidiCallback(
                output_folder=OUTPUT_FOLDER,
                sequence_length=SEQUENCE_LENGTH,
                num_notes_to_generate=max(50, NUM_NOTES_TO_GENERATE // 4),
                pitchnames=pitchnames_for_model, 
                event_to_int=event_to_int_for_model,
                n_vocab=n_vocab_for_model,
                seed_input_sequences=callback_seed_sequences_np,
                music_generator_instance=music_model_instance, 
                midi_file_writer=writer
            )
            music_model_instance.train(network_input, network_output, MODEL_FILE, EPOCHS, BATCH_SIZE, custom_callbacks=[midi_callback])
        else:
            print("Critical: No seed sequences available for callback. Training without MIDI generation callback.")
            music_model_instance.train(network_input, network_output, MODEL_FILE, EPOCHS, BATCH_SIZE)

    else:
        print("\n--- Using Loaded Model ---")
        if music_model_instance is None: 
            print("Error: Expected a loaded model but it's None. Exiting.")
            exit()

    print("\n--- Generating Final Music ---")
    
    # Define temperature and number of songs
    final_temperature = 0.5
    num_final_songs = 4
    
    # Prepare seed for final generation using the model's vocabulary (pitchnames_for_model, event_to_int_for_model)
    final_seed_sequences_list = []
    for i in range(len(notes_corpus) - SEQUENCE_LENGTH):
        seq_events = notes_corpus[i:i+SEQUENCE_LENGTH]
        try:
            # Map using the event_to_int_for_model, which corresponds to the active model (new or loaded)
            mapped_seq = [event_to_int_for_model[s] for s in seq_events]
            final_seed_sequences_list.append(mapped_seq)
        except KeyError:
            # This sequence contains an event not in the model's vocabulary, skip it for seeding.
            pass 
            
    final_seed_sequences_np = None
    if final_seed_sequences_list:
        final_seed_sequences_np = np.array(final_seed_sequences_list)
    
    if final_seed_sequences_np is None or len(final_seed_sequences_np) == 0:
        print("Warning: Could not create diverse seed sequences for final generation using model's vocabulary.")
        if pitchnames_for_model and n_vocab_for_model > 0:
            print("Attempting a single random seed for final generation as fallback.")
            default_seed_int = np.random.randint(0, n_vocab_for_model)
            pattern = [default_seed_int] * SEQUENCE_LENGTH
            final_seed_sequences_np = np.array([pattern])
        else:
            print("Error: Cannot create even a fallback random seed. No pitchnames/vocab for model.")

    if music_model_instance and final_seed_sequences_np is not None and len(final_seed_sequences_np) > 0:
        # Generate music multiple times with the constant temperature
        for i in range(num_final_songs):
            print(f"\nGenerating final music sample {i+1}/{num_final_songs} with temperature {final_temperature}...")
            
            generated_sequence = music_model_instance.generate_music(
                keras_model_to_predict_with=music_model_instance.model, 
                pitchnames=pitchnames_for_model,       
                event_to_int=event_to_int_for_model,   
                num_events_to_generate=NUM_NOTES_TO_GENERATE,
                network_input_for_start=final_seed_sequences_np,
                temperature=final_temperature
            )

            if generated_sequence:
                output_midi_file = os.path.join(OUTPUT_FOLDER, f"final_generated_s{SEQUENCE_LENGTH}_e{EPOCHS}_temp{final_temperature}_sample{i+1}.mid")
                midi_written_successfully = writer.create_midi(generated_sequence, output_midi_file)
                if midi_written_successfully:
                    print(f"Final music generated (sample {i+1}) with temperature {final_temperature}: {output_midi_file}")
                else:
                    print(f"Final music generation (sample {i+1}) with temperature {final_temperature} failed - MIDI file was not created or was empty.")
            else:
                print(f"Final music generation (sample {i+1}) with temperature {final_temperature} failed - no sequence produced.")
                
    elif not music_model_instance:
        print("Music model instance is not available for final generation. Skipping.")
    else: # final_seed_sequences_np is None or empty
        print("No valid seed sequences available for final music generation, and fallback also failed. Skipping generation.")

    print("\n--- Process Finished ---")