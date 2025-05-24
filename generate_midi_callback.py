import os
import numpy as np
import keras
# from music_generation_model import MusicGenerationModel # Will be needed
# from midi_file_writer import MidiFileWriter # Will be needed

class GenerateMidiCallback(keras.callbacks.Callback):
    def __init__(self, output_folder, sequence_length, num_notes_to_generate,
                 pitchnames, event_to_int, n_vocab, seed_input_sequences,
                 music_generator_instance, # Instance of MusicGenerationModel
                 midi_file_writer,         # Instance of MidiFileWriter
                 base_filename="generated_epoch"):
        super(GenerateMidiCallback, self).__init__()
        self.output_folder = output_folder
        self.sequence_length = sequence_length
        self.num_notes_to_generate = num_notes_to_generate
        self.pitchnames = pitchnames
        self.event_to_int = event_to_int
        self.n_vocab = n_vocab 
        self.seed_input_sequences = seed_input_sequences
        self.music_generator_instance = music_generator_instance
        self.midi_file_writer = midi_file_writer
        self.base_filename = base_filename
        os.makedirs(self.output_folder, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} finished. Generating sample MIDI...")

        current_seed_sequences = self.seed_input_sequences
        if self.seed_input_sequences is None or len(self.seed_input_sequences) == 0:
            print("Callback: No seed input sequences available.")
            if self.pitchnames and self.n_vocab > 0:
                print("Callback: Attempting basic seed for MIDI generation...")
                default_seed_event_int = np.random.randint(0, self.n_vocab) 
                pattern = [default_seed_event_int] * self.sequence_length
                current_seed_sequences = np.array([pattern])
            else:
                print("Callback: Cannot generate seed, pitchnames or n_vocab missing/invalid.")
                return
        
        generated_sequence = self.music_generator_instance.generate_music(
            keras_model_to_predict_with=self.model, 
            pitchnames=self.pitchnames,      
            event_to_int=self.event_to_int,   
            num_events_to_generate=self.num_notes_to_generate,
            network_input_for_start=current_seed_sequences
        )

        if generated_sequence:
            output_midi_file = os.path.join(self.output_folder, f"{self.base_filename}_{epoch+1}.mid")
            self.midi_file_writer.create_midi(generated_sequence, output_midi_file)
            print(f"Callback: Sample MIDI saved to {output_midi_file}")
        else:
            print("Callback: MIDI generation failed, no sequence produced.") 