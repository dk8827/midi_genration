import os
import numpy as np
import keras
# from music21 import converter # No longer needed here
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
        self.temperatures = [0.5]  # Set temperature to constant 0.5
        self.num_songs_to_generate_per_epoch = 4 # Number of songs to generate per epoch
        os.makedirs(self.output_folder, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} finished. Generating {self.num_songs_to_generate_per_epoch} sample MIDI files with temperature 0.5...")

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
        
        # Generate music for the specified temperature, multiple times
        for i in range(self.num_songs_to_generate_per_epoch):
            temp = self.temperatures[0] # Use the constant temperature
            print(f"\nGenerating music sample {i+1}/{self.num_songs_to_generate_per_epoch} with temperature {temp}...")
            
            generated_sequence = self.music_generator_instance.generate_music(
                keras_model_to_predict_with=self.model, 
                pitchnames=self.pitchnames,      
                event_to_int=self.event_to_int,   
                num_events_to_generate=self.num_notes_to_generate,
                network_input_for_start=current_seed_sequences,
                temperature=temp
            )

            print(f"Callback: Raw generated sequence for epoch {epoch+1}, temp {temp}, sample {i+1}: {generated_sequence[:10]}..." if len(generated_sequence) > 10 else f"Callback: Raw generated sequence for epoch {epoch+1}, temp {temp}, sample {i+1}: {generated_sequence}")

            if generated_sequence:
                output_midi_file = os.path.join(self.output_folder, f"{self.base_filename}_{epoch+1}_temp{temp}_sample{i+1}.mid")
                # MidiFileWriter.create_midi now returns True if a meaningful MIDI was written, False otherwise.
                midi_written_successfully = self.midi_file_writer.create_midi(generated_sequence, output_midi_file)

                if midi_written_successfully:
                    # The MidiFileWriter already prints a success message if it writes the file.
                    print(f"Callback: Sample MIDI for epoch {epoch+1}, temp {temp}, sample {i+1} was generated: {output_midi_file}")
                else:
                    # The MidiFileWriter prints detailed reasons if it fails to write or deems the content empty.
                    print(f"Callback: INFO - MIDI file for epoch {epoch+1}, temp {temp}, sample {i+1} ('{output_midi_file}') was not generated or was deemed empty by the MIDI writer. See writer logs for details.")
            else:
                print(f"Callback: MIDI generation failed for epoch {epoch+1}, temp {temp}, sample {i+1}, no sequence produced.") 