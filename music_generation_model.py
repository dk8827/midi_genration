import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Embedding, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

class MusicGenerationModel:
    def __init__(self, sequence_length, n_vocab):
        self.sequence_length = sequence_length
        self.n_vocab = n_vocab
        self.model = self._create_network()

    def _create_network(self):
        model = Sequential()
        model.add(Embedding(self.n_vocab, 256, input_shape=(self.sequence_length,)))
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
        model.add(Dense(self.n_vocab, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, network_input, network_output, model_file, epochs, batch_size, custom_callbacks=None):
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

        history = self.model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
        print(f"Training complete. Model saved to {model_file}")
        return history

    def generate_music(self, keras_model_to_predict_with, pitchnames, event_to_int, num_events_to_generate, network_input_for_start, temperature=0.5):
        int_to_event = dict((number, event) for number, event in enumerate(pitchnames))

        if network_input_for_start is None or len(network_input_for_start) == 0:
            print("Error: No seed patterns available for generation.")
            if not pitchnames or self.n_vocab == 0:
                print("No pitchnames/vocab available for default seed. Cannot generate.")
                return []
            print("Generating a default random seed (less ideal)...")
            current_n_vocab_for_seed = len(pitchnames)
            if current_n_vocab_for_seed == 0:
                 print("Cannot generate seed, pitchnames list is empty for random choice.")
                 return []
            default_seed_event_int = np.random.randint(0, current_n_vocab_for_seed)
            # The pattern needs to be mapped to indices that are valid for the model.
            # If default_seed_event_int is an event string, it needs mapping via event_to_int.
            # However, the original logic used np.random.randint(0, self.n_vocab) for default seed int.
            # Let's stick to int seed for pattern if pitchnames exist for int_to_event mapping.
            # The `pattern` should be a list of integers (indices).
            # The `default_seed_event_int` chosen from `current_n_vocab_for_seed` (which is `len(pitchnames)`)
            # implies it's an index into `pitchnames`. This is fine.
            pattern = [default_seed_event_int] * self.sequence_length
        else:
            start_index = np.random.randint(0, len(network_input_for_start))
            pattern = network_input_for_start[start_index].tolist()

        prediction_output = []
        print(f"\n--- Generating Music (Temperature: {temperature}, Seed: {pattern[:5]}...)---")

        for event_index in range(num_events_to_generate):
            prediction_input_np = np.reshape(pattern, (1, len(pattern)))
            prediction_probs = keras_model_to_predict_with.predict(prediction_input_np, verbose=0)[0]

            prediction_probs = np.asarray(prediction_probs).astype('float64')
            
            prediction_probs_safe = np.log(prediction_probs + 1e-8) / temperature
            exp_preds = np.exp(prediction_probs_safe - np.max(prediction_probs_safe))

            if np.isinf(exp_preds).any() or np.isnan(exp_preds).any() or np.sum(exp_preds) < 1e-8: # Check sum for near zero
                index = np.argmax(keras_model_to_predict_with.predict(prediction_input_np, verbose=0)[0])
            else:
                prediction_probs_norm = exp_preds / np.sum(exp_preds)
                # Ensure probabilities sum to 1 after normalization due to potential floating point inaccuracies
                if abs(np.sum(prediction_probs_norm) - 1.0) > 1e-6 : # Add tolerance for sum check
                    prediction_probs_norm = prediction_probs_norm / np.sum(prediction_probs_norm)
                
                try:
                    # Ensure n_vocab for choice matches the length of probability distribution
                    index = np.random.choice(len(prediction_probs_norm), p=prediction_probs_norm)
                except ValueError as e: 
                    print(f"Warning: np.random.choice failed. Sum(p): {np.sum(prediction_probs_norm)}. Error: {e}. Using argmax.")
                    index = np.argmax(keras_model_to_predict_with.predict(prediction_input_np, verbose=0)[0])
            
            if index >= len(int_to_event):
                print(f"Warning: Predicted index {index} is out of bounds for int_to_event map (size {len(int_to_event)}). Using last valid index.")
                index = len(int_to_event) - 1
                if index < 0: # Should not happen if int_to_event is not empty
                    print("Error: int_to_event map is empty. Cannot generate event.")
                    break # Stop generation for this item
            
            result = int_to_event[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        
        print(f"Finished generating {num_events_to_generate} events.")
        return prediction_output

    def load_model(self, model_file):
        self.model = load_model(model_file)
        self.n_vocab = self.model.layers[-1].output_shape[-1] 
        print(f"Model loaded from {model_file}. Model vocabulary size: {self.n_vocab}") 