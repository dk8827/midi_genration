import glob
import re
import numpy as np
from music21 import converter, instrument, note, chord
from keras.utils import to_categorical
from config import DEFAULT_INSTRUMENT_NAME, INSTRUMENT_MAP

class MidiProcessor:
    def __init__(self, data_path, default_instrument_name=DEFAULT_INSTRUMENT_NAME, time_resolution=0.25):
        self.data_path = data_path
        self.default_instrument_name = default_instrument_name
        self.instrument_map = INSTRUMENT_MAP
        self.time_resolution = time_resolution  # Time resolution in quarter notes (0.25 = sixteenth note)

    def get_instrument_from_name(self, name_str):
        """Attempts to get a music21 instrument object from its name."""
        if name_str in self.instrument_map:
            return self.instrument_map[name_str].__class__()

        try:
            cleaned_name = re.sub(r'\\d+', '', name_str).strip()
            instr_obj = instrument.fromString(cleaned_name)
            if instr_obj:
                if instr_obj.__class__ == instrument.Instrument:
                    pass
                else:
                    return instr_obj
        except:
            pass

        name_lower = name_str.lower()
        if "piano" in name_lower: return instrument.Piano()
        if "guitar" in name_lower: return instrument.AcousticGuitar()
        if "violin" in name_lower: return instrument.Violin()
        if "flute" in name_lower: return instrument.Flute()
        if "drum" in name_lower or "percussion" in name_lower: return instrument.Percussion()
        if "sax" in name_lower: return instrument.Saxophone()
        if "trumpet" in name_lower: return instrument.Trumpet()
        if "bass" in name_lower and "drum" not in name_lower : return instrument.ElectricBass()

        print(f"Warning: Could not map instrument name '{name_str}' reliably. Defaulting to Piano.")
        return instrument.Piano()

    def get_instrument_name(self, m21_instrument):
        """Gets a simplified, consistent name for a music21 instrument object."""
        if not m21_instrument:
            return self.default_instrument_name

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
            return self.default_instrument_name

        temp_name = re.sub(r'\\s*\\d+\\s*$', '', original_name_str).strip()
        temp_name_lower = temp_name.lower()

        for k_map in self.instrument_map.keys():
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
        if "guitar" in name_l: return "Guitar" # Generic Guitar if not specified
        if "violin" in name_l: return "Violin"
        if "viola" in name_l: return "Viola"
        if "cello" in name_l or "violoncello" in name_l: return "Cello"
        if ("electric bass" in name_l or "finger" in name_l or "pick" in name_l or "fretless" in name_l) and "bass" in name_l : return "Electric Bass"
        if "acoustic bass" in name_l or "contrabass" in name_l or "double bass" in name_l : return "Contrabass"
        if "bass" in name_l and "drum" not in name_l: return "Bass" # Generic Bass
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
        
        for k_map_perc in self.instrument_map.keys():
            is_percussion_map_key = any(p_term in k_map_perc.lower() for p_term in ["drum", "cymbal", "tom", "hi-hat", "snare", "kick", "percussion", "bongo", "conga", "timbale", "cowbell", "tambourine", "claves", "wood", "agogo", "guiro", "maracas", "triangle"])
            if is_percussion_map_key and k_map_perc.lower() in name_l:
                return k_map_perc

        if any(p_term in name_l for p_term in ["drum", "percussion", "cymbal", "tom", "hat", "snare", "kick", "conga", "bongo", "timbale", "agogo", "woodblock", "claves", "guiro", "maracas", "triangle"]):
            return "Drums"

        final_name = re.sub(r'\\d+', '', original_name_str).strip()
        final_name = re.sub(r'\\s+', ' ', final_name) 
        final_name = final_name.replace("Instrument", "").strip()
        final_name = ''.join(char for char in final_name if char.isalnum() or char.isspace() or char in ['-', '_', '(', ')'])
        final_name = final_name.strip()

        if not final_name or final_name.lower() == "instrument" or len(final_name) < 2:
            class_name = m21_instrument.__class__.__name__
            if class_name and class_name != "Instrument" and class_name != "UnpitchedPercussion" and class_name != "PitchedPercussion":
                for k_map, v_map_obj in self.instrument_map.items():
                    if v_map_obj.__class__.__name__ == class_name:
                        return k_map
                return class_name
            return self.default_instrument_name
            
        return final_name

    def get_time_aware_notes_from_midi_files(self):
        """
        Extract MIDI events with timing information to preserve coordination between instruments.
        Returns events grouped by time slots to maintain simultaneity.
        """
        all_timed_events = []  # List of (time_slot, instrument, event) tuples
        
        for file_path in glob.glob(self.data_path + "*.mid"):
            try:
                midi = converter.parse(file_path)
                print(f"Parsing with timing: {file_path}")
                
                # Collect all events with their absolute timing
                file_events = []
                
                parts = instrument.partitionByInstrument(midi)
                if parts:
                    for part in parts:
                        instr = part.getInstrument()
                        if instr is None and len(part.getElementsByClass(instrument.Instrument)) > 0:
                            instr = part.getElementsByClass(instrument.Instrument)[0]
                        instr_name = self.get_instrument_name(instr)
                        
                        for element in part.recurse().notesAndRests:
                            absolute_offset = element.getOffsetInHierarchy(midi)
                            duration = element.duration.quarterLength
                            
                            event_str = None
                            if isinstance(element, note.Note):
                                event_str = str(element.pitch)
                            elif isinstance(element, chord.Chord):
                                event_str = '.'.join(str(n) for n in element.normalOrder)
                            elif isinstance(element, note.Rest):
                                event_str = "Rest"
                            
                            if event_str:
                                # Quantize time to our resolution grid
                                time_slot = round(absolute_offset / self.time_resolution) * self.time_resolution
                                file_events.append((time_slot, instr_name, event_str, duration))
                else:
                    # Handle flat MIDI files
                    current_instrument_name = self.default_instrument_name
                    instr_at_start = midi.flat.getElementsByClass(instrument.Instrument).first()
                    if instr_at_start:
                        current_instrument_name = self.get_instrument_name(instr_at_start)
                    
                    for element in midi.flat.notesAndRests:
                        absolute_offset = element.getOffsetInHierarchy(midi)
                        duration = element.duration.quarterLength
                        
                        # Check for instrument changes
                        el_instr = element.getInstrument(returnDefault=False)
                        if el_instr:
                            current_instrument_name = self.get_instrument_name(el_instr)
                        
                        event_str = None
                        if isinstance(element, note.Note):
                            event_str = str(element.pitch)
                        elif isinstance(element, chord.Chord):
                            event_str = '.'.join(str(n) for n in element.normalOrder)
                        elif isinstance(element, note.Rest):
                            event_str = "Rest"
                        
                        if event_str:
                            time_slot = round(absolute_offset / self.time_resolution) * self.time_resolution
                            file_events.append((time_slot, current_instrument_name, event_str, duration))
                
                all_timed_events.extend(file_events)
                
                if not file_events:
                    print(f"    No timed events found in {file_path}")
                    
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        if not all_timed_events:
            print("Warning: No timed events were extracted. Check your MIDI files and parsing logic.")
            return []
        
        # Group events by time slots
        time_grouped_events = self._group_events_by_time(all_timed_events)
        
        # Convert to sequence format
        time_aware_sequence = self._create_time_aware_sequence(time_grouped_events)
        
        print(f"Total time-grouped events extracted: {len(time_aware_sequence)}")
        return time_aware_sequence

    def _group_events_by_time(self, timed_events):
        """Group events that occur at the same time slot."""
        from collections import defaultdict
        
        time_groups = defaultdict(dict)  # {time_slot: {instrument: event}}
        
        # Sort events by time
        timed_events.sort(key=lambda x: x[0])
        
        for time_slot, instrument, event, duration in timed_events:
            # If an instrument already has an event at this time slot, 
            # handle the conflict (could be overlapping notes)
            if instrument in time_groups[time_slot]:
                # For now, we'll concatenate overlapping events with '+'
                existing_event = time_groups[time_slot][instrument]
                if existing_event != event:  # Only if it's actually different
                    time_groups[time_slot][instrument] = f"{existing_event}+{event}"
            else:
                time_groups[time_slot][instrument] = event
        
        return dict(time_groups)

    def _create_time_aware_sequence(self, time_grouped_events):
        """Convert time-grouped events into a sequence suitable for the neural network."""
        sequence = []
        
        # Sort time slots
        sorted_times = sorted(time_grouped_events.keys())
        
        for time_slot in sorted_times:
            instrument_events = time_grouped_events[time_slot]
            
            if len(instrument_events) == 1:
                # Single instrument playing
                instrument, event = list(instrument_events.items())[0]
                sequence.append(f"{instrument}:{event}")
            else:
                # Multiple instruments playing simultaneously
                # Create a compound event that represents simultaneity
                simultaneous_parts = []
                for instrument in sorted(instrument_events.keys()):  # Sort for consistency
                    event = instrument_events[instrument]
                    simultaneous_parts.append(f"{instrument}:{event}")
                
                # Encode as simultaneous event
                compound_event = f"SIMUL[{','.join(simultaneous_parts)}]"
                sequence.append(compound_event)
        
        return sequence

    def get_notes_from_midi_files(self):
        """
        Updated method that uses time-aware extraction by default.
        """
        return self.get_time_aware_notes_from_midi_files()

    def analyze_time_aware_encoding(self, time_aware_sequence):
        """
        Analyze the time-aware encoding to provide insights about simultaneity patterns.
        """
        total_events = len(time_aware_sequence)
        simultaneous_events = sum(1 for event in time_aware_sequence if event.startswith("SIMUL["))
        single_events = total_events - simultaneous_events
        
        print(f"\n--- Time-Aware Encoding Analysis ---")
        print(f"Total events: {total_events}")
        print(f"Single instrument events: {single_events} ({single_events/total_events*100:.1f}%)")
        print(f"Simultaneous events: {simultaneous_events} ({simultaneous_events/total_events*100:.1f}%)")
        
        # Analyze instrument participation
        instruments_in_simul = set()
        max_simul_size = 0
        simul_sizes = []
        
        for event in time_aware_sequence:
            if event.startswith("SIMUL["):
                simul_content = event[6:-1]
                parts = simul_content.split(',')
                simul_sizes.append(len(parts))
                max_simul_size = max(max_simul_size, len(parts))
                
                for part in parts:
                    if ':' in part:
                        instrument = part.split(':', 1)[0]
                        instruments_in_simul.add(instrument)
        
        if simul_sizes:
            avg_simul_size = sum(simul_sizes) / len(simul_sizes)
            print(f"Average instruments per simultaneous event: {avg_simul_size:.1f}")
            print(f"Maximum instruments playing simultaneously: {max_simul_size}")
            print(f"Instruments participating in simultaneous events: {sorted(instruments_in_simul)}")
        
        # Sample some simultaneous events
        sample_simul = [event for event in time_aware_sequence if event.startswith("SIMUL[")][:3]
        if sample_simul:
            print(f"\nSample simultaneous events:")
            for i, event in enumerate(sample_simul):
                print(f"  {i+1}: {event}")
        
        print("--- End Analysis ---\n")

    def prepare_sequences(self, notes_with_instruments, n_vocab, sequence_length):
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