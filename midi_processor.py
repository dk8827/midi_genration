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
        if not name_str or not isinstance(name_str, str):
            return instrument.Piano()
            
        # Direct mapping first
        if name_str in self.instrument_map:
            return self.instrument_map[name_str].__class__()

        # Try music21's built-in instrument mapping
        try:
            cleaned_name = re.sub(r'\d+', '', name_str).strip()
            instr_obj = instrument.fromString(cleaned_name)
            if instr_obj and instr_obj.__class__ != instrument.Instrument:
                return instr_obj
        except:
            pass

        # Enhanced keyword-based mapping with more comprehensive coverage
        name_lower = name_str.lower().strip()
        
        # Remove common prefixes/suffixes that might interfere
        name_lower = re.sub(r'(^(acoustic|electric|digital|synthetic?|midi)\s*)', '', name_lower)
        name_lower = re.sub(r'(\s*(1|2|3|4|5|6|7|8|9|0)+$)', '', name_lower)
        
        # Piano family
        if any(term in name_lower for term in ["piano", "pno", "pf"]):
            if any(term in name_lower for term in ["electric", "elec", "ep", "rhodes", "wurly"]):
                return instrument.ElectricPiano()
            return instrument.Piano()
            
        # Guitar family
        if any(term in name_lower for term in ["guitar", "gtr", "gt"]):
            if any(term in name_lower for term in ["electric", "elec", "jazz", "clean", "dist", "overdrive"]):
                return instrument.ElectricGuitar()
            elif any(term in name_lower for term in ["acoustic", "nylon", "steel", "classical"]):
                return instrument.AcousticGuitar()
            return instrument.AcousticGuitar()  # Default to acoustic
            
        # Bass family
        if "bass" in name_lower and "drum" not in name_lower:
            if any(term in name_lower for term in ["electric", "elec", "finger", "pick", "fretless"]):
                return instrument.ElectricBass()
            elif any(term in name_lower for term in ["acoustic", "double", "upright", "contrabass"]):
                return instrument.Contrabass()
            return instrument.ElectricBass()
            
        # String instruments
        if "violin" in name_lower: return instrument.Violin()
        if "viola" in name_lower: return instrument.Viola()
        if any(term in name_lower for term in ["cello", "violoncello"]): return instrument.Violoncello()
        if any(term in name_lower for term in ["contrabass", "double bass", "upright bass"]): return instrument.Contrabass()
        
        # Wind instruments - Woodwinds
        if "flute" in name_lower: return instrument.Flute()
        if "piccolo" in name_lower: return instrument.Piccolo()
        if "recorder" in name_lower: return instrument.Recorder()
        if "clarinet" in name_lower: return instrument.Clarinet()
        if "oboe" in name_lower: return instrument.Oboe()
        if "bassoon" in name_lower: return instrument.Bassoon()
        if "english horn" in name_lower: return instrument.EnglishHorn()
        
        # Saxophone family
        if "sax" in name_lower:
            if "soprano" in name_lower: return instrument.SopranoSaxophone()
            elif "alto" in name_lower: return instrument.AltoSaxophone()
            elif "tenor" in name_lower: return instrument.TenorSaxophone()
            elif "baritone" in name_lower or "bari" in name_lower: return instrument.BaritoneSaxophone()
            return instrument.Saxophone()
            
        # Brass instruments
        if "trumpet" in name_lower: return instrument.Trumpet()
        if "trombone" in name_lower: return instrument.Trombone()
        if "tuba" in name_lower: return instrument.Tuba()
        if any(term in name_lower for term in ["french horn", "horn"]) and "english" not in name_lower:
            return instrument.Horn()
            
        # Percussion - be more inclusive
        percussion_terms = [
            "drum", "percussion", "perc", "cymbal", "tom", "hi-hat", "hihat", "hat",
            "snare", "kick", "bass drum", "bongo", "conga", "timbale", "cowbell",
            "tambourine", "claves", "wood", "agogo", "guiro", "maracas", "triangle",
            "shaker", "bell", "chime", "gong", "crash", "ride", "splash"
        ]
        if any(term in name_lower for term in percussion_terms):
            return instrument.Percussion()
            
        # Keyboard/Synth family
        if any(term in name_lower for term in ["harpsichord", "harp"]): return instrument.Harpsichord()
        if "celesta" in name_lower: return instrument.Celesta()
        if "glockenspiel" in name_lower: return instrument.Glockenspiel()
        if "vibraphone" in name_lower or "vibes" in name_lower: return instrument.Vibraphone()
        if "marimba" in name_lower: return instrument.Marimba()
        if "xylophone" in name_lower: return instrument.Xylophone()
        if any(term in name_lower for term in ["tubular bells", "chimes"]): return instrument.TubularBells()
        if "dulcimer" in name_lower: return instrument.Dulcimer()
        if "organ" in name_lower: return instrument.Organ()
        
        # Voice/Choir
        if any(term in name_lower for term in ["voice", "vocal", "choir", "chorus", "singer", "aah", "ooh"]):
            return instrument.Choir()
            
        # Synth instruments
        if any(term in name_lower for term in ["synth", "synthesizer", "pad", "lead", "fx", "effect"]):
            return instrument.ElectricPiano()  # Using ElectricPiano as synth substitute
            
        # Check if it's a common instrument class name
        common_instruments = {
            'piano': instrument.Piano(),
            'guitar': instrument.AcousticGuitar(),
            'violin': instrument.Violin(),
            'flute': instrument.Flute(),
            'trumpet': instrument.Trumpet(),
            'percussion': instrument.Percussion(),
            'drums': instrument.Percussion(),
            'organ': instrument.Organ(),
            'harp': instrument.Harp(),
            'banjo': instrument.Banjo(),
            'mandolin': instrument.Mandolin(),
        }
        
        for key, instr_obj in common_instruments.items():
            if key in name_lower:
                return instr_obj
        
        # If we still can't map it, only show warning for truly unusual names
        # Skip warning for generic terms or empty strings
        skip_warning_terms = ["instrument", "track", "channel", "midi", "", " "]
        should_warn = not any(term in name_lower for term in skip_warning_terms) and len(name_str.strip()) > 2
        
        if should_warn:
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

        # Fix the regex pattern for removing trailing numbers
        temp_name = re.sub(r'\s*\d+\s*$', '', original_name_str).strip()
        temp_name_lower = temp_name.lower()

        # Check direct mapping first
        for k_map in self.instrument_map.keys():
            if k_map.lower() == temp_name_lower:
                return k_map 

        name_l = original_name_str.lower() 

        # Enhanced instrument name detection with better patterns
        # Keyboard family
        if "harpsichord" in name_l: return "Harpsichord"
        if "celesta" in name_l: return "Celesta"
        if "glockenspiel" in name_l: return "Glockenspiel"
        if "vibraphone" in name_l or "vibes" in name_l: return "Vibraphone"
        if "marimba" in name_l: return "Marimba"
        if "xylophone" in name_l: return "Xylophone"
        if "tubular bells" in name_l or "chimes" in name_l: return "Tubular Bells"
        if "dulcimer" in name_l: return "Dulcimer"
        if "clavinet" in name_l: return "Clavinet"
        if any(term in name_l for term in ["electric grand piano", "electric piano", "rhodes", "wurly", "ep"]): 
            return "Electric Piano"
        if "piano" in name_l or "pno" in name_l: return "Piano"
        
        # Guitar family
        if any(term in name_l for term in ["electric guitar", "jazz gtr", "clean gtr", "mute gtr", "dist gtr", "overdrive"]):
            return "Electric Guitar"
        if any(term in name_l for term in ["acoustic guitar", "nylon", "steel", "classical guitar"]):
            return "Acoustic Guitar"
        if "guitar" in name_l or "gtr" in name_l: return "Guitar"
        
        # String family
        if "violin" in name_l: return "Violin"
        if "viola" in name_l: return "Viola"
        if "cello" in name_l or "violoncello" in name_l: return "Cello"
        
        # Bass family
        if any(term in name_l for term in ["electric bass", "finger", "pick", "fretless"]) and "bass" in name_l:
            return "Electric Bass"
        if any(term in name_l for term in ["acoustic bass", "contrabass", "double bass", "upright"]):
            return "Contrabass"
        if "bass" in name_l and "drum" not in name_l: return "Bass"
        
        # Wind instruments
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
        
        # Saxophone family
        if "soprano sax" in name_l: return "Soprano Sax"
        if "alto sax" in name_l: return "Alto Sax"
        if "tenor sax" in name_l: return "Tenor Sax"
        if "baritone sax" in name_l or "bari sax" in name_l: return "Baritone Sax"
        if "sax" in name_l: return "Saxophone"
        
        # Brass instruments
        if "trumpet" in name_l: return "Trumpet"
        if "trombone" in name_l: return "Trombone"
        if "tuba" in name_l: return "Tuba"
        if "french horn" in name_l or ("horn" in name_l and "english" not in name_l): return "French Horn"
        if "brass section" in name_l or "brass ensemble" in name_l: return "BrassSection"
        
        # Ensemble instruments
        if "string ensemble" in name_l or ("strings" in name_l and "synth" not in name_l): return "StringEnsemble"
        if any(term in name_l for term in ["voice", "choir", "vocal", "aah", "ooh"]) and "synth" not in name_l: 
            return "Voice"
        if "synth voice" in name_l or "synth choir" in name_l: return "Synth Voice"
        
        # Synth categories
        if any(term in name_l for term in ["synth lead", "synth pad", "synth brass", "synth strings", "polysynth", "fx"]):
            return "Synth"
        if "synth" in name_l: return "Synth"
        
        # Check percussion instruments from instrument map
        for k_map_perc in self.instrument_map.keys():
            is_percussion_map_key = any(p_term in k_map_perc.lower() for p_term in [
                "drum", "cymbal", "tom", "hi-hat", "snare", "kick", "percussion", 
                "bongo", "conga", "timbale", "cowbell", "tambourine", "claves", 
                "wood", "agogo", "guiro", "maracas", "triangle"
            ])
            if is_percussion_map_key and k_map_perc.lower() in name_l:
                return k_map_perc

        # General percussion detection
        if any(p_term in name_l for p_term in [
            "drum", "percussion", "cymbal", "tom", "hat", "snare", "kick", 
            "conga", "bongo", "timbale", "agogo", "woodblock", "claves", 
            "guiro", "maracas", "triangle", "bell", "chime", "gong"
        ]):
            return "Drums"

        # Clean up the final name with proper regex patterns
        final_name = re.sub(r'\d+', '', original_name_str).strip()
        final_name = re.sub(r'\s+', ' ', final_name) 
        final_name = final_name.replace("Instrument", "").strip()
        final_name = ''.join(char for char in final_name if char.isalnum() or char.isspace() or char in ['-', '_', '(', ')'])
        final_name = final_name.strip()

        # If we still don't have a good name, try the class name
        if not final_name or final_name.lower() == "instrument" or len(final_name) < 2:
            class_name = m21_instrument.__class__.__name__
            if class_name and class_name not in ["Instrument", "UnpitchedPercussion", "PitchedPercussion"]:
                # Check if this class name maps to any instrument in our map
                for k_map, v_map_obj in self.instrument_map.items():
                    if v_map_obj.__class__.__name__ == class_name:
                        return k_map
                # Return the class name if it seems reasonable
                if len(class_name) > 2:
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