import os
from music21 import stream, note, chord, tempo, meter
from config import DEFAULT_INSTRUMENT_NAME
# MidiProcessor is needed if we pass it for get_instrument_from_name
# from midi_processor import MidiProcessor # Assuming midi_processor.py will exist

class MidiFileWriter:
    def __init__(self, midi_processor, default_instrument_name=DEFAULT_INSTRUMENT_NAME):
        self.midi_processor = midi_processor 
        self.default_instrument_name = default_instrument_name

    def create_midi(self, prediction_output, output_file="output_instrumental.mid"):
        score = stream.Score()
        ts = meter.TimeSignature('4/4')
        score.insert(0, ts)
        tm = tempo.MetronomeMark(number=120)
        score.insert(0, tm)

        parts = {} # Maps instrument_name_str to stream.Part
        part_offsets = {} # Maps instrument_name_str to its current offset in quarter lengths
        
        # Track if any actual musical events (notes/chords) are processed from the input
        # Rests are important for timing but don't count towards "non-empty" for this check.
        any_notes_or_chords_processed = False

        for i, pattern_item in enumerate(prediction_output):
            try:
                # Handle simultaneous events (new format: SIMUL[Piano:C4,Guitar:E4])
                if pattern_item.startswith("SIMUL[") and pattern_item.endswith("]"):
                    # Extract simultaneous events
                    simul_content = pattern_item[6:-1]  # Remove "SIMUL[" and "]"
                    simultaneous_events = simul_content.split(',')
                    
                    # Find the maximum current offset among all involved instruments
                    max_current_offset = 0.0
                    involved_instruments = []
                    
                    for event in simultaneous_events:
                        if ':' not in event:
                            print(f"INFO (MidiWriter): Skipping malformed simultaneous event (no ':') in '{pattern_item}': {event}")
                            continue
                        instrument_name_str, _ = event.split(':', 1)
                        involved_instruments.append(instrument_name_str)
                        
                        # Ensure part exists
                        if instrument_name_str not in parts:
                            m21_instr_obj = self.midi_processor.get_instrument_from_name(instrument_name_str)
                            new_part = stream.Part(id=instrument_name_str)
                            new_part.insert(0, m21_instr_obj)
                            parts[instrument_name_str] = new_part
                            part_offsets[instrument_name_str] = 0.0
                            score.insert(0, new_part)
                        
                        max_current_offset = max(max_current_offset, part_offsets[instrument_name_str])
                    
                    # Synchronize all involved instruments to the same offset
                    for instrument_name_str in involved_instruments:
                        if part_offsets[instrument_name_str] < max_current_offset:
                            # Add rest to bring this instrument up to the sync point
                            rest_duration = max_current_offset - part_offsets[instrument_name_str]
                            if rest_duration > 0:
                                rest_event = note.Rest()
                                rest_event.duration.quarterLength = rest_duration
                                parts[instrument_name_str].insert(part_offsets[instrument_name_str], rest_event)
                                part_offsets[instrument_name_str] = max_current_offset
                    
                    # Now add the simultaneous events
                    for event in simultaneous_events:
                        if ':' not in event:
                            continue
                        instrument_name_str, event_str = event.split(':', 1)
                        
                        current_part = parts[instrument_name_str]
                        current_offset_for_this_part = part_offsets[instrument_name_str]
                        
                        m21_event = None
                        event_duration_ql = 0.5 

                        if event_str == "Rest":
                            m21_event = note.Rest()
                            m21_event.duration.quarterLength = event_duration_ql
                        elif ('.' in event_str) or event_str.isdigit(): 
                            notes_in_chord_pitches = event_str.split('.')
                            chord_notes_obj = []
                            for p_str in notes_in_chord_pitches:
                                try:
                                    n_obj = note.Note(int(p_str)) 
                                    chord_notes_obj.append(n_obj)
                                except ValueError:
                                    continue
                                except Exception as pitch_e:
                                    continue
                            if chord_notes_obj:
                                m21_event = chord.Chord(chord_notes_obj)
                                m21_event.duration.quarterLength = event_duration_ql
                                any_notes_or_chords_processed = True
                        else: 
                            try:
                                m21_event = note.Note(event_str)
                                m21_event.duration.quarterLength = event_duration_ql
                                any_notes_or_chords_processed = True
                            except Exception:
                                continue 
                        
                        if m21_event:
                            current_part.insert(current_offset_for_this_part, m21_event)
                            part_offsets[instrument_name_str] += m21_event.duration.quarterLength
                
                # Handle regular single instrument events
                elif ':' in pattern_item:
                    instrument_name_str, event_str = pattern_item.split(':', 1)

                    if instrument_name_str not in parts:
                        m21_instr_obj = self.midi_processor.get_instrument_from_name(instrument_name_str)
                        new_part = stream.Part(id=instrument_name_str)
                        new_part.insert(0, m21_instr_obj)
                        parts[instrument_name_str] = new_part
                        part_offsets[instrument_name_str] = 0.0
                        score.insert(0, new_part)

                    current_part = parts[instrument_name_str]
                    current_offset_for_this_part = part_offsets[instrument_name_str]
                    
                    m21_event = None
                    event_duration_ql = 0.5 

                    if event_str == "Rest":
                        m21_event = note.Rest()
                        m21_event.duration.quarterLength = event_duration_ql
                    elif ('.' in event_str) or event_str.isdigit(): 
                        notes_in_chord_pitches = event_str.split('.')
                        chord_notes_obj = []
                        for p_str in notes_in_chord_pitches:
                            try:
                                n_obj = note.Note(int(p_str)) 
                                chord_notes_obj.append(n_obj)
                            except ValueError:
                                continue
                            except Exception as pitch_e:
                                continue
                        if chord_notes_obj:
                            m21_event = chord.Chord(chord_notes_obj)
                            m21_event.duration.quarterLength = event_duration_ql
                            any_notes_or_chords_processed = True
                    else: 
                        try:
                            m21_event = note.Note(event_str)
                            m21_event.duration.quarterLength = event_duration_ql
                            any_notes_or_chords_processed = True
                        except Exception:
                            continue 
                    
                    actual_duration_for_offset_update = 0.0
                    if m21_event:
                        current_part.insert(current_offset_for_this_part, m21_event)
                        actual_duration_for_offset_update = m21_event.duration.quarterLength
                    
                    part_offsets[instrument_name_str] += actual_duration_for_offset_update
                else:
                    print(f"INFO (MidiWriter): Skipping malformed item (no ':' and not SIMUL) in sequence for '{output_file}': {pattern_item}")
                    continue

            except Exception as e:
                print(f"ERROR (MidiWriter): Processing generated item '{pattern_item}' for '{output_file}' at index {i}: {e}")
                continue 
            
        # Primary check: if no notes or chords were ever successfully processed from the input sequence.
        if not any_notes_or_chords_processed:
            print(f"INFO (MidiWriter): The input sequence for '{output_file}' did not result in any processable note or chord events. MIDI file will not be written or will be effectively empty.")
            return False

        # Music21 structural checks:
        # Check if score has elements and at least one part has notesAndRests.
        # This is a broader check; `any_notes_or_chords_processed` is more specific to our goal.
        if not score.elements or not any(isinstance(el, stream.Part) and len(el.notesAndRests) > 0 for el in score.elements):
            print(f"INFO (MidiWriter): No valid music elements (notes, chords, or rests in parts) were added to the score for '{output_file}'. MIDI will be empty.")
            return False

        try:
            # Remove parts that ended up with no notes or rests at all.
            # Then check if any parts *with actual notes/chords* (not just rests) remain.
            final_parts_have_notes = False
            for p in list(score.getElementsByClass(stream.Part)): # Iterate copy for safe removal
                if not p.notesAndRests: # Part is completely empty
                    score.remove(p)
                    print(f"INFO (MidiWriter): Removed completely empty part '{p.id if p.id else 'Unnamed Part'}' from score for '{output_file}'.")
                elif p.notes: # Part has notes or chords
                    final_parts_have_notes = True
            
            if not final_parts_have_notes:
                 print(f"INFO (MidiWriter): No parts with actual note or chord events remaining after cleanup for '{output_file}'. MIDI will be effectively empty.")
                 return False
                 
            score.write('midi', fp=output_file)
            print(f"INFO (MidiWriter): Multi-instrument MIDI file with notes/chords saved as {output_file}")
            return True
        except Exception as e:
            print(f"ERROR (MidiWriter): Writing MIDI file '{output_file}': {e}")
            return False 