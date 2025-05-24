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

        parts = {} 
        current_global_offset = 0.0
        last_event_duration = 0.5 

        for i, pattern_item in enumerate(prediction_output):
            try:
                if ':' not in pattern_item:
                    print(f"Skipping malformed item (no ':'): {pattern_item}")
                    continue

                instrument_name_str, event_str = pattern_item.split(':', 1)

                if instrument_name_str not in parts:
                    m21_instr_obj = self.midi_processor.get_instrument_from_name(instrument_name_str)
                    new_part = stream.Part(id=instrument_name_str)
                    new_part.insert(0, m21_instr_obj)
                    parts[instrument_name_str] = new_part
                    score.insert(0, new_part)

                current_part = parts[instrument_name_str]
                m21_event = None
                duration = 0.5 

                if event_str == "Rest":
                    m21_event = note.Rest()
                    m21_event.duration.quarterLength = duration
                elif ('.' in event_str) or event_str.isdigit(): 
                    notes_in_chord_pitches = event_str.split('.')
                    chord_notes_obj = []
                    for p_str in notes_in_chord_pitches:
                        try:
                            n_obj = note.Note(int(p_str)) 
                            chord_notes_obj.append(n_obj)
                        except Exception:
                            # print(f"Warning: Could not parse pitch '{p_str}' in chord '{event_str}'. Skipping.")
                            continue
                    if chord_notes_obj:
                        m21_event = chord.Chord(chord_notes_obj)
                        m21_event.duration.quarterLength = duration
                else: 
                    try:
                        m21_event = note.Note(event_str)
                        m21_event.duration.quarterLength = duration
                    except Exception:
                        # print(f"Warning: Could not parse note '{event_str}'. Skipping.")
                        continue
                
                if m21_event:
                    current_part.insert(current_global_offset, m21_event)
                    last_event_duration = m21_event.duration.quarterLength
                else:
                    last_event_duration = 0.5 
                
                current_global_offset += last_event_duration

            except Exception as e:
                print(f"Error processing generated item '{pattern_item}' at index {i}: {e}")
                continue
            
        if not score.elements or not any(isinstance(el, stream.Part) and len(el.notesAndRests) > 0 for el in score.elements):
            print("No valid music elements were added to the score. MIDI will be empty.")
            return

        try:
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