import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from music21 import converter, instrument, note, chord, stream
import os

def get_instrument_name_for_plot(m21_instrument):
    """Gets a simplified name for plotting, similar to program.py but less complex."""
    if not m21_instrument:
        return "Unknown Instrument"
    if hasattr(m21_instrument, 'instrumentName') and m21_instrument.instrumentName:
        return str(m21_instrument.instrumentName)
    if hasattr(m21_instrument, 'bestName') and callable(m21_instrument.bestName):
        name = m21_instrument.bestName()
        if name:
            return str(name)
    return m21_instrument.__class__.__name__

def visualize_midi_file(filepath):
    """
    Parses a MIDI file and visualizes its notes over time (piano roll style).
    """
    if not filepath:
        print("No file selected.")
        return

    print(f"Loading MIDI file: {filepath}")
    try:
        midi_score = converter.parse(filepath)
    except Exception as e:
        print(f"Error parsing MIDI file: {e}")
        tk.messagebox.showerror("MIDI Parse Error", f"Could not parse the MIDI file: {e}")
        return

    fig, ax = plt.subplots(figsize=(20, 10))
    
    y_ticks_labels = {} # To store pitch names for y-axis
    
    # Get a list of distinct colors for different parts/instruments
    # Exclude very light colors that might not be visible on a white background
    all_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    distinct_colors = [c for i, c in enumerate(all_colors) if i % 3 == 0 and not mcolors.to_rgb(c)[0] > 0.9 and not mcolors.to_rgb(c)[1] > 0.9 and not mcolors.to_rgb(c)[2] > 0.9 ] 
    color_idx = 0

    parts_data = [] # To store (instrument_name, notes_data)

    # Try to partition by instrument first
    instrument_parts = instrument.partitionByInstrument(midi_score)
    if instrument_parts:
        print(f"Found {len(instrument_parts)} instrument parts.")
        parts_to_iterate = instrument_parts.parts
    else:
        print("No instrument parts found by partitionByInstrument. Using flat score.")
        # If no instrument parts, treat the whole score as one or iterate flat parts
        if midi_score.parts:
             parts_to_iterate = midi_score.parts
        else: # If no parts at all, use the flat stream
            new_part = stream.Part(id="Default Part")
            new_part.append(midi_score.flat.notesAndRests)
            instr = midi_score.flat.getElementsByClass(instrument.Instrument).first()
            if instr:
                new_part.insert(0, instr)
            else:
                new_part.insert(0, instrument.Piano()) # Default to Piano
            parts_to_iterate = [new_part]


    for part_idx, part in enumerate(parts_to_iterate):
        instr = part.getInstrument()
        if not instr and len(part.getElementsByClass(instrument.Instrument)) > 0 :
            instr = part.getElementsByClass(instrument.Instrument)[0]
        
        instr_name = get_instrument_name_for_plot(instr)
        if part.id:
             instr_name = f"{part.id} ({instr_name})"
        else:
             instr_name = f"Part {part_idx} ({instr_name})"

        print(f"  Processing: {instr_name}")
        
        current_color = distinct_colors[color_idx % len(distinct_colors)]
        color_idx += 1
        
        notes_in_part = []

        for element in part.recurse().notesAndRests: # Recurse to get all notes in nested streams
            offset = element.getOffsetInHierarchy(midi_score) # Get absolute offset
            duration = element.duration.quarterLength

            if isinstance(element, note.Note):
                pitch_val = element.pitch.midi # MIDI note number
                notes_in_part.append({'offset': offset, 'pitch': pitch_val, 'duration': duration, 'label': str(element.pitch)})
                if pitch_val not in y_ticks_labels or len(str(element.pitch)) < len(y_ticks_labels[pitch_val]):
                     y_ticks_labels[pitch_val] = str(element.pitch) # Store pitch name like C#4
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    pitch_val = p.midi
                    notes_in_part.append({'offset': offset, 'pitch': pitch_val, 'duration': duration, 'label': str(p)})
                    if pitch_val not in y_ticks_labels or len(str(p)) < len(y_ticks_labels[pitch_val]):
                        y_ticks_labels[pitch_val] = str(p)
            # Rests are not explicitly plotted as bars, but they create the space

        if notes_in_part:
            parts_data.append({'name': instr_name, 'color': current_color, 'notes': notes_in_part})

    if not parts_data:
        print("No notes found to plot.")
        tk.messagebox.showinfo("Info", "No notes found in the MIDI file to visualize.")
        plt.close(fig)
        return
        
    # Plotting
    legend_handles = []
    min_pitch = float('inf')
    max_pitch = float('-inf')
    max_offset = 0

    for part_info in parts_data:
        for n_info in part_info['notes']:
            ax.barh(n_info['pitch'], n_info['duration'], left=n_info['offset'], height=0.8, 
                    color=part_info['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
            min_pitch = min(min_pitch, n_info['pitch'])
            max_pitch = max(max_pitch, n_info['pitch'])
            max_offset = max(max_offset, n_info['offset'] + n_info['duration'])
        
        # Create a proxy artist for the legend
        legend_handles.append(plt.Rectangle((0,0),1,1, color=part_info['color'], label=part_info['name']))

    if min_pitch == float('inf'): # No notes were actually plotted
        print("No valid note data to plot boundaries for.")
        tk.messagebox.showinfo("Info", "No notes could be plotted from the MIDI file.")
        plt.close(fig)
        return

    ax.set_xlabel("Time (Quarter Lengths)")
    ax.set_ylabel("Pitch (MIDI Note Number)")
    ax.set_title(f"MIDI Visualization: {os.path.basename(filepath)}")
    
    # Set Y-axis ticks and labels
    sorted_pitches = sorted(y_ticks_labels.keys())
    if sorted_pitches:
        ax.set_yticks(sorted_pitches)
        ax.set_yticklabels([y_ticks_labels[p] for p in sorted_pitches])
        ax.set_ylim(min_pitch - 2, max_pitch + 2) # Add some padding
    
    ax.set_xlim(0, max_offset + 1) # Add padding to x-axis
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend outside the plot
    ax.legend(handles=legend_handles, title="Instruments/Parts", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.subplots_adjust(right=0.75) # Adjust layout to make space for legend

    plt.show()

def open_file_dialog():
    """
    Opens a file dialog to select a MIDI file and then visualizes it.
    """
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    file_path = filedialog.askopenfilename(
        title="Select a MIDI file",
        filetypes=(("MIDI files", "*.mid *.midi"), ("All files", "*.*"))
    )
    root.destroy() # Close the hidden root window after dialog

    if file_path:
        visualize_midi_file(file_path)
    else:
        print("No file selected by the user.")

if __name__ == "__main__":
    # Check if running in a headless environment (e.g., if DISPLAY is not set)
    try:
        # Attempt to create a Tk root window to check for display
        root_check = tk.Tk()
        root_check.withdraw()
        root_check.destroy()
        can_show_gui = True
    except tk.TclError:
        print("No display environment available (e.g., running in a headless server or no X server).")
        print("This script requires a GUI to select a file and display the plot.")
        print("If you have a MIDI file path, you could modify the script to call visualize_midi_file(filepath) directly.")
        can_show_gui = False

    if can_show_gui:
        open_file_dialog()
    else:
        # Example: If you want to allow running with a hardcoded path for testing in non-GUI envs
        # test_midi_path = "path/to/your/test.mid" 
        # if os.path.exists(test_midi_path):
        #    visualize_midi_file(test_midi_path)
        # else:
        #    print(f"Test MIDI file not found: {test_midi_path}")
        pass 