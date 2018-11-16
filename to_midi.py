import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import pprint
'''
mid = MidiFile('/home/johannes/github/autoencoder/sorted_violin_midis/C_major/bach_ave-maria.mid') 
notes = []
velocities = []

def midi_to_int():
    for msg in mid:
        if not msg.is_meta:
            if msg.time == 0:
                continue
            print(msg)
            if msg.channel == 0:
                if msg.type == 'note_on':
                    data = msg.bytes()
                    # append note and velocity from [type, note, velocity]
                    notes.append(data[1])
                    velocities.append(data[2])      
def pred_to_midi(prediction):
    mid = MidiFile()
    track = MidiTrack()
    t = 0
    for note in prediction:
        # 147 means note_on
        note = np.asarray([147, note[0], note[1]])
        bytes = note.astype(int)
        msg = Message.from_bytes(bytes[0:3])
        t += 100
        msg.time = t
        track.append(msg)

    mid.tracks.append(track)
    mid.save('INPUT.mid')
'''
from midiutil import MIDIFile

degrees  = [10,2,3,5,10,2,10,0,10,0,3,0,10,2,0,10]  # MIDI note number
track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = 60   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, time, tempo)

for i, pitch in enumerate(degrees):
    MyMIDI.addNote(track, channel, pitch, time + i, duration, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
'''
new_notes = [10, 2, 3, 5, 10, 2, 10, 0, 10, 0, 3, 0, 10, 2, 0, 10, 2, 2, 0, 10, 5, 2, 0, 10, 5, 3, 2, 3, 0, 10, 10, 2]
velocities = [90]*len(new_notes)

combine = [[i,j] for i,j in zip(new_notes, velocities)]
pred_to_midi(combine)

note_min = np.min(notes)
note_max = np.max(notes)

for line in data:
    for i in line:
        #minmax scaling (0-1)
        i = (i- note_min)/(note_max-note_min)
        #Reverse minmax scaling
        i = int(i*(note_max - note_min) + note_min)
'''

'''
def minmax_norm(in_data):
    in_data = np.array(in_data, dtype=float)
    note_min = np.min(in_data)
    note_max = np.max(in_data)

    for line in in_data:
        for i in range(len(line)):
            #minmax scaling (0-1)
            line[i] = (line[i]- note_min)/(note_max-note_min)
    return in_data, note_min, note_max


def minmax_reverse(in_data, note_min, note_max):
    in_data = np.array(in_data, dtype=float)

    for line in in_data: 
        for i in range(len(line)):
            #Reverse minmax scaling
            line[i] = int(line[i]*(note_max - note_min) + note_min) 
    return in_data
'''