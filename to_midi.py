import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np

mid = MidiFile('/home/johannes/github/in5490_AI/sorted_violin_midis/A_major/trio_IV.mid') 
notes = []
velocities = []

def midi_to_int():
    for msg in mid:
        if not msg.is_meta:
            print(msg)
            if msg.time == 0:
                continue
            if msg.channel == 0:
                if msg.type == 'note_on':
                    data = msg.bytes()
                    # append note and velocity from [type, note, velocity]
                    notes.append(data[1])
                    velocities.append(data[2])      
    combine = [[i,j] for i,j in zip(notes, velocities)]
    int_to_midi(combine)

def pred_to_midi(prediction):
    mid = MidiFile()
    track = MidiTrack()
    t = 0
    for note in prediction:
        # 147 means note_on
        note = np.asarray([147, note[0], note[1]])
        bytes = note.astype(int)
        print(note)
        msg = Message.from_bytes(bytes[0:3])
        t += 1
        msg.time = t
        track.append(msg)

    mid.tracks.append(track)
    mid.save('JK.mid')

midi_to_int()

new_notes = []
for i in range(127):   
    new_notes.append(i)
combine = [[i,j] for i,j in zip(new_notes, velocities[:127])]
pred_to_midi(combine)
