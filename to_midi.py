import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import pprint
mid = MidiFile('/home/johannes/github/in5490_AI/sorted_violin_midis/A_major/trio_IV.mid') 
notes = []
velocities = []

def midi_to_int():
    for msg in mid:
        if not msg.is_meta:
            if msg.time == 0:
                continue
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
        t += 1
        msg.time = t
        track.append(msg)

    mid.tracks.append(track)
    mid.save('JK.mid')

midi_to_int()

new_notes = []
for i in range(127):   
    new_notes.append(i)
combine = [[i,j] for i,j in zip(notes, velocities)]
#pred_to_midi(combine)


'''
note_min = np.min(notes)
note_max = np.max(notes)

for line in data:
    for i in line:
        #minmax scaling (0-1)
        i = (i- note_min)/(note_max-note_min)
        #Reverse minmax scaling
        i = int(i*(note_max - note_min) + note_min)
'''