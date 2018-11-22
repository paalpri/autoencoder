import midi, sys, os
import argparse
from my_parser_utils import pitch_to_note_conv, instrument_name

# parser = argparse.ArgumentParser(description='Process midi files to create the neural network\'s input')

# parser.add_argument('-f', '--filename', type=str,
#                     help='the midi filename')
# parser.add_argument('-m', '--get_midi', type=int, metavar='TRACK_ID', 
#                     help="print the notes and times")

# args = parser.parse_args()

class note:
    def __init__(self, _type, _id, _pitch, _pitch_number, _octave, _time, _relative,_unique,_velocity=0):
        self.type   = _type
        self.id = _id
        self.pitch  = _pitch
        self.pitch_number  = _pitch_number
        self.octave = _octave
        self.time   = _time
        self.relative   = _relative
        self.velocity = _velocity
        self.unique = _unique

    def __repr__(self):
        s = "%s[%d]\t(t=%d) " % (self.pitch, self.octave, self.time)
        #s = "(t=%d, %s) %s[%d]" % (self.time, self.type, self.pitch, self.octave)        
        #s = "%s[%d]" % (self.pitch, self.octave)
        return s

    def __lt__(self, other):
        if self.id == other.id:
            return self.relative > other.relative
        else:
            self.id < other.id

def get_elements_from_track(track):
    track_notes = []
    track_name = 'Unknown Track Title'
    track_instrument = 'Unknown Instrument'
    track_instrument_number = 99
    cur_abs_time = 0

    cont = 0
    dic = {}
    elements = []
    for element in track:
        if type(element) == midi.TrackNameEvent:
            track_name = element.text
        elif type(element) == midi.ProgramChangeEvent:
            track_instrument = instrument_name[element.data[0]] 
            track_instrument_number = element.data[0]
        
        elif type(element) == midi.events.NoteOnEvent:
            note_relative_time = element.tick
            (note_id, note_velocity) = element.data

            # # 21  22   23  24  25 26 27 28 29 30 31 32 33 ... 108
            # # A0  A#0  B0  C1  C#  D D# E  F  F# G  G# A      C8
            note_pitch  = (note_id-21)%12
            note_octave = (note_id)//12
           
            cur_abs_time += note_relative_time
            if note_velocity == 0: # this is actually a note being released, so we ignore it
                track_notes.append(note('RELEASE', note_id, pitch_to_note_conv[note_pitch], note_pitch, note_octave, cur_abs_time, note_relative_time, dic[note_id]))
                dic[note_id] = 0
                pass
            else:
                cont+=1
                dic[note_id] = cont
                track_notes.append(note('PRESS', note_id, pitch_to_note_conv[note_pitch], note_pitch, note_octave, cur_abs_time, note_relative_time, cont, note_velocity))

        elif type(element) != midi.events.EndOfTrackEvent:
            elements.append(element)


    return (track_name, track_instrument, track_instrument_number, track_notes, elements)

def get_groups(track_notes):
    groups = []
    cur_group = [track_notes[0]]
    last_note_time = track_notes[0].time
    for note_info in track_notes[1:]:
    	if(note_info.type == 'PRESS'):
	        if note_info.time != last_note_time:
	            if len(cur_group) != 0:  
	                groups.append(cur_group)
	                cur_group = []
	        cur_group.append(note_info)
	        last_note_time = note_info.time
    if len(groups) == 0:
        groups.append(cur_group)
    return groups


def reconstruct_original(filename, id_track, window):
    pattern = midi.read_midifile(filename)
    for (track_id, ttt) in enumerate(pattern):
        if id_track == track_id:
            (track_name, track_instrument, track_instrument_number, track_notes,elements) = get_elements_from_track(ttt)
            
            #print track_notes


            if len(track_notes) == 0:
                continue

            groups = get_groups(track_notes)

            pattern = midi.Pattern()
            track = midi.Track()
            pattern.append(track)

            for element in elements:
                track.append(element)

            instrument = midi.ProgramChangeEvent(data=[42])
            track.append(instrument)


            all_notes = []
            ids = []
            for cur_group in groups:
                cur_group.sort()
                if cur_group[0].pitch_number in [5,7,9,10,0,2,4]: # only consider the d-major scale notes
                    if len(all_notes) == 0:
                        all_notes.append(cur_group[0])
                        ids.append(cur_group[0].unique)      
                    elif cur_group[0].pitch_number != all_notes[-1].pitch_number:
                        all_notes.append(cur_group[0])      
                        ids.append(cur_group[0].unique)
                cur_group = []

            aux = {5:0, 7:1, 9:2, 10:3, 0:4, 2:5, 4:6}

            if len(all_notes) % window != 0:
                all_notes = all_notes[:- (len(all_notes) % window)]

            all_end_notes = []
            for note in all_notes:
                on = midi.NoteOnEvent(tick=note.relative, velocity=note.velocity, pitch=note.id)
                track.append(on)
                for tn in track_notes:
                    if tn.unique == note.unique and tn.type == 'RELEASE':
                        all_end_notes.append(tn)
                        off = midi.NoteOnEvent(tick=tn.relative, velocity=0, pitch=tn.id)
                        track.append(off)

            # Add the end of track event, append it to the track
            eot = midi.EndOfTrackEvent()
            track.append(eot)
            # Save the pattern to disk
            midi.write_midifile("original.mid", pattern)

    return all_notes, all_end_notes,elements


def vae_midi(all_notes, all_end_notes, elements, filename):
    pattern = midi.Pattern()
    track = midi.Track()


    for element in elements:
        track.append(element)

    instrument = midi.ProgramChangeEvent(data=[42])
    track.append(instrument)


    for note in all_notes:
        on = midi.NoteOnEvent(tick=note.relative, velocity=note.velocity, pitch=note.id)
        track.append(on)
        for tn in all_end_notes:
            if tn.unique == note.unique and tn.type == 'RELEASE':
                off = midi.NoteOnEvent(tick=tn.relative, velocity=0, pitch=note.id)
                track.append(off)

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent()
    track.append(eot)

    pattern.append(track)

    # Save the pattern to disk
    midi.write_midifile(filename, pattern)