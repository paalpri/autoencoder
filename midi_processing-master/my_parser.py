# First, install this package: https://github.com/vishnubob/python-midi
# Here's the documentation on their classes/data structures: https://github.com/vishnubob/python-midi/issues/57

class note:
    def __init__(self, _type, _id, _pitch, _pitch_number, _octave, _time):
        self.type   = _type
        self.id = _id
        self.pitch  = _pitch
        self.pitch_number  = _pitch_number
        self.octave = _octave
        self.time   = _time

    def __repr__(self):
        s = "%s[%d]\t(t=%d) " % (self.pitch, self.octave, self.time)
        #s = "(t=%d, %s) %s[%d]" % (self.time, self.type, self.pitch, self.octave)        
        #s = "%s[%d]" % (self.pitch, self.octave)
        return s

    def __lt__(self, other):
        return self.id < other.id

def get_elements_from_track(track):
    track_notes = []
    track_name = 'Unknown Track Title'
    track_instrument = 'Unknown Instrument'
    track_instrument_number = 99
    cur_abs_time = 0

    for element in track:
    
        if type(element) == midi.TrackNameEvent:
            track_name = element.text
        if type(element) == midi.ProgramChangeEvent:
            track_instrument = instrument_name[element.data[0]] 
            track_instrument_number = element.data[0]
        
        if type(element) == midi.events.NoteOnEvent:
            
            note_relative_time = element.tick
            (note_id, note_velocity) = element.data

            # # 21  22   23  24  25 26 27 28 29 30 31 32 33 ... 108
            # # A0  A#0  B0  C1  C#  D D# E  F  F# G  G# A      C8
            note_pitch  = (note_id-21)%12
            note_octave = (note_id-21)/12
            if note_id > 23:
                note_octave += 1

            cur_abs_time += note_relative_time
            if note_velocity == 0: # this is actually a note being released, so we ignore it
                #track_notes.append( note('RELEASE', note_id, pitch_to_note_conv[note_pitch], note_pitch, note_octave, cur_abs_time) )
                pass
            else:
                track_notes.append( note('PRESS', note_id, pitch_to_note_conv[note_pitch], note_pitch, note_octave, cur_abs_time) )

    return (track_name, track_instrument, track_instrument_number, track_notes)

def get_groups(track_notes):
    groups = []
    cur_group = [track_notes[0]]
    last_note_time = track_notes[0].time
    for note_info in track_notes[1:]:
        if note_info.time != last_note_time:
            if len(cur_group) != 0:      
                groups.append(cur_group)
                cur_group = []
        cur_group.append(note_info)
        last_note_time = note_info.time

    return groups

def print_track(pattern, wanted_track, window, shift):
    for (track_id, track) in enumerate(pattern):
        if track_id == wanted_track:
            
            (track_name, track_instrument, track_instrument_number, track_notes) = get_elements_from_track(track)

            if len(track_notes) == 0:
                continue
               
            groups = get_groups(track_notes)

            all_notes = []
            for cur_group in groups:
                cur_group.sort()
                all_notes.append(cur_group[0])      
                cur_group = []

            for i in range(0, len(all_notes)-window, shift):
                print ' '.join([str(all_notes[i+j].pitch_number) for j in range(window)])

def check_tracks(filename, pattern, search_instrument, percentage):
    for (track_id, track) in enumerate(pattern):

        (track_name, track_instrument, track_instrument_number, track_notes) = get_elements_from_track(track)

        if len(track_notes) == 0:
            continue
                
        groups = get_groups(track_notes)

        single = 0
        not_single = 0
        for cur_group in groups:
            if(len(cur_group) == 1):
                single += 1
            else:
                not_single += 1
        if (single+not_single) != 0:
            if(track_instrument_number == search_instrument and float(single)/(single+not_single) >= percentage):
                print "\n%s %d" % (filename, track_id)

       

import midi, sys, os
import argparse
from my_parser_utils import pitch_to_note_conv, instrument_name

parser = argparse.ArgumentParser(description='Process midi files to create the neural network\'s input')

parser.add_argument('-f', '--filename', type=str,
                    help='the midi filename')
parser.add_argument('-i', '--info', action='store_true',
                    help='print the name, id and number of notes of each track in the midi file')
parser.add_argument('-c', '--check_tracks', nargs=2, type=float, metavar=('INTRUMENT', 'PERCENTAGE'),
                    help="print the filename and the track id for all the tracks that are with the given instrument number and with more than the given percentage of single notes")
parser.add_argument('-p', '--format', nargs=3, type=str, metavar=('FILENAME', 'WINDOW', 'SHIFT'),
                    help="print the notes in the input format to the neural net, WINDOW notes per line with a shift of SHIFT notes")
parser.add_argument('-t', '--track_info', type=int, metavar='TRACK_ID', 
                    help="print the notes of that track")
parser.add_argument('-a', '--all', action='store_true',
                    help="print the notes of all tracks")
parser.add_argument('-g', '--get_files', nargs=3, type=str, metavar=('DIR_NAME','INTRUMENT', 'PERCENTAGE'),
                    help="print the filename and the track id for all the tracks that are with the given instrument number and with more than the given percentage of single notes for each file in the informed dir")

args = parser.parse_args()



if args.check_tracks:
    search_instrument = int(args.check_tracks[0])
    percentage = args.check_tracks[1]
    pattern = midi.read_midifile(args.filename)
    check_tracks(args.filename, pattern, search_instrument, percentage)

elif args.get_files:
    dirname = args.get_files[0]
    search_instrument = int(args.get_files[1])
    percentage = float(args.get_files[2])

    paths = [os.path.join(dirname, name) for name in os.listdir(dirname)]
    files = [file for file in paths if os.path.isfile(file)]
    midis = [file for file in files if file.lower().endswith(".mid")]

    for m in midis:
        pattern = midi.read_midifile(m)
        check_tracks(m, pattern, search_instrument, percentage)


elif args.format:
    window = int(args.format[1])
    shift = int(args.format[2])
    filename_tracks = args.format[0]
 
    file = open(filename_tracks, 'r')
    lines = file.readlines()
    for line in lines:
        line = line.split(" ")
        if len(line)==2:
            pattern = midi.read_midifile(line[0])
            print_track(pattern, int(line[1]), window, shift)

elif args.info:
    pattern = midi.read_midifile(args.filename)
    for (track_id, track) in enumerate(pattern):
        (track_name, track_instrument, track_instrument_number, track_notes) = get_elements_from_track(track)
        print "\tTrack %02d [%25s]\t('%s'; %d notes)" %(track_id, track_instrument, track_name, len(track_notes))
        
elif args.all or args.track_info != None:
    pattern = midi.read_midifile(args.filename)
    for (track_id, track) in enumerate(pattern):
        if args.all or args.track_info == track_id:
            (track_name, track_instrument, track_instrument_number, track_notes) = get_elements_from_track(track)
            
            print "\nProcessing track %02d [%25s]\t('%s'; %d notes)" %(track_id, track_instrument, track_name, len(track_notes))
            #print track_notes

            if len(track_notes) == 0:
                continue

            groups = get_groups(track_notes)
            for cur_group in groups:     
                print '\t', ', '.join([str(i) for i in cur_group])
                #print "[%d]\t%s(%d)" % (note_info.time, note_info.pitch, note_info.octave)

