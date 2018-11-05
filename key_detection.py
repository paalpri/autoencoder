from music21 import converter, instrument, note, chord
from music21 import *
import pretty_midi
from pprint import pprint
import os
import sys
import glob
from pathlib import Path

import ntpath



directory = sys.argv[1]
#save_directiory = sys.argv[2]
newMajorDirectory = 'sorted_violin_midis'


for filepath in glob.iglob(directory + '/' + '**/*.mid', recursive = True):
    print(filepath)
    midi = converter.parse(filepath) 
    key = midi.analyze('key')
    newDic = newMajorDirectory + '/' + key.tonic.name + '_' +  key.mode
    print(newDic)
    if  not os.path.isdir(newDic):
        os.mkdir(newDic)

    Path(filepath).rename(newDic + '/' + ntpath.basename(filepath))