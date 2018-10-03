
from __future__ import division, print_function, absolute_import
import glob
#import os
#import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from music21 import converter, instrument, note, chord
from music21 import *
import pretty_midi
from pprint import pprint
SONG_LENGTH = 256


def midi_to_roll(files):
    piano_roll = [[0] * SONG_LENGTH for x in range(128)]
    notes = []
    for file in glob.glob(files):
        counter = 0
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if(counter < SONG_LENGTH): 
                print(element)
            if isinstance(element, note.Note):
                midi_numb = int(element.pitch.ps)
                if(counter < SONG_LENGTH):
                    piano_roll[midi_numb][counter] = 1
            elif isinstance(element, chord.Chord):
                for p in element.pitches: 
                    midi_numb = int(p.ps) 
                    if(counter < SONG_LENGTH):  
                        piano_roll[midi_numb][counter] = 1
        counter += 1
pprint(piano_roll)

