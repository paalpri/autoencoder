
from __future__ import division, print_function, absolute_import
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from skimage import transform
from music21 import converter, instrument, note, chord
from music21 import *
import pretty_midi
from pprint import pprint
import os



base_dir = 'data_2'
for dir_item in base_dir:
    files = glob.glob(dir_item + '/*.mid')
    i = 1
    for f in files:
        os.rename(f, os.path.join(dir_item, str(i) + '.mid'))
        i += 1