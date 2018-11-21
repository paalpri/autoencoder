
from pprint import pprint
import os
import sys
import glob
from pathlib import Path

import ntpath


cont = 0
for filepath in glob.iglob('sorted_midis/A_minor' + '/' + '**/*.mid', recursive = True):

    try: 
        Path(filepath).rename('sorted_midis/A_minor' + '/' +'midi'+str(cont)+'.mid')
        cont += 1
    
    except:
        continue

