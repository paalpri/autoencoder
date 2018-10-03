import os
path = 'data_2/'
files = os.listdir(path)
i = 0

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.mid'))
    i = i+1