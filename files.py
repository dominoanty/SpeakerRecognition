from os import listdir
from os.path import isfile, join
import re

mypath = './newdata/data/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
onlyones = []
for filename in onlyfiles:
    dups = re.search('[\w]+_2.wav',filename)
    if dups is None:
        onlyones.append(''.join(filename.split('_')[0]))
print(onlyones)

        

