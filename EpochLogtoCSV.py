import os
import nltk

"""
Preparing file
"""

name = raw_input('Enter file name: ')
myfile = open(name+'.txt', 'r')

outfile = open(name+'.csv', 'w')
outfile.write('Epoch, F-1 Score\n')
print "Processing..."

mydict = []
for line in myfile:
    mydict.append(line)

nextIsF1 = False

for line in mydict:
    if 'Epoch:' in line:
        outfile.write(line[7:-1])
    elif nextIsF1:
        outfile.write(', ' + line[:-1] + '\n')
        nextIsF1 = False
    elif 'F-1 Score (without O)' in line:
        nextIsF1 = True


outfile.close()

print "Done!"
