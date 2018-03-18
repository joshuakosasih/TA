import os

myfile = open('ner_2_test.txt', 'r')

netype = []
necnt = []
mydict = []
enx = '</'
eny = '>'

for line in myfile:
    mydict.append(line)


def updateNEType(s):
    found = False
    for idx, x in enumerate(netype):
        if s == x:
            found = True
            necnt[idx] = necnt[idx]+1
    if not found:
        netype.append(s)
        necnt.append(1)

def enamexCounter(s):
    count = 0
    while enx in s:
        count = count + 1
	s = s[s.find(enx):]
	updateNEType(s[s.find(enx) + len(enx):s.find(eny)])
        s = s[len(enx):]
    return count

sums = 0
for line in mydict:
    sums = sums + enamexCounter(line)

print "Sums", sums
print "NE type", netype
print "NE count", necnt
