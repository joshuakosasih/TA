import csv
import numpy as np

cname = raw_input('Enter csv file name: ')

f = open(cname + '.csv', 'r')

creader = csv.reader(f)

cdict = []

for row in creader:
	cdict.append(row)

def addRdict(row):
	for r in rdict:
		if r[0] == row[0]:
			if int(row[1]) in seedChoice:
				r[1] = r[1] + float(row[2])

sC = input('Enter seed choice array: ')
seedChoice = sC

rowsum = 0
firstCol = ''
rdict = []
for row in cdict:
	if row[0] != firstCol:
		rdict.append([row[0], 0])
		firstCol = row[0]

for row in cdict[1:]:
	addRdict(row)

for row in rdict[1:]:
	row[1] = row[1] / float(len(seedChoice))

print np.array(rdict)
f.close()
