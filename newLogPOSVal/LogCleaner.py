fname = raw_input('Enter file name: ')

fr = open(fname, 'r')
fw = open(fname.split('.')[0] + '-cleaned.txt', 'w')

x = 0
for line in fr:
    if '>' in line:
        x = 0  # do nothing
    elif '...' in line:
        x = 0  # do nothing
    elif '!' in line:
        x = 0  # do nothing
    else:
        fw.write(line)

fr.close()
fw.close()
