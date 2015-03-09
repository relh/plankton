import math

results = open('output.txt')

outfile = open('output_2.txt','w')

line = results.readline()

while line != '':
 stuff = line.split('m')[1]
 if ',' in stuff:
  stuffsplit = stuff.split(',',1)
  outputline = stuffsplit[0]
  outputarray = [math.exp(float(x)) for x in stuffsplit[1:]]
  outputline += ',' + ','.join(map(str, outputarray[:118]))
  outputline += ',0,'
  outputline += str(outputarray[119])
  outputline += ',0\n'
  outfile.write(outputline)

