# /path/to/build/tools/caffe train --solver=../src/solver.prototxt
# export PYTHONPATH=$PYTHONPATH:/path/to/caffe/python

import caffe
import numpy as np
import cv2
import time, sys
import scipy.io
import matplotlib.pyplot as plt

start = time.time()
model = './opflownet.prototxt'
trained = './opflownet.caffemodel'
flicPrefix = '../FLIC/images/'
moviePrefix = '../FLICMOVIES/'

exampleIdxs_all = [201,239,244,263,279,287,335,656,689,758,870,874,878,959,970,996,998]
exampleIdxs = [664, 656, 870, 874, 998]

LSHO,LELB,LWRI,RSHO,RELB,RWRI = 0,1,2,3,4,5
def gaussian(pt1,pt2,sig):
	d = np.linalg.norm(pt2-pt1)
	return np.exp(-d**2/(2*sig**2))

def ptToHeatmap(x,y,dim,sig):
	xRng = xrange(int(max(0,x-3*sig)), int(min(dim[0],x+3*sig)))
	yRng = xrange(int(max(0,y-3*sig)), int(min(dim[1],y+3*sig)))
	e = .001
	arr = np.zeros(dim,dtype=np.float)
	pt = np.array([x,y])
	for i in xRng:
		for j in yRng:
			arr[i,j] = gaussian(np.array([i,j]),pt,sig)
			if arr[i,j] < e: arr[i,j] = 0.
	return arr

def heatmapToPt(hm):
	total = 0.
	totalX = 0.
	totalY = 0.
	for i in xrange(hm.shape[0]):
		for j in xrange(hm.shape[1]):
			total += hm[i,j]
			totalX += i*hm[i,j]
			totalY += j*hm[i,j]
	if total == 0:
		return -1,-1
	return np.array([totalX/total, totalY/total])

def getPartCoords(ex,part):
	return np.array([ex["coords"][0][part], ex["coords"][1][part]])

def loadArray(filename):
	arr = []
	with open(filename,'r') as f:
		for line in f:
			line = line[:-1] # Remove \n
			vals = np.array(line.split(', '), np.float)
			arr += [vals]
	return np.array(arr)
	
def coordTransform(torso, pt):
    # Translate
    newPt = pt - torso[:2] + pad
    # Scale
    w = torso[2]-torso[0]+2*pad[0]
    h = torso[3]-torso[1]+2*pad[1]
    newPt = [newPt[1]*float(res[0])/w, newPt[0]*float(res[1])/h]
    return newPt

def invCoordTransform(torso, pt):
    # Scale
    w = torso[2]-torso[0]+2*pad[0]
    h = torso[3]-torso[1]+2*pad[1]
    newPt = [pt[1]*float(h)/res[1], pt[0]*float(w)/res[0]]
    # Translate
    newPt = newPt - pad + torso[:2]
    return newPt

net = caffe.Classifier(model, trained)
net.set_phase_test()
net.set_mode_cpu()

inputFile = '../data/input_lw_test.txt'
outputFile = '../data/output_lw_test.txt'

e = scipy.io.loadmat('../data/examples.mat')
e = e['examples'][0]
e.sort(order=('moviename','currframe'))
modec = loadArray("../data/MODEC_out_test.txt")
res = (60,45)
pad = np.array([100,80])
inputSize = res[0]*res[1]*2
outputSize = res[0]*res[1]

inputs = np.fromfile(inputFile,sep=' ')
inputs = inputs.reshape((inputs.shape[0]/inputSize,inputSize))
outputs = np.fromfile(outputFile,sep=' ')
outputs = outputs.reshape((outputs.shape[0]/outputSize,outputSize))

data = np.zeros((len(inputs),2,res[0],res[1]),np.float32)
label = np.zeros((len(inputs),2),np.float32)

for i in xrange(len(inputs)):
	torso = e[modec[i][0]]['torsobox'][0]
	pt = getPartCoords(e[modec[i][0]], LWRI)
	data[i][0] = inputs[i][:outputSize].reshape(res)
	data[i][1] = inputs[i][outputSize:].reshape(res)
	label[i] = coordTransform(torso,pt)
	
allD1 = []
allD2 = []
count = 0
diffArray = np.zeros(5)
for i in xrange(len(inputs)):
	print "\r",i,
	sys.stdout.flush()
	torso = e[modec[i][0]]['torsobox'][0]
	net.blobs['data'].data[...] = data[i][:]
	net.forward()
	netin = net.blobs['data'].data
	in1 = netin[0][0][:]
	in2 = netin[0][1][:]
	convdata = net.blobs['conv1'].data
	conv1 = convdata[0][0][:]
	netout = net.blobs['full3'].data
	netout = np.array([netout[0][0][0][0]*res[0],netout[0][1][0][0]*res[1]])
	
	hm = ptToHeatmap(netout[0],netout[1],res,3)
	
	correct = outputs[i].reshape(res)
	pt = heatmapToPt(correct)
	ptModec = heatmapToPt(in1)

	flicpath = flicPrefix+e[modec[i][0]]["filepath"][0]
	flicImg = cv2.imread(flicpath)

	flicPt = map(int,invCoordTransform(torso,pt))
	flicPtModec = map(int,invCoordTransform(torso,ptModec))
	flicPtNet = map(int,invCoordTransform(torso,netout))
	
	cv2.circle(flicImg, (flicPt[0],flicPt[1]), 5, (0, 255, 0), -1)
	cv2.circle(flicImg, (flicPtModec[0],flicPtModec[1]), 5, (255, 0, 0), -1)
	cv2.circle(flicImg, (flicPtNet[0],flicPtNet[1]), 5, (0, 0, 255), -1)
	
	if not (pt[0] == -1 or ptModec[0] == -1): 
		d1 = np.linalg.norm(pt-netout)
		allD1 += [d1]
		d2 = np.linalg.norm(pt-ptModec)
		allD2 += [d2]
		count += 1

		dDiff = d2 - d1
		if dDiff > 7:
			diffIdx = 4
		elif dDiff > 1:
			diffIdx = 3
		elif dDiff > -1:
			diffIdx = 2
		elif dDiff > -7:
			diffIdx = 1
		else:
			diffIdx = 0
		diffArray[diffIdx] += 1
	
	if d1 > 9 and d1 < 11:
		# Get the movie image
		moviename = e[modec[i][0]]["moviename"][0]
		moviefile = moviePrefix+moviename+".m4v"
		v = cv2.VideoCapture(moviefile)
		frame = e[modec[i][0]]["currframe"][0][0] - 4
		v.set(1,frame)
		ret,movimg = v.read()
		print "Net:",netout, "Actual:",pt, "MODEC:",ptModec
		print i,"Net dist:",d1,"MODEC dist:",d2
		
		in1 = cv2.resize(in1,(3*res[1],3*res[0]))
		in2 = cv2.resize(in2,(3*res[1],3*res[0]))
		conv1 = cv2.resize(conv1,(2*res[1],2*res[0]))
		cv2.imshow("Modec Estimate", in1)
		cv2.imshow("Optical flow", in2)
		cv2.imshow("Conv 1", conv1)
		
		cv2.imshow("FLIC",flicImg)
		cv2.imshow("Movie",movimg)

		k = cv2.waitKey()
		if k == 27:
			break

	
#print betterCount, count, float(betterCount)/count
print diffArray
t = [0.5,1.5,2.5,3.5,4.5]
ticks = ["Much worse","Worse","Same","Better","Much Better"]
plt.bar(t,diffArray)
plt.xticks(t, ticks)
plt.show()

dists = [.5*(i+1) for i in xrange(20)]
curveNet = np.zeros(len(dists),np.float)
curveModec = np.zeros(len(dists),np.float)
curveBest = np.zeros(len(dists),np.float)
for i in xrange(len(allD1)):
	for j in xrange(len(dists)):
		minDist = min(allD1[i],allD2[i])
		if minDist < dists[j]:
			curveBest[j] += 1
		if allD1[i] < dists[j]:
			curveNet[j] += 1
		if allD2[i] < dists[j]:
			curveModec[j] += 1

curveNet = curveNet/len(allD1)
curveModec = curveModec/len(allD1)
curveBest = curveBest/len(allD1)

print ""
print "Mean distance: MODEC -",np.mean(allD1),"Net -",np.mean(allD2)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(dists,curveNet)
ax.plot(dists,curveModec)
ax.plot(dists,curveBest)
ax.set_ylim([0,1])
plt.show()

