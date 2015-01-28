###########################
# Predict using Caffe model
###########################
import numpy as np
import matplotlib.pyplot as plt
import os

#Make sure that caffe is on the python path:
caffe_root = '/home/aqwin/Code/planktonCaffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/aqwin/Code/planktonCaffe/plankton/deploy.prototxt'
PRETRAINED = '/home/aqwin/Code/planktonCaffe/plankton/planknet_iter_20000.caffemodel'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net.set_phase_test()
net.set_mode_gpu()

with open('./categories') as f:
	categories = f.read().splitlines()

f = open('./submission.csv', 'w')

f.write('image')
for species in categories:
	f.write(","+species)
f.write('\n')

imageFileNames = os.listdir("../data/flat100/test/")
i = 0
for fileWithExt in imageFileNames:
	prediction = net.predict([caffe.io.load_image("/home/aqwin/Code/planktonCaffe/plankton/data/flat100/test/"+fileWithExt)])
	f.write(fileWithExt + ',' + ','.join(map(str,prediction[0])) + '\n')
	i += 1
	print i

f.close()

#	i += 1
# 	print i
#plt.imshow(input_image)

# prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
# #print 'prediction shape:', prediction[0].shape
# plt.plot(prediction[0])
# print 'predicted class:', prediction[0]
# print sum(prediction[0])

# #plot result
# plt.plot(prediction[0])
# plt.show()

# data4D = np.zeros([256,1,1,130400]) #create 4D array, first value is batch_size, last number of inputs
# data4DL = np.zeros([256,1,1,121])  # need to create 4D array as output, first value is batch_size, last number of outputs
# data4D[0:256,0,0,:] = xtrain[0:256,:] # fill value of input xtrain is your value which you would like to predict

# print [(k, v[0].data.shape) for k, v in net.params.items()]
# net.set_input_arrays(data4D.astype(np.float32),data4DL.astype(np.float32))
# pred = net.forward()
# pred_normal = np.zeros([max_value,5])
# for i in range(0,max_value):
#  pred_normal[i,0] = pred['fc3'][i][0]
