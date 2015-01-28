import os
#import shutil
from PIL import Image
import numpy

size = (100, 100)

# BUILD TEST SET
def makeTest():
	f = open('./data/flat100/test.txt', 'w')

	imageFileNames = os.listdir("./data/original/test/")
	for fileWithExt in imageFileNames:
		image = Image.open("./data/original/test/"+fileWithExt)
		image.thumbnail(size, Image.ANTIALIAS)
		flat100 = Image.new('RGB', size, (255, 255, 255))
		flat100.paste(image, ((size[0] - image.size[0]) / 2, (size[1] - image.size[1]) / 2))
		flat100.save("./data/flat100/test/"+fileWithExt, "JPEG")
		f.write(fileWithExt + " " + "0" + "\n")

	f.close()

# BUILD TRAINING SET
def makeTrain():
	with open('./categories') as f:
	    categories = f.read().splitlines()

	f1 = open('./data/flat100/train.txt', 'w')
	f2 = open('./data/flat100/val.txt', 'w')

	i = 0
	for label in categories:
		imageFileNames = os.listdir("./data/original/train/"+label+"/")
		for fileWithExt in imageFileNames:
			image = Image.open("./data/original/train/"+label+"/"+fileWithExt)
			image.thumbnail(size, Image.ANTIALIAS)
			flat100 = Image.new('RGB', size, (255, 255, 255))
			flat100.paste(image, ((size[0] - image.size[0]) / 2, (size[1] - image.size[1]) / 2))
			flat100.save("./data/flat100/train/"+fileWithExt, "JPEG")
			flat100.save("./data/flat100/val/"+fileWithExt, "JPEG")
			f1.write(fileWithExt + " " + str(i) + "\n")
			f2.write(fileWithExt + " " + str(i) + "\n")
		print i
		i = i + 1

	f1.close()
	f2.close()
	
#fileNoExt = os.path.splitext(fileWithExt)[0]
	#shutil.copyfile("../data/train/"+label+"/"+fileWithExt, "../data/flat100/train/"+label+"|"+fileWithExt)

import lmdb

def openDbs():
	env = lmdb.open('../plankton_train_db', max_dbs=100)
	plankton_train_db = env.open_db()
	print "We're in!"
	print env.info()
	with env.begin() as txn:
		print "status"
 		cursor = txn.cursor(plankton_train_db)
 		print "status1"
 		for key, value in cursor:
 			print "weee"
 			print key + " " + value

# PUT TRAINING DATA INTO DB
# 
# 	imageFileNames = os.listdir("../data/flat100/train/")
# 	for fileWithExt in imageFileNames:
# 		label, key = fileWithExt.split('|')
# 		imageFile = Image.open("../data/flat100/train/"+fileWithExt)
# 		value = list(imageFile.getdata())
		# other option: numpy.asarray(imageFile)
		#cursor.put(key, value)
