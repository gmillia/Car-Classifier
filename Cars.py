import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import os
import re

#plt.imshow(res) 
##plt.show()  #will show the scaled image

#Starting path for training images
train_start = "C:\\Users\\gmill\\Documents\\Python\\Car_Class\\train"
train_images = [None] * 8145  #Array of matrices that hold images
train_images_names = []  #image names: 0001.img etc
train_images_targets = []  #image target: class 150 etc
train_file = "anno_train.csv"

#Helper function to read the initial data
def readData(file, names, targets):
	with open(file) as csv_file:
		lines = csv_file.readlines()

	for line in lines:
		data = line.split(',')
		names.append(data[0])
		targets.append(data[5])

#Helper function to exctract the indeces from the image
#E.g.: path/00010.jpg -> 10
def getIndex(imageName):
	n = imageName.lstrip('0')  #strip from leading 0's
	n = re.sub("\D", "", n)  #strip from .jpg
	return n

def getImages(startingPath):
	cnt = 1
	for dirpath, dirnames, filenames in os.walk(startingPath):
		for filename in filenames:
			index = getIndex(filename)
			filename = os.path.join(dirpath, filename)
			img = cv2.imread(filename)
			res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
			train_images[int(index)] = res
			print("Training image #", cnt)
			cnt+=1

readData(train_file, train_images_names, train_images_targets)
getImages(train_start)
