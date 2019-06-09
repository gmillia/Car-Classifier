import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import os
import re
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

#brands = ['Acura', 'Hummer', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Daewoo', 'Dodge', 'Eagle', 'Ferrari', 'FIAT', 'Fisker', 'Ford', 'Geo Metro', 'GMC', 'Honda', 'Hyundai', 'HUMMER', 'Infinity', 'Isuzu', 'Jeep', 'Lambroghini', 'Land Rover', 'Lincoln', 'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz', 'MINI', 'Mitsubishi', 'Nissan', 'Plymouth', 'Porsche', 'Ram', 'Rolls-Royce', 'Scion', 'smart', 'Spyker', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']
brands = ['Acura', 'Audi', 'BMW', 'Chevrolet', 'Mercedes-Benz']
n = 5
limit_per_brand = 40
trainAcura = []
trainAudi = []
trainBMW = []
trainChevrolet = []
trainMercedes = []
brand_train_images = [trainAcura, trainAudi, trainBMW, trainChevrolet, trainMercedes]

testAcura = []
testAudi = []
testBMW = []
testChevrolet = []
testMercedes = []
brand_test_images = [testAcura, testAudi, testBMW, testChevrolet, testMercedes]


#brand_images = [None] * len(brands)
# We need to grab all the images that belong to a certain brand and put them in the 
# so we are going to go into the folder and grab each image and put into matrix

def getIndex(imageName):
	n = imageName.lstrip('0')  #strip from leading 0's
	n = re.sub("\D", "", n)  #strip from .jpg
	return n

def getImages(startingPath, brand, brand_list):
	#index = -1
	count = 0
	for dirpath, dirnames, filenames in os.walk(startingPath):
		#image_list = []
		#image_list = [None] * 8145
		#find index of the brand
		dirname = dirpath.split(os.path.sep)[-1]

		if brand in dirname:
			for filename in filenames:
				if count <= limit_per_brand:
				#ind = getIndex(filename)
					filename = os.path.join(dirpath, filename)
					img = image.load_img(filename, target_size=(224,224,3))
					x = image.img_to_array(img)
					#image_list[int(ind)] = x  #for finding the target
					#image_list.append(x)
					brand_list.append(x)
					
					print("Reading: ", brand, count)
					count+=1

		#brands_imgaes will contain all the images of a certain brand (a list)
		#if index != -1:
			#brand_images[index] = image_list


for i in range(0, len(brands)):
	getImages('train', brands[i], brand_train_images[i])


for i in range(0, len(brands)):
	getImages('test', brands[i], brand_test_images[i])


x_train = np.vstack((trainAcura, trainAudi, trainBMW, trainChevrolet, trainMercedes))
x_test = np.vstack((testAcura, testAudi, testBMW, testChevrolet, testMercedes))

y_train = np.hstack((np.zeros(len(x_train)//n), np.ones(len(x_train)//n), np.ones(len(x_train)//n)*2, np.ones(len(x_train)//n)*3, np.ones(len(x_train)//n)*4))
y_test = np.hstack(   (np.zeros(len(x_test)//n), np.ones(len(x_test)//n), np.ones(len(x_test)//n)*2, np.ones(len(x_test)//n)*3, np.ones(len(x_test)//n)*4)  )

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

base_model = ResNet50(weights = 'imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(n, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers: layer.trainable = False

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs = 20, batch_size = 32, shuffle=True)
#getImages('train')

"""
targets = []
for i in range (0, len(brands)):
	targets.append(i)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(47))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(brand_images, targets, epochs=5, batch_size=32)

"""
#x_train = np.vstack(brand_images)