import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
import argparse
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
import random

slash='\\'
steering_correction = 0.2
nb_epoch=20

def generator(data_path, samples, batch_size=32):
	if len(data_path) > 0 and data_path[-1] != '\\' and data_path[-1] != '/':
		data_path += slash
	num_samples = len(samples)
	while 1: #loop forever to keep generator from early termination
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images=[]
			angles=[]
			for batch_sample in batch_samples:
				center_img_path = batch_sample[0]
				left_img_path = batch_sample[1]
				right_img_path = batch_sample[2]
				center_image = cv2.imread('{}IMG{}{}'.format(data_path, slash, center_img_path.split(slash)[-1]))
				left_image = cv2.imread('{}IMG{}{}'.format(data_path, slash, left_img_path.split(slash)[-1]))
				right_image = cv2.imread('{}IMG{}{}'.format(data_path, slash, right_img_path.split(slash)[-1]))
				center_angle = float(batch_sample[3])
				left_angle = center_angle + steering_correction
				right_angle = center_angle - steering_correction
				if center_image is not None and left_image is not None and right_image is not None:
					images.append(center_image)
					images.append(left_image)
					images.append(right_image)
					angles.append(center_angle)
					angles.append(left_angle)
					angles.append(right_angle)
					#flip imagesage to increase data variability
					images.append(cv2.flip(center_image, 1))
					images.append(cv2.flip(left_image, 1))
					images.append(cv2.flip(right_image, 1))
					angles.append(center_angle * (-1))
					angles.append(left_angle * (-1))
					angles.append(right_angle * (-1))
				else:
					print('Error: could not load {}'.format(curr))
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def load_data(data_path):
	if len(data_path) > 0 and data_path[-1] != '\\' and data_path[-1] != '/':
		data_path += slash
	csvfilename = '{}driving_log.csv'.format(data_path)
	lines = []
	with open(csvfilename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)				
	train_samples, validation_samples = train_test_split(lines, test_size=0.2)
	return train_samples, validation_samples


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='train driving model')
	parser.add_argument(
		'data_path',
		type=str,
		help='Path to training data and images.',
		nargs='?',
		default=''
	)
	args = parser.parse_args()

	train_samples, validation_samples = load_data(args.data_path)
	train_generator = generator(args.data_path, train_samples, batch_size=32)
	validation_generator = generator(args.data_path, validation_samples, batch_size=32)

#	ch, row, col = 3, 80, 320 #Trimmed image format

	model = Sequential()

	model.add(Lambda(lambda x: x/255. -0.5, 
		input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
	#model.add(MaxPooling2D())
	model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
	#model.add(MaxPooling2D())
	model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
	#model.add(MaxPooling2D())
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')

	#histogram
	history_object = model.fit_generator( train_generator, 
		samples_per_epoch = len(train_samples), 
		validation_data=validation_generator, 
		nb_val_samples=len(validation_samples),
		nb_epoch=nb_epoch)

	#print the keys contained in the history object
	print(history_object.history.keys())

	# plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

	import gc; gc.collect()

	model.save('model.h5')