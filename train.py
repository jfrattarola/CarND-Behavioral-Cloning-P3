import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
import argparse

slash='\\'
steering_correction = 0.2

def load_data(data_path):
	if len(data_path) > 0 and data_path[-1] != '\\' and data_path[-1] != '/':
		data_path += slash
	csvfilename = '{}driving_log.csv'.format(data_path)
	lines = []
	with open(csvfilename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	images=[]
	measurements=[]
	for line in lines:
		center_img_path = line[0]
		left_img_path = line[1]
		right_img_path = line[2]
		center_image = cv2.imread('{}IMG{}{}'.format(data_path, slash, center_img_path.split(slash)[-1]))
		left_image = cv2.imread('{}IMG{}{}'.format(data_path, slash, left_img_path.split(slash)[-1]))
		right_image = cv2.imread('{}IMG{}{}'.format(data_path, slash, right_img_path.split(slash)[-1]))
		center_measurement = float(line[3])
		left_measurement = center_measurement + steering_correction
		right_measurement = center_measurement - steering_correction

		if center_image is not None and left_image is not None and right_image is not None:
			images.append(center_image)
			images.append(left_image)
			images.append(right_image)
			measurements.append(center_measurement)
			measurements.append(left_measurement)
			measurements.append(right_measurement)
			#flip image to increase data variability
			images.append(cv2.flip(center_image, 1))
			images.append(cv2.flip(left_image, 1))
			images.append(cv2.flip(right_image, 1))
			measurements.append(center_measurement * (-1))
			measurements.append(left_measurement * (-1))
			measurements.append(right_measurement * (-1))
		else:
			print('Error: could not load {}'.format(curr))
	X_train = np.array(images)
	y_train = np.array(measurements)
	return X_train, y_train


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

	X_train, y_train = load_data(args.data_path)
	
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
	model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

	model.save('model.h5')