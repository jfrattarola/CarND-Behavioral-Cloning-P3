import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import argparse

slash='\\'

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
		source_path = line[0]
		filename = source_path.split(slash)[-1]
		current_path = '{}IMG{}{}'.format(data_path, slash, filename)
		#print(current_path)
		image = cv2.imread(current_path)
		if image is not None:
			images.append(image)
			measurement = float(line[3])
			measurements.append(measurement)
			#flip image to increase data variability
			images.append(cv2.flip(image, 1))
			measurements.append(measurement * (-1.))
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
	model.add(Flatten(input_shape=(160,320,3)))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

	model.save('model.h5')