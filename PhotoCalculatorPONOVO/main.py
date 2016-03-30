import cv2
import collections
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.spatial import distance
import wolframalpha
# k-means
from sklearn.cluster import KMeans
# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
from fun import *
from train import *
import pickle

def recognize_digit(ann, slika2):
	sli = image_bin(image_gray(slika2)) 
	test_inputs = resize_region(sli)
	test = matrix_to_vector(scale_to_range(test_inputs))
	
	test = test.reshape(1, test.shape[0]) # strange error when doing with both theano and numpy
	print(test.shape)
	result = ann.predict(test, batch_size=1)

	brojevi = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	print display_result(result, brojevi)


def rec_all(ann, naziv):
	slicica = load_image(naziv)
	images  = img_pre_img(slicica)
	plt.imshow(slicica,'gray')
	plt.waitforbuttonpress()
	for img in images:
		print recognize_digit(ann, img)


def recognize_digit(ann, img):
	test = matrix_to_vector(scale_to_range(img))	
	test = test.reshape(1, test.shape[0]) # strange error when doing with both theano and numpy
	result = ann.predict(test, batch_size=1)
	brojevi = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	ret_val =  display_result(result, brojevi)
	return ret_val

def validate_recognition(ann, path, file_name,validation_digit):
	number_recognized = 0
	
	fajl = path+file_name
	
	with open(fajl) as f:
		content = f.readlines()

	for i in range(0,100):
		naziv = path+content[i][:-1]
		slicica = img_preproc_train(naziv)
		val =  recognize_digit(ann, slicica)
		if val[0] == validation_digit:
			number_recognized+=1

	print ("Recognized " + str(number_recognized))


def main():

	print ('Start!')

	digits1000, images1000, avg = create_train_hundred_test()
	
	inputs1000 = prepare_for_ann(images1000)
	outputs1000 = sredi_izlaz_hundred(digits1000)
	#ann1000_test = create_ann()
	#print ('Obucavanje...')
	#ann1000_test = train_ann(ann1000_test, inputs1000, outputs1000)
	#print('Gotovo obucavanje')
	#pickle.dump( ann1000_test, open("saveANN1000_test.p", "wb"))
	ann1000_test = pickle.load( open( "saveANN1000_test.p", "rb" ))
	#print ("Moja slika")
	#rec_all(ann1000_test, 'images/img5.png')
	print ("MNIST slicice")
	validate_recognition(ann1000_test, 'digit_separated/digit_0/', 'test100.txt', '0')

	#plt.imshow(avg[9], 'gray', interpolation='nearest')
	#plt.waitforbuttonpress()

	print ('Kraj')


if __name__ == "__main__":
	main()
