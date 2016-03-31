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
from printRecognition import *

def recognize_all(ann, naziv):
	slicica = load_image(naziv)
	images  = img_pre_img(slicica)
	res = ''
	for img in images:
		res +=  recognize_digit(ann, img)[0]

	print ("Recognized "+res)


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

	print ("cypher "+validation_digit+" recognized " + str(number_recognized)+' of 100')

def validate_mnist(ann):

	for i in range(10):
		naziv = 'digit_separated/digit_'+str(i)+'/'
		validate_recognition(ann, naziv, 'test100.txt', str(i))


def main():

	print ('Start...')

	#digits1000, images1000, avg = create_train_hundred_test()
	
	#inputs1000 = prepare_for_ann(images1000)
	#outputs1000 = sredi_izlaz_thousand(digits1000)
	#ann_best = create_ann()
	#print ('Obucavanje...')
	#ann_best= train_ann(ann_best, inputs1000, outputs1000)
	#print('Gotovo obucavanje')
	#pickle.dump( ann_best, open("saveANN_best.p", "wb"))
	

	ann1000_best = pickle.load( open( "saveANN1000_test.p", "rb" ))

	print ("\n\nMNIST dataset")
	validate_mnist(ann1000_best)


	print ("\n\nHandwritten digits recognition")
	recognize_all(ann1000_best, 'images/img6.png')


	#plt.imshow(avg[8], 'gray', interpolation='nearest')
	#plt.waitforbuttonpress()


	# PRINTED RECOGNITION
	print ("\n\nPrinted cyphers recognition")
	print_recognition()


if __name__ == "__main__":
	main()
