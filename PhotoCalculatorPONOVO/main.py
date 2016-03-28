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

def recognize_digit(ann, path):
	slika2 = load_image(path)
	sli = image_bin(image_gray(slika2)) 
	test_inputs = resize_region(sli)
	test = matrix_to_vector(scale_to_range(test_inputs))
	
	test = test.reshape(1, test.shape[0]) # strange error when doing with both theano and numpy
	print(test.shape)
	result = ann.predict(test, batch_size=1)

	brojevi = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	print display_result(result, brojevi)


def get_region(path):
	slika = load_image(path)
	slika_bin = image_bin(image_gray(slika))
	sel_reg, kont = konture(slika.copy(), slika_bin)
	
	plt.imshow(slika_bin, 'gray')
	plt.waitforbuttonpress()

def image_resize(img):
	width = img.shape[1]
	height = img.shape[0]

	new_width = 28.0
	new_height =28.0

	if width>height:
		max_dim = width
	else:
		max_dim = height

	new_ratio = new_width / max_dim
	print ("Stare dimenzije i new ratio")
	print (width, height, new_ratio)
	img = cv2.resize(img, (0,0), fx = new_ratio, fy = new_ratio)
	print ("Nove dimenzije ")
	print (img.shape)
	# plt.imshow(img,  'gray')
	# plt.waitforbuttonpress()
	new_img = np.ones((28,28))*255
	
	if (img.shape[1]==28):
		d = (new_img.shape[0] - img.shape[0])/2
		print ("D je "+ str(d))
		for i in range(0, img.shape[0]):
			for j in range (0, img.shape[1]):
				new_img[d+i][j] = img[i][j]
	else:
		d = (new_img.shape[1] - img.shape[1])/2
		print ("D je "+str(d))
		for i in range(0, img.shape[0]):
			for j in range (0, img.shape[1]):
				new_img[i][d+j] = img[i][j]	

	return new_img


def nesto(path):
	naziv = path
	sl = load_image(naziv)
	slicica = image_bin(image_gray(sl))
	sel_reg, kont = konture_test(sl.copy(), slicica)
	# kont[0] je prva kontura 

	img = kont[0]
	#img = cv2.resize(img, (28,28), interpolation = cv2.INTER_NEAREST)
	img = image_resize(img)
	plt.imshow(img,  'gray')
	plt.waitforbuttonpress()

def main():
	print ('Hello world!')

	digits, images = create_test10() # get grayscale images with their labels
	inputs = prepare_for_ann(images) # 
	outputs =  sredi_izlaz(digits)

	#ann = create_ann()
	#ann = train_ann(ann, inputs, outputs)
	#print ('Gotovo obucavanje!')
	#pickle.dump( ann, open( "saveANN.p", "wb" ) )

	#ann = pickle.load( open( "saveANN.p", "rb" ) )
	#recognize_digit(ann, 'test_set/dva_3.bmp')


	digits1000, images1000 = create_train_hundred()
	inputs1000 = prepare_for_ann(images1000)
	outputs1000 = sredi_izlaz_hundred(digits1000)
	#ann1000 = create_ann()
	#print ('Obucavanje...')
	#ann1000 = train_ann(ann1000, inputs1000, outputs1000)
	#print('Gotovo obucavanje')
	#pickle.dump( ann1000, open("saveANN1000.p", "wb"))
	ann1000 = pickle.load( open("saveANN1000.p", "rb") )
	#recognize_digit(ann1000, 'test_set/dva_3.bmp')


	#img = load_image('images/img1.png')
	#image_resize(img)

	nesto('images/img1.png')

	print ('Kraj')


if __name__ == "__main__":
	main()
