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
import wolframalpha

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
    return image_bin
def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((6,6)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((6,6)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

#Funkcionalnost implementirana u V2
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255 # TODO: shoul this be float?
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def konture_print(image_orig, image_bin):
    
    contours, hierarchy = cv2.findContours(image_bin.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_regions = []
    regions_dic = {}
    i = -1
    
    for contour in contours:
    	i+=1
    	if(hierarchy[0][i][3]==0 or hierarchy[0][i][3]==-1):
			x,y,w,h = cv2.boundingRect(contour)
			region = image_bin[y:y+h+1,x:x+w+1]
			#regions_dic[x] = resize_region(region)  
			regions_dic[x] = region  
			cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
		

    
    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions

    return image_orig, sorted_regions 


def create_ann_print():
    
    ann = Sequential()
    ann.add(Dense(1024, input_dim=784, activation='sigmoid'))
    ann.add(Dense(14, activation='sigmoid'))
    return ann

def train_ann_print(ann, X_train, y_train):
   
    X_train = np.array(X_train, np.float32) 
    y_train = np.array(y_train, np.float32) 
   
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(X_train, y_train, nb_epoch=2000, batch_size=1, verbose = 1, shuffle=False, show_accuracy = False) 
      
    return ann


def display_result_print(outputs, alphabet):

    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result



def print_recognition():
	train_img = load_image('print_train/train.jpg')
	train_bin = invert(image_bin(image_gray(train_img)))

	n, kont = konture_print(train_img.copy(), train_bin)
	konture =[]
	i=0
	for k in kont:
		konture.append(img_resize(k))
		i+=1
	
	#plt.imshow(konture[11], 'gray')
	#plt.waitforbuttonpress()

	#alphabet = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/']
	#inputs = prepare_for_ann(konture)
	#outputs = convert_output(alphabet)
	#ann_print = create_ann_print()
	#ann_print = train_ann_print(ann_print, inputs, outputs)
	#pickle.dump( ann_print, open("saveANN_stampano.p", "wb"))
	
	ann_print = pickle.load( open( "saveANN_stampano.p", "rb" ) )
	#recog_con(ann_print, kont[10])
	#validate_print_recognition(ann_print)
	print get_result('2+8*2')


def validate_print_recognition(ann):

	print('Prepoznate cifre')
	res=''
	test_img = load_image('print_test/img2.jpg')
	test_bin = invert(image_bin(image_gray(test_img)))
	
	n, kont = konture_print(test_img.copy(), test_bin)
	konture = []
	for k in kont:
		konture.append(img_resize(k))

	for k in konture:
		res+=recog_con(ann, k)[0]

	print res
	#plt.imshow(konture[0], 'gray')
	#plt.waitforbuttonpress()

	#n, kont = konture_print(train_img.copy(), train_bin)

def recog_con(ann_print, kont):
	alphabet = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/']
	test_img = matrix_to_vector(scale_to_range(kont))
	test_img = test_img.reshape(1, test_img.shape[0])
	#print test_img.shape

	result = ann_print.predict(np.array(test_img, np.float32))
	ret_val = display_result(result, alphabet)
	return ret_val


def get_result(exp):

	app_id = 'HUEHQG-VKXQR866QA'
	client = wolframalpha.Client(app_id)
	res = client.query(exp)
	#for pod in res.pods:
	#   print pod
    
	ret_val = next(res.results).text
	return ret_val