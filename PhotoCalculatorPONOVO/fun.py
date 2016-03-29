import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import collections

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

def sredi_izlaz(outputs):
    izlaz = np.zeros((100,10))
    for i in range(0,10):
        izlaz[i][0] = 1
    for i in range(10,20):
        izlaz[i][1] = 1  
    for i in range(20,30):
        izlaz[i][2] = 1
    for i in range(30,40):
        izlaz[i][3] = 1
    for i in range(40, 50):
        izlaz[i][4] = 1
    for i in range(50, 60):
        izlaz[i][5] = 1
    for i in range(60, 70):
        izlaz[i][6] = 1
    for i in range(70, 80):
        izlaz[i][7] = 1
    for i in range(80, 90):
        izlaz[i][8] = 1    
    for i in range(90, 100):
        izlaz[i][9] = 1

    return izlaz


def sredi_izlaz_hundred(outputs):
    izlaz = np.zeros((1000,10))
    for i in range(0,100):
        izlaz[i][0] = 1
    for i in range(100,200):
        izlaz[i][1] = 1  
    for i in range(200,300):
        izlaz[i][2] = 1
    for i in range(300,400):
        izlaz[i][3] = 1
    for i in range(400, 500):
        izlaz[i][4] = 1
    for i in range(500, 600):
        izlaz[i][5] = 1
    for i in range(600, 700):
        izlaz[i][6] = 1
    for i in range(700, 800):
        izlaz[i][7] = 1
    for i in range(800, 900):
        izlaz[i][8] = 1    
    for i in range(900, 1000):
        izlaz[i][9] = 1

    return izlaz

def resize_test(region):
    resized = cv2.resize(region,(region.shape[1],region.shape[0]), interpolation = cv2.INTER_NEAREST)
    return resized

def konture(image_orig, image_bin):
    
    contours, hierarchy = cv2.findContours(image_bin.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_regions = []
    regions_dic = {}
    i = 0
    
    for contour in contours:
        if(hierarchy[0][i][3]==0):
            x,y,w,h = cv2.boundingRect(contour)
            region = image_bin[y:y+h+1,x:x+w+1]
            #regions_dic[x] = resize_region(region)
            #regions_dic[x] = resize_test(region)  
            regions_dic[x] = region     
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
        i=i+1
    
    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions

    return image_orig, sorted_regions 


def konture_testtest(image_orig, image_bin):
    
    contours, hierarchy = cv2.findContours(image_bin.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_regions = []
    regions_dic = {}
    i = 0

    for contour in contours:
        if(hierarchy[0][i][3]==0 or hierarchy[0][i][3]==-1):
            x,y,w,h = cv2.boundingRect(contour)
            region = image_bin[y:y+h+1,x:x+w+1]
            #regions_dic[x] = resize_region(region)
            #regions_dic[x] = resize_test(region)  
            regions_dic[x] = region     
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
        i=i+1
    
    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions

    return image_orig, sorted_regions     

def create_ann():
    
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
   
    X_train = np.array(X_train, np.float32) 
    y_train = np.array(y_train, np.float32) 
   
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(X_train, y_train, nb_epoch=2000, batch_size=1, verbose = 1, shuffle=False, show_accuracy = False) 
      
    return ann


def display_result(outputs, alphabet):

    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result