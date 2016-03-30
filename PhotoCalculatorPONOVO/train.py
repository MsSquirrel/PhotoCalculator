from fun import *

def create_test10():

    digit_0 = ['0']*10
    digit_1 = ['1']*10
    digit_2 = ['2']*10
    digit_3 = ['3']*10
    digit_4 = ['4']*10
    digit_5 = ['5']*10
    digit_6 = ['6']*10
    digit_7 = ['7']*10
    digit_8 = ['8']*10
    digit_9 = ['9']*10
    digits = digit_0 + digit_1 + digit_2+ digit_3+digit_4+digit_5+digit_6+digit_7+digit_8+digit_9

    images_0 = []
    images_1 = []
    images_2 = []
    images_3 = []
    images_4 = []
    images_5 = []
    images_6 = []
    images_7 = []
    images_8 = []
    images_9 = []

    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_0/nula'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_0.append(slicica)

        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_1/jedan'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_1.append(slicica)    

    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_2/dva'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_2.append(slicica) 
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_3/tri'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_3.append(slicica) 
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_4/cetiri'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_4.append(slicica) 
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_5/pet'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_5.append(slicica) 
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_6/sest'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_6.append(slicica) 
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_7/sedam'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_7.append(slicica)     
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_8/osam'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_8.append(slicica) 
        
    for i in range(0,10):
        img_suf = '_'+str(i)
        naziv = 'cifre/digit_9/devet'+img_suf+'.bmp'
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_9.append(slicica) 
        
    images = images_0 + images_1 + images_2 + images_3 + images_4 + images_5 + images_6 + images_7 + images_8 + images_9

    return digits, images

def create_train_hundred():

    digit_0 = ['0']*1000
    digit_1 = ['1']*1000
    digit_2 = ['2']*1000
    digit_3 = ['3']*1000
    digit_4 = ['4']*1000
    digit_5 = ['5']*1000
    digit_6 = ['6']*1000
    digit_7 = ['7']*1000
    digit_8 = ['8']*1000
    digit_9 = ['9']*1000

    digits = digit_0 + digit_1 + digit_2+ digit_3+digit_4+digit_5+digit_6+digit_7+digit_8+digit_9

    images_0 = []
    images_1 = []
    images_2 = []
    images_3 = []
    images_4 = []
    images_5 = []
    images_6 = []
    images_7 = []
    images_8 = []
    images_9 = []


    with open('digit_separated/digit_0/train1000.txt') as f:
        content_0 = f.readlines()
    with open('digit_separated/digit_1/train1000.txt') as f:
        content_1 = f.readlines()  
    with open('digit_separated/digit_2/train1000.txt') as f:
        content_2 = f.readlines()   
    with open('digit_separated/digit_3/train1000.txt') as f:
        content_3 = f.readlines()
    with open('digit_separated/digit_4/train1000.txt') as f:
        content_4 = f.readlines()
    with open('digit_separated/digit_5/train1000.txt') as f:
        content_5 = f.readlines()
    with open('digit_separated/digit_6/train1000.txt') as f:
        content_6 = f.readlines()    
    with open('digit_separated/digit_7/train1000.txt') as f:
        content_7 = f.readlines()
    with open('digit_separated/digit_8/train1000.txt') as f:
        content_8 = f.readlines()
    with open('digit_separated/digit_9/train1000.txt') as f:
        content_9 = f.readlines()



    for i in range(0,100):
        naziv = 'digit_separated/digit_0/' + content_0[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_0.append(slicica)

    for i in range(0,100):
        naziv = 'digit_separated/digit_1/' + content_1[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_1.append(slicica) 
    for i in range(0,100):
        naziv = 'digit_separated/digit_2/' + content_2[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_2.append(slicica) 
    for i in range(0,100):
        naziv = 'digit_separated/digit_3/' + content_3[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_3.append(slicica) 
    for i in range(0,100):
        naziv = 'digit_separated/digit_4/' + content_4[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_4.append(slicica)  
    for i in range(0,100):
        naziv = 'digit_separated/digit_5/' + content_5[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_5.append(slicica)
    for i in range(0,100):
        naziv = 'digit_separated/digit_6/' + content_6[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_6.append(slicica) 
    for i in range(0,100):
        naziv = 'digit_separated/digit_7/' + content_7[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_7.append(slicica) 
    for i in range(0,100):
        naziv = 'digit_separated/digit_8/' + content_8[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_8.append(slicica) 
    for i in range(0,100):
        naziv = 'digit_separated/digit_9/' + content_9[i][:-1]
        slicica = image_bin(image_gray(load_image(naziv)))
        slicica = resize_region(slicica)
        images_9.append(slicica)     

    images = images_0 + images_1 + images_2 + images_3 + images_4 + images_5 + images_6 + images_7 + images_8 + images_9

    return digits, images


def img_resize(img):

    width = img.shape[1]
    height = img.shape[0]

    new_width = 28.0
    new_height =28.0

    if width>height:
        max_dim = width
    else:
        max_dim = height

    new_ratio = new_width / max_dim
    img = cv2.resize(img, (0,0), fx = new_ratio, fy = new_ratio, interpolation = cv2.INTER_CUBIC)
    img = image_bin(img)

    new_img = np.ones((28,28))*0
    
    if (img.shape[1]==28):
        d = (new_img.shape[0] - img.shape[0])/2
        for i in range(0, img.shape[0]):
            for j in range (0, img.shape[1]):
                new_img[d+i][j] = img[i][j]
    else:
        d = (new_img.shape[1] - img.shape[1])/2
        for i in range(0, img.shape[0]):
            for j in range (0, img.shape[1]):
                new_img[i][d+j] = img[i][j] 

    return new_img


def img_preproc_train(path):
    naziv = path
    sl = load_image(naziv)
    slicica = invert(image_bin(image_gray(sl)))
    sel_reg, kont = konture_testtest(sl.copy(), slicica)
    img = kont[0]
    img = img_resize(img)
    return img

def img_pre_img(img):
    slicica = invert(image_bin(image_gray(img)))
    sel_reg, kont = konture_testtest(img.copy(), slicica)
    images = np.zeros((len(kont), 28, 28))

    #plt.imshow(kont[0], 'gray')
    #plt.waitforbuttonpress()

    i=0
    for k in kont:
        images[i] = img_resize(k)
        i = i+1

    return images

def img_preproc(path):
    naziv = path
    sl = load_image(naziv)
    slicica = invert(image_bin(image_gray(sl)))
    sel_reg, kont = konture(sl.copy(), slicica)
    img = kont[0]
    img = img_resize(img)
    return img



def create_train_hundred_test():

    digit_0 = ['0']*1000
    digit_1 = ['1']*1000
    digit_2 = ['2']*1000
    digit_3 = ['3']*1000
    digit_4 = ['4']*1000
    digit_5 = ['5']*1000
    digit_6 = ['6']*1000
    digit_7 = ['7']*1000
    digit_8 = ['8']*1000
    digit_9 = ['9']*1000
    digits = digit_0 + digit_1 + digit_2+ digit_3+digit_4+digit_5+digit_6+digit_7+digit_8+digit_9

    images_0 = []
    images_1 = []
    images_2 = []
    images_3 = []
    images_4 = []
    images_5 = []
    images_6 = []
    images_7 = []
    images_8 = []
    images_9 = []

    avg_0 = np.zeros((28,28))
    avg_1 = np.zeros((28,28))
    avg_2 = np.zeros((28,28))
    avg_3 = np.zeros((28,28))
    avg_4 = np.zeros((28,28))
    avg_5 = np.zeros((28,28))
    avg_6 = np.zeros((28,28))
    avg_7 = np.zeros((28,28))
    avg_8 = np.zeros((28,28))
    avg_9 = np.zeros((28,28))
    avg = np.zeros((10,28,28))

    with open('digit_separated/digit_0/train1000.txt') as f:
        content_0 = f.readlines()
    with open('digit_separated/digit_1/train1000.txt') as f:
        content_1 = f.readlines()  
    with open('digit_separated/digit_2/train1000.txt') as f:
        content_2 = f.readlines()   
    with open('digit_separated/digit_3/train1000.txt') as f:
        content_3 = f.readlines()
    with open('digit_separated/digit_4/train1000.txt') as f:
        content_4 = f.readlines()
    with open('digit_separated/digit_5/train1000.txt') as f:
        content_5 = f.readlines()
    with open('digit_separated/digit_6/train1000.txt') as f:
        content_6 = f.readlines()    
    with open('digit_separated/digit_7/train1000.txt') as f:
        content_7 = f.readlines()
    with open('digit_separated/digit_8/train1000.txt') as f:
        content_8 = f.readlines()
    with open('digit_separated/digit_9/train1000.txt') as f:
        content_9 = f.readlines()



    for i in range(0,1000):
        naziv = 'digit_separated/digit_0/' + content_0[i][:-1]
        slicica = img_preproc_train(naziv)
        images_0.append(slicica)
        avg_0 = add_matrices(avg_0, slicica)

    avg[0] = avg_value(avg_0, 1000.0)

    for i in range(0,1000):
        naziv = 'digit_separated/digit_1/' + content_1[i][:-1]
        slicica = img_preproc_train(naziv)
        images_1.append(slicica)
        avg_1 = add_matrices(avg_1, slicica)

    avg[1] = avg_value(avg_1, 1000.0)

    for i in range(0,1000):
        naziv = 'digit_separated/digit_2/' + content_2[i][:-1]
        slicica = img_preproc_train(naziv)
        images_2.append(slicica) 
        avg_2 = add_matrices(avg_2, slicica)

    avg[2] = avg_value(avg_2, 1000.0) 

    for i in range(0,1000):
        naziv = 'digit_separated/digit_3/' + content_3[i][:-1]
        slicica = img_preproc_train(naziv)
        images_3.append(slicica) 
        avg_3 = add_matrices(avg_3, slicica)
        
    avg[3] = avg_value(avg_3, 1000.0) 

    for i in range(0,1000):
        naziv = 'digit_separated/digit_4/' + content_4[i][:-1]
        slicica = img_preproc_train(naziv)
        images_4.append(slicica)  
        avg_4 = add_matrices(avg_4, slicica)
        
    avg[4] = avg_value(avg_4, 1000.0) 

    for i in range(0,1000):
        naziv = 'digit_separated/digit_5/' + content_5[i][:-1]
        slicica = img_preproc_train(naziv)
        images_5.append(slicica)
        avg_5 = add_matrices(avg_5, slicica)
        
    avg[5] = avg_value(avg_5, 1000.0) 

    for i in range(0,1000):
        naziv = 'digit_separated/digit_6/' + content_6[i][:-1]
        slicica = img_preproc_train(naziv)
        images_6.append(slicica)
        avg_6 = add_matrices(avg_6, slicica)
        
    avg[6] = avg_value(avg_6, 1000.0)  

    for i in range(0,1000):
        naziv = 'digit_separated/digit_7/' + content_7[i][:-1]
        slicica = img_preproc_train(naziv)
        images_7.append(slicica)
        avg_7 = add_matrices(avg_7, slicica)
        
    avg[7] = avg_value(avg_7, 1000.0)  

    for i in range(0,1000):
        naziv = 'digit_separated/digit_8/' + content_8[i][:-1]
        slicica = img_preproc_train(naziv)
        images_8.append(slicica) 
        avg_8 = add_matrices(avg_8, slicica)
        
    avg[8] = avg_value(avg_8, 1000.0) 

    for i in range(0,1000):
        naziv = 'digit_separated/digit_9/' + content_9[i][:-1]
        slicica = img_preproc_train(naziv)
        images_9.append(slicica)  
        avg_9 = add_matrices(avg_9, slicica)
        
    avg[9] = avg_value(avg_9, 1000.0)    

    images = images_0 + images_1 + images_2 + images_3 + images_4 + images_5 + images_6 + images_7 + images_8 + images_9

    return digits, images, avg


def add_matrices(A,B):

    Z = np.zeros((len(A), len(A[0])))
    for i in range(len(A)):
        for j in range(len(A[i])):
            Z[i][j] = A[i][j] + B[i][j]

    return Z


def avg_value(A, k):
    Z = np.zeros((len(A), len(A[0])))
    for i in range(len(A)):
        for j in range(len(A[i])):
            Z[i][j] = A[i][j] / k

    return Z

