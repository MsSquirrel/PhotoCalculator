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

    digit_0 = ['0']*100
    digit_1 = ['1']*100
    digit_2 = ['2']*100
    digit_3 = ['3']*100
    digit_4 = ['4']*100
    digit_5 = ['5']*100
    digit_6 = ['6']*100
    digit_7 = ['7']*100
    digit_8 = ['8']*100
    digit_9 = ['9']*100
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


    with open('digit_separated/digit_0/train100.txt') as f:
        content_0 = f.readlines()
    with open('digit_separated/digit_1/train100.txt') as f:
        content_1 = f.readlines()  
    with open('digit_separated/digit_2/train100.txt') as f:
        content_2 = f.readlines()   
    with open('digit_separated/digit_3/train100.txt') as f:
        content_3 = f.readlines()
    with open('digit_separated/digit_4/train100.txt') as f:
        content_4 = f.readlines()
    with open('digit_separated/digit_5/train100.txt') as f:
        content_5 = f.readlines()
    with open('digit_separated/digit_6/train100.txt') as f:
        content_6 = f.readlines()    
    with open('digit_separated/digit_7/train100.txt') as f:
        content_7 = f.readlines()
    with open('digit_separated/digit_8/train100.txt') as f:
        content_8 = f.readlines()
    with open('digit_separated/digit_9/train100.txt') as f:
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



