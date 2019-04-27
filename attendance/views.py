from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import os
import time
import pickle
import sys
from django.shortcuts import render

# Create your views here.
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from string import digits

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=96)
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
def get_dataset(paths, has_class_directories=True):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())

# GRADED FUNCTION: triplet_loss

def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
    return loss


# In[39]:

with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))

print("Training wait for some minutes.......")
#FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
#load_weights_from_FaceNet(FRmodel)


database = {}
train_img="./Preprocessed Images"
	
dataset = get_dataset(train_img)
for cls in dataset:
	for image_path in cls.image_paths:
		
		filename = os.path.splitext(os.path.split(image_path)[1])[0]
		print(filename)
		temp_path=image_path[2:]
		
		database[filename] = img_to_encoding(temp_path, FRmodel)



def who_is_it_min(image_path, database, model):
    """
   
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    remove_digits = str.maketrans('', '', digits)
    result_name = identity.translate(remove_digits)    


    #os.remove("temp.jpg")      
    return min_dist, result_name




def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    dic={}
    dic_co={}
    for (name, db_enc) in database.items():
    	remove_digits = str.maketrans('', '', digits)
    	r_name = name.translate(remove_digits)
    	dic[r_name]=0
    	dic_co[r_name]=0
	    	 
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
    	dist = np.linalg.norm(encoding-database[name])
    	remove_digits = str.maketrans('', '', digits)
    	r_name = name.translate(remove_digits)
    	dic[r_name]=dic[r_name]+dist
    	dic_co[r_name]=dic_co[r_name]+1  
    dic_mean={}	

    for (name,dist) in dic.items():
    	dic_mean[name]=dic[name]/dic_co[name]
    
    
    result_name=min(dic_mean, key=lambda k: dic_mean[k])
    min_dist = dic_mean[result_name]
    	
    
    
    
         
    return min_dist, result_name
def record(ipc):
	cap = cv2.VideoCapture(ipc)
	ret, frame = cap.read()
	fshape = frame.shape
	fheight = fshape[0]
	fwidth = fshape[1]
	 

	# Define the codec and create VideoWriter object 
 
	fourcc = cv2.VideoWriter_fourcc(*'MPEG')
	out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (fwidth,fheight))
	now = time.time()
	future = now + 10
	# loop runs if capturing has been initialized. 
	while(True): 


		# Converts to HSV color space, OCV reads colors as BGR 
		# frame is converted to hsv 
		#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

		# output the frame 
		out.write(frame) 

		# reads frames from a camera 
		# ret checks return at each frame 
		ret, frame = cap.read() 
		

		# The window showing the operated video stream 
		#cv2.imshow('frame', hsv) 


		# Wait for 'a' key to stop the program 
		# 	if cv2.waitKey(1) & 0xFF == ord('a'): 
		# 		break
		if time.time() > future:
			break
	cap.release() 

	# After we release our webcam, we also release the output 
	out.release() 




	


def FrameCapture(path): 
      
	# Path to video file 
	vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1
	length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
	print( length )
	images = []
	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 
		# Saves the frames with frame-count 
		if count % 8 == 0 and success == 1:
			images.append(image)

		count += 1
	return images	

def attendance(request):
	h_ip=request.get_host()
	h_ip = h_ip[:-5]
	print(h_ip)
	
	
	try:	
		client_address = request.META['HTTP_X_FORWARDED_FOR']
	except:
		client_address = request.META['REMOTE_ADDR']
	
	ip_c="http://"+client_address+":8080/video"
	print(ip_c)
	
	if(client_address==h_ip):
		ip_c=0
	
	if not database:
		request.session['mes']="No Face in Database"
		return redirect('alert')
	record(ip_c)
	images=FrameCapture("output1.avi")
	
	names=[]
	flag=0
	hs=np.empty(0)
	for img in images:

	
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(img_gray)
		no_faces=len(faces)
		id_faces=0
		
		name_array=[]
		dist_array=[]
		if len(faces) >= 1:
			flag=1
			count=1
			for f in faces:

				face = f
				(x, y, w, h) = face_utils.rect_to_bb(face)
				face_img = img_gray[y-50:y + h+100, x-50:x + w+100]
				face_aligned = face_aligner.align(img, img_gray, face)

				face_img = face_aligned
				cv2.imwrite("temp"+str(count)+".jpg", face_img)
				
		
				min_dist, identity=who_is_it_min("temp"+str(count)+".jpg", database, FRmodel)
				dist_array.append(min_dist)
				name_array.append(identity)
				count=count+1
		else:
			dump=1
		
			
		
		
		
		
		for i in range (no_faces):
			font = cv2.FONT_HERSHEY_SIMPLEX

			fm=cv2.imread("temp"+str(i+1)+".jpg")
			fm= cv2.resize(fm,(int(400),int(400)))
			if dist_array[i] > 0.50:
				name_array[i]="Can't Identify"
			else:
				if name_array[i] not in names:
					names.append(name_array[i])
					cv2.putText(fm,name_array[i],(100,40), font, 1,(255,255,255),2)
					if (len(names)==1):
						hs=fm
					else:
						hs=np.hstack((hs,fm))
			
			

		
			
		
			


	cv2.imwrite("result1.jpg",hs)       
	request.session['uname'] = names
	
	request.session['idf'] = len(names)
	
		

	
	return render(request, 'att.html')		




                
        
