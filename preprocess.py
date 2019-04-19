from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as numpy
import os, time
import tensorflow as tf
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
# Create your views here.

from pathlib import Path


def training():
	Path('face_recognition/views.py').touch()


























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


FACE_DIR = "Preprocessed Images/"


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(username):
    create_folder(FACE_DIR)
    while True:
        name=username
        
        try:
            
            face_folder = FACE_DIR + str(name) + "/"
            create_folder(face_folder)
            break
        except:
            
            continue

    # get beginning image number
    while True:
        DIR=face_folder
        init_img_no = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])+1
        try:
            init_img_no = int(init_img_no)
            break
        except:
            
            continue

    img_no = init_img_no
    cap = cv2.VideoCapture(0)
    total_imgs = 10
    while True:
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        if len(faces) == 1:
            face = faces[0]
            
            
            face_aligned = face_aligner.align(img, img_gray, face)

            face_img = face_aligned
            img_path = face_folder +name+ str(img_no) + ".jpg"
            cv2.imwrite(img_path, face_img)
            
            cv2.imshow("aligned", face_img)
            img_no += 1


train_img="./notalligned"
	
dataset = get_dataset(train_img)
for cls in dataset:
	name=cls.name
	print(cls.name)
	

	face_folder = FACE_DIR + str(cls.name) + "/"
	create_folder(face_folder)
	
	for image_path in cls.image_paths:
		
		
		filename = os.path.splitext(os.path.split(image_path)[1])[0]
		print(filename)
		temp_path=image_path[2:]
		
		DIR=face_folder
		init_img_no = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])+1
		
		init_img_no = int(init_img_no)
		
		img_no = init_img_no
		timg=cv2.imread(temp_path)
		img_gray = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
		faces=detector(img_gray)
		if len(faces) == 1:
			face = faces[0]
            
            
			face_aligned = face_aligner.align(timg, img_gray, face)
			face_img = face_aligned
			img_path = face_folder +name+ str(img_no) + ".jpg"
			cv2.imwrite(img_path, face_img)
            
			cv2.imshow("aligned", face_img)
			img_no += 1
		os.remove(temp_path)
		

