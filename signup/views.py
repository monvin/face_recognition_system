from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
import cv2
import numpy as numpy
import os, time
import tensorflow as tf
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
# Create your views here.
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from pathlib import Path


def training():
	Path('face_recognition/views.py').touch()


























detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=96)

FACE_DIR = "Preprocessed Images/"
FACE_DIR2= "notalligned/"

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(username,ipc):
    create_folder(FACE_DIR)
    while True:
        name=username
        
        try:
            
            face_folder = FACE_DIR + str(name) + "/"
            face_folder2= FACE_DIR2 + str(name) + "/"
            create_folder(face_folder)
            create_folder(face_folder2)
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
    cap = cv2.VideoCapture(ipc)
    total_imgs = 5
    while True:
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        if len(faces) == 1:
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = img_gray[y-50:y + h+100, x-50:x + w+100]
            face_aligned = face_aligner.align(img, img_gray, face)

            face_img = face_aligned
            img_path = face_folder +name+ str(img_no) + ".jpg"
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.imshow("aligned", face_img)
            img_no += 1

        cv2.imshow("Saving", img)
        cv2.waitKey(1)
        if img_no == init_img_no + total_imgs:
            break

    cap.release()
    cv2.destroyAllWindows()

def signup(request):

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
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
		    form.save()
		    username = form.cleaned_data.get('username')
		    raw_password = form.cleaned_data.get('password1')
		    main(username,ip_c)
		    training()
		    
		            

		    user = authenticate(username=username, password=raw_password)
		    login(request, user)
		    return redirect('home')
	else:
		form = UserCreationForm()
	return render(request, 'signup.html', {'form': form})
