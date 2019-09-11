#  This script is adopted from the example on flask tutorial page: 
#  https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/

import sys
import os     
from flask import Flask, request, flash, make_response, redirect, url_for
from werkzeug.utils import secure_filename
from DigitRecognizer import Prediction as pre

RECOGNIZER_DIR = "DigitRecognizer"
MODEL_DIR = "trained_models"
MOEDEL_NAME = "deepCNN.ckpt"
DATA_FOLDER = "data"
IMAGES_FOLDER = "user_images"
#USER_IMAGES_FOLDER = os.path.join( DATA_FOLDER , IMAGES_FOLDER)
USER_IMAGES_FOLDER = "D:\\tmp"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


Digit_recognizer = Flask(__name__)
Digit_recognizer.config['UPLOAD_FOLDER'] = USER_IMAGES_FOLDER
Digit_recognizer.secret_key = 'super secret'




@Digit_recognizer.route('/digitRecognition/localService/v1.0', methods = ['POST'])
def Get_Digit_Recognized():
    if 'file' not in request.files:
        flash('No file part.')
        return redirect(request.url)
    image_file = request.files['file']
    filename = image_file.filename
    if not file_is_valid(filename):
        return "\n uploaded file invalid! \n"
    
    
    ## Todo: call cassandra module to save infos about input images
    
    img_str_data= load_image(image_file)
    precoessed_img, original_img = pre.recieve_image(img_str_data, 't_7_1.png')
    recognizer = pre.DigitRecognizer()
    recognizer.Load_Model(os.path.join(RECOGNIZER_DIR, MODEL_DIR), MOEDEL_NAME)
    predicted_label = recognizer.Predict_Label(precoessed_img)
    message = "The digit is recognized as : " + str(predicted_label) + ".\n"

    ## Todo call cassandra module to save predicted infos along with input images
    return  message
@Digit_recognizer.route('/test', methods = ['POST'])
def image_test():
    if 'file' not in request.files:
        flash('No file part.')
        return redirect(request.url)
    image_file = request.files['file']
    save = open('image_name.txt', 'w')
    save.write(str(type(image_file)))
    save.close()
    image_file.save('recievd7_1.png')
    return


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## check whether the file type or file name are valid.
def file_is_valid(filename):
    valid = False
    if filename != "":
        if allowed_file(filename):
            valid = True
    return valid
def load_image(image_file):
        image_str_data = image_file.read()
        return image_str_data
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    if not os.path.exists(USER_IMAGES_FOLDER):
        os.mkdir(USER_IMAGES_FOLDER)


if __name__ == "__main__":
    Digit_recognizer.run(debug = True)