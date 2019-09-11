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
USER_IMAGES_FOLDER = "D:\tmp"
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
    ##Digit_recognizer.logger.info("the image file in files has type of : " + str(type(image_file)))
    img_str_data= load_image(image_file)
    precoessed_img, original_img = pre.recieve_image(img_str_data, 't_7_1.png')
    recognizer = pre.DigitRecognizer()
    recognizer.Load_Model(os.path.join(RECOGNIZER_DIR, MODEL_DIR), MOEDEL_NAME)
    predicted_label = recognizer.Predict_Label(precoessed_img)
    message = "The digit is recognized as : " + str(predicted_label) + ".\n"
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

def load_image(image_file):
#    if image_file.filename() == '':
#        flash('Missing image file to be recognized.')
#        return redirect(request.url)
#if image_file and allowed_file(image_file.filename):
#    filename = secure_filename(image_file.filename)
#    file_path = os.path.join(USER_IMAGES_FOLDER, filename)
#    image_file.save(file_path)

#    image_str_data = image_file.read()
#    return image_str_data, filename
    img_str_data = image_file.read()
    return img_str_data
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    if not os.path.exists(USER_IMAGES_FOLDER):
        os.mkdir(USER_IMAGES_FOLDER)


if __name__ == "__main__":
    Digit_recognizer.run(debug = True)