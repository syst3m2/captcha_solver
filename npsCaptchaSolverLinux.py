# Environment CY3650

from selenium import webdriver
import pandas as pd
#from google.cloud import vision
#from google.cloud import storage
#from google.cloud import automl
#from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import os
#import shutil
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import datetime
import urllib
import time
from selenium.common.exceptions import NoSuchElementException
import tensorflow as tf
import base64
import io
import json
import requests

def npsLogin(driver, username, password):
    nps_username = username
    nps_password = password
    driver.get("https://exapps.nps.edu/StudentMuster/index.aspx")
    url = driver.current_url

    if "https://cas.nps.edu/NPS/login?" in url:
        driver.find_element_by_id("username").send_keys(nps_username)
        driver.find_element_by_id ("password").send_keys(nps_password)
        driver.find_element_by_name("submit").click()
        return True
    else:
        print("User already logged in")
        return False

def npsGrabCaptcha(driver, folder, filename):
    driver.get("https://exapps.nps.edu/StudentMuster/index.aspx")
    
    filepath = folder + filename
    image = driver.find_element_by_id("ctl00_ContentPlaceHolder1_ImageControlCaptcha1_Image1").screenshot(filepath)
    im = Image.open(filepath) 

    # Setting the points for cropped image 
    left = 1
    top = 1
    right = 134
    bottom = 39

    # Cropped image of above dimension 
    # (It will not change orginal image) 
    crop_im = im.crop((left, top, right, bottom))
    #crop_im.save(filepath)

    # Convert PIL format to opencv
    pil_image = crop_im.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    cv2.imwrite(filepath, open_cv_image)

    return open_cv_image

def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv

	return im_out


def npsPrepCaptcha(src_image, folder, filename):

    # return im_out1, im_out2, im_out3, im_out4, im_out5, im_out6
    # if bad, return could not process
    # Load the image and convert it to grayscale
    #filepath = img_filepath
    #image = cv2.imread(filepath)
    image = src_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    folder = folder
    filename = filename
    # applying threshold
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

    thresh = cv2.bitwise_not(thresh)

    original = thresh
    thresh = fillHole(thresh)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #draw_cont = cv2.drawContours(thresh.copy(), contours, -1, (0,255,0),3)

    contour_list = []
    for contour in contours:

        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        if w > 5 and h > 5:
            contour_list.append((x,y,w,h))

    contour_list = sorted(contour_list, key=lambda x: x[0])

    counter = 0
    dim = (50,50)
    img_list = []
    imgpath_list = []
    for contour in contour_list:
        (x,y,w,h) = contour

        letter_image = original[y:y + h, x:x + w]

        # Resize to 50, 50 to prep for the model, save both versions
        resized_letter_image = cv2.resize(letter_image, dim, interpolation = cv2.INTER_AREA)

        #original_im_filepath = folder + "grayscaleCaptchas/"
        resized_im_filepath = folder + "resizedGrayscaleCaptchas/"

        #if not os.path.isdir(original_im_filepath):
            #os.mkdir(original_im_filepath)

        if not os.path.isdir(resized_im_filepath):
            os.mkdir(resized_im_filepath)

        #filename = contour_list[counter]

        #original_im_file = original_im_filepath + str(counter) + "_grayscale_" + filename
        resized_im_file = resized_im_filepath + str(counter) + "_resizedgrayscale_" + filename

        #cv2.imwrite(original_im_file, letter_image)
        cv2.imwrite(resized_im_file, resized_letter_image)
        img_list.append(resized_letter_image)
        imgpath_list.append(resized_im_file)
        counter += 1
    
    if len(img_list) == 6:
        return img_list, imgpath_list
    else:
        return False, False

'''
def npsPredictCaptchaAPI(image_list):

    # TODO(developer): Uncomment and set the following variables
    project_id = 'captcha-solver-303801'
    model_id = 'ICN5211344267053629440'
    file_path = "captcha-solver-b544f1f10c18.json"

    # Set environment variable to authenticate with Google Cloud
    credential_path = file_path
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    prediction_client = automl.PredictionServiceClient()

    # Get the full path of the model.
    model_full_id = automl.AutoMlClient.model_path(
        project_id, "us-central1", model_id
    )

    # Read the file.
    captcha_solution = ''

    #grayscale_list = os.listdir(resized_im_filepath)

    for letter in image_list:
        
        #graypath = pre_im_filepath + grayfile
        #print(graypath)

        #with open(graypath, "rb") as content_file:
            #content = content_file.read()

        #print("Content is this many long: ", len(content))

        letter_str = cv2.imencode('.png', letter)[1].tostring()

        image = automl.Image(image_bytes=letter_str)
        payload = automl.ExamplePayload(image=image)

        # params is additional domain-specific parameters.
        # score_threshold is used to filter the result
        # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#predictrequest
        params = {"score_threshold": "0.3"}

        request = automl.PredictRequest(
            name=model_full_id,
            payload=payload,
            params=params
        )
        response = prediction_client.predict(request=request)

        try: 
            captcha_solution += str(response.payload[0].display_name)
        except IndexError:
            captcha_solution += ''
    
    if len(captcha_solution) == 6:
        return captcha_solution
    else:
        return False
'''

def npsPredictCaptchaLocal(imagepath_list, image_key='1', port_number='8502'):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_file_path: Path to a local image for the prediction request.
        image_key: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
    """
    captcha_solution = ''   
    for path in imagepath_list:
        with io.open(path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # The example here only shows prediction with one image. You can extend it
        # to predict with a batch of images indicated by different keys, which can
        # make sure that the responses corresponding to the given image.
        instances = {
                'instances': [
                        {'image_bytes': {'b64': str(encoded_image)},
                        'key': image_key}
                ]
        }

        # This example shows sending requests in the same server that you start
        # docker containers. If you would like to send requests to other servers,
        # please change localhost to IP of other servers.
        url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)

        response = requests.post(url, data=json.dumps(instances))
        response = response.json()

        scores = response['predictions'][0]['scores']
        labels = response['predictions'][0]['labels']
        response_dict = {"Scores": scores, "Labels": labels}

        response_df = pd.DataFrame(response_dict)

        response_df = response_df.sort_values(by=['Scores'], ascending=False, ignore_index=True)

        predicted_label = response_df['Labels'][0]

        captcha_solution += predicted_label

    if len(captcha_solution) == 6:
        return captcha_solution
    else:
        return False


def npsSubmitCaptcha(driver, prediction):
    driver.find_element_by_id("ctl00_ContentPlaceHolder1_StudentGuess").send_keys(prediction)
    driver.find_element_by_id("ctl00_ContentPlaceHolder1_Button1").click()
    try:
        result = driver.find_element_by_class_name('alert-success')
        if result:
            return True
    except NoSuchElementException:
        result = driver.find_element_by_class_name('alert-danger')
        if result:
            return False 

def main():
    # Version with functions

    nps_username = "nicholas.villemez"
    nps_password = "fxn0dL2R!y8WU&MA"

    LOGIN_FLAG = False

    fail_preprocess_counter = 0
    fail_predict_counter = 0
    fail_muster_counter = 0
    attempt_counter = 1

    base_folder = "Captchas/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)

    while True: 

        # Login to NPS website
        driver = webdriver.Chrome()
        login = npsLogin(driver = driver, username = nps_username, password = nps_password)

        # Create filepath for images
        date = datetime.datetime.now()
        full_date = date.strftime("%Y-%m-%d-%H%M%S")
        folder1 = base_folder + date.strftime("%Y-%m-%d") + "/"
        folder = folder1 + full_date + "_Captcha/"
        filename = "Captcha_" + full_date + ".png"
        filepath = folder + filename

        if not os.path.isdir(folder1):
            os.mkdir(folder1)
            print(folder1 + " has been created")

        if not os.path.isdir(folder):
            os.mkdir(folder)
            print(folder + " has been created")


        # Grab the captcha image from the muster page
        captcha_image = npsGrabCaptcha(driver = driver, folder = folder, filename = filename)

        # Format the captcha, split into letters, grayscale, and resize
        image_list, imagepath_list = npsPrepCaptcha(src_image = captcha_image, folder = folder, filename = filename)

        if image_list == False:
            print("Some letters could not be segmented or interpreted during preprocessing")
            driver.close()
            attempt_counter += 1
            continue

        # Perform prediction on the preprocessed captcha image, returns False if prediction cannot be made

        # This function uses the model deployed on the Google API
        #prediction = npsPredictCaptchaAPI(image_list)

        # This function uses a local model running in a docker container
        prediction = npsPredictCaptchaLocal(imagepath_list)

        if prediction == False:
            print("Predictions could not be made for some letters")
            driver.close()
            attempt_counter += 1
            print("Prediction: ", prediction)
            continue
        else:
            print("Prediction: ", prediction)

        # Check if prediction correct. Else, try again
        result = npsSubmitCaptcha(driver, prediction)

        if result:
            print("Muster Successful!")
            driver.close()
            print(str(attempt_counter) + " Attempts performed before success")
            break
        else:
            driver.close()
            attempt_counter += 1
            continue

if __name__ == "__main__":
    main()