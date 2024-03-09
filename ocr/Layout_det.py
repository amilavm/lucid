# import the necessary packages
import cv2
import pytesseract
import os
import numpy as np
# import glob
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pdf2image import convert_from_path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import pytesseract
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
# pytesseract.pytesseract.tesseract_ocr = r'/home/shane/anaconda3/envs/muve_ocr/lib/python3.8/site-packages/pytesseract/pytesseract.py'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import glob
# import layoutparser as lp



# pip install -U layoutparser
# pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2'
# !pip install layoutparser[ocr]
# !git clone https://github.com/Layout-Parser/layout-parser.git
# for j in img_pdfs:
# from OCR_pipeline.image_generator import pdf_out_2
# from ocr.img_gen import pdf_out_2


def structure_selector(pdf_out_2):
    # print('Starting layout Processing')
    content_list = []
    figure_list = []
    table_list = []
    para_list = []

    # Getting all the images to a list from the pdf out folder
    pdf_name = f"{pdf_out_2}/*.jpg"
    img_pdfs = []
    for i in glob.glob(pdf_name):
        img_pdfs.append(i)
    # print('alll images lst ',img_pdfs)
    # pdf_out = (f'Output_files/{tail}')
    # print(pdf_out)

    model = load_model('ocr/test_07_14.h5')

    for j in img_pdfs:
        # print('the j', j)
        image = cv2.imread(j)
        img = j
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        load = load_img(img, target_size=(299, 299))
        ar = img_to_array(load)
        scale = ar / 255.
        dim = np.expand_dims(scale, axis=0)
        a = np.argmax(model.predict(dim), axis=1)
        if a == 0:
            content_list.append(str(j))

            # save the images for QA
            # head, tail = os.path.split(j)
            # tail = tail.split('.')[0]
            #
            # save_file_name = pdf_out_2 + '/' +'content_' +str(tail) + '.jpg'
            # # save_file_name =(str(j))
            # print(save_file_name)
            #cv2.imwrite(save_file_name, image)

        elif a == 1:
            figure_list.append(str(j))

            # # save the images for QA
            # head, tail = os.path.split(j)
            # tail = tail.split('.')[0]
            #
            # save_file_name = pdf_out_2 + '/' +'figure_' +str(tail) + '.jpg'
            # # save_file_name =(str(j))
            # print(save_file_name)
            #cv2.imwrite(save_file_name, image)

        elif a == 2:
            para_list.append(str(j))

            # # save the images for QA
            # head, tail = os.path.split(j)
            # tail = tail.split('.')[0]
            #
            # save_file_name = pdf_out_2 + '/' + 'para_' + str(tail) + '.jpg'
            # # save_file_name =(str(j))
            # print(save_file_name)
            # # cv2.imwrite(save_file_name, image)

        else:
            table_list.append(str(j))

            # # save the images for QA
            # head, tail = os.path.split(j)
            # tail = tail.split('.')[0]
            #
            # save_file_name = pdf_out_2 + '/' + 'table_' + str(tail) + '.jpg'
            # # save_file_name =(str(j))
            # print(save_file_name)
            # cv2.imwrite(save_file_name, image)
    para_list_f = np.concatenate((content_list, para_list, figure_list))

    # print('pdf_out_2: ', pdf_out_2, '\n')
    # print('table_list: ', table_list, '\n')
    # print('para: ', para_list_f, '\n')

    return table_list, para_list_f, pdf_out_2

