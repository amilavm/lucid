# import the necessary packages
from pytesseract import Output
import cv2
import pytesseract

# pytesseract.pytesseract.tesseract_ocr = r'....../lib/python3.8/site-packages/pytesseract/pytesseract.py'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import imutils

def pdf_out_table(table_list, pdf_out):
    # print('\n tables cellwise DataFrame Creation')
    column_names = ['page_num', 'block_num', 'structure', 'xmin', 'ymin', 'xmax', 'ymax', 'text']
    df_pdf_table = pd.DataFrame(columns=column_names)

    # for word outputs
    df_pdf_word_table = pd.DataFrame(columns=column_names)
    column_names1 = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'structure',
                     'xmin', 'ymin', 'xmax', 'ymax', 'conf', 'text']
    df_pdf_word_table = pd.DataFrame(columns=column_names1)

    for j in table_list:

        image = cv2.imread(j)
        #         img2 = image.copy()
        # (hei,wid,_) = image.shape

        # Grayscale and blur the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Threshold the image
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Retrieve contours
        # contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Create box-list
        box = []
        # Get position (x,y), width and height for every contour
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box.append([x, y, w, h])

        # Create separate lists for all values
        heights = []
        widths = []
        xs = []
        ys = []
        # Store values in lists
        for b in box:
            heights.append(b[3])
            widths.append(b[2])
            xs.append(b[0])
            ys.append(b[1])
        # Retrieve minimum and maximum of lists
        min_height = np.min(heights)
        min_width = np.min(widths)
        min_x = np.min(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)
        max_x = np.max(xs)

        # Retrieve height where y is maximum (edge at bottom, last row of table)
        for b in box:
            if b[1] == max_y:
                max_y_height = b[3]
        # Retrieve width where x is maximum (rightmost edge, last column of table)
        for b in box:
            if b[0] == max_x:
                max_x_width = b[2]

        # Obtain horizontal lines mask
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations=9)

        # Obtain vertical lines mask
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        vertical_mask = cv2.dilate(vertical_mask, vertical_kernel, iterations=9)

        # Bitwise-and masks together
        result = 255 - cv2.bitwise_or(vertical_mask, horizontal_mask)

        # plt.figure(figsize=(16,16))
        # plt.imshow(result, cmap='gray')
        # plt.show()

        img = cv2.imread(j)
        # print(img.shape)
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, thresh_value = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        # #     ret, thresh_value = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)    -- images missing
        # #     ret, thresh_value = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV) --original
        kernel = np.ones((3, 3), np.uint8)
        # #     kernel = np.ones((5,5),np.uint8) --ori

        dilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
        # contours, hierarchy = cv2.findContours(img_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # sorting contours top to bottom
        try:
            (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][1], reverse=False))
        except:
            pass
        # print('dialated_value')
        # plt.figure(figsize=(16,16))
        # plt.imshow(dilated_value)
        # plt.show()

        # doing changes for the img2 to read the OCR
        # Creating a copy of image

        # image manipulation for OCR
        img2 = cv2.imread(j)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        gray2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
        #         thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        blur2 = cv2.medianBlur(gray2, 5)
        dkernel = np.ones((5, 5), np.uint8)
        dialate2 = cv2.dilate(blur2, dkernel, iterations=1)

        block_num = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # bounding the images
            #     if h > 100 and w > 500 and (cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 50000) :
            if (cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 200000) and h < 3000:
                table_image = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
                #             print(cnt)

                # Cropping the text block for giving input to OCR
                cropped = dialate2[y:y + h, x:x + w]

                # ocr detection on the contour detected OCR
                results_para_text = pytesseract.image_to_string(cropped)  # , output_type=Output.DICT)
                results_para_text = str(results_para_text.strip())
                #                 print(results_para_text)
                block_num += 1

                # adding the text from ocr with contour cordinates to a df
                df = pd.DataFrame({"page_num": [str(j)], "block_num": [str(block_num)], 'structure': 'table',
                                   "xmin": [str(x)], "ymin": [str(y)],
                                   "xmax": [str(w + x)], "ymax": [str(h + y)],
                                   "text": results_para_text})

                # concat with the previous df
                df_pdf_table = pd.concat([df_pdf_table, df], axis=0, ignore_index=True, sort=False)

                # get the word output from tables in table mode

                results_word_t = pytesseract.image_to_data(cropped, output_type=Output.DICT)

                df_wt = pd.DataFrame(results_word_t)

                df_wt.rename(columns={'top': 'ymin', 'height': 'ymax', 'left': 'xmin', 'width': 'xmax'}, inplace=True)
                df_wt['ymax'] = df_wt['ymin'] + df_wt['ymax']
                df_wt['xmax'] = df_wt['xmin'] + df_wt['xmax']

                df_wt['page_num'] = str(j)
                df_wt['block_num'] = block_num
                df_wt['structure'] = 'table'

                df_pdf_word_table = pd.concat([df_pdf_word_table, df_wt], axis=0, ignore_index=True, sort=False)

                min_conf = 50

                # loop over each of the individual text localizations
                for i in range(0, len(results_word_t["text"])):

                    (x, y, w, h) = (results_word_t['left'][i], results_word_t['top'][i],
                                    results_word_t['width'][i], results_word_t['height'][i])
                    text = results_word_t["text"][i]

                    conf = float(results_word_t["conf"][i])

                    # filter out weak confidence text localizations
                    if conf > min_conf:
                        # display the confidence and text to our terminal
                        #         print("Confidence: {}".format(conf))
                        #         print("Text: {}".format(text))
                        #         print("")
                        # strip out non-ASCII text so we can draw the text on the image
                        # using OpenCV, then draw a bounding box around the text along
                        # with the text itself
                        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                        cv2.rectangle(cropped, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(cropped, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        #                     plt.figure(figsize=(16,16))
        #                     plt.imshow(cropped)
        #                     plt.show()

        # print(str(j))
    # print('\n tables cellwise DataFrame Creation completed')

    # df_pdf_para.to_csv(f'Output_files/csv_files/table_bb_cords_{pdf_n}.csv', index = False, line_terminator="\n")
    return df_pdf_table, df_pdf_word_table  # , pdf_name_o




def df_clean(df_pdf_table,  df_pdf_word_table, df_pdf_para, df_pdf_word_para, pdf_name):
    # print('\nDataFrame cleaning ')

    Output_df_path = '/output/csv/'

    df_pdf_context = pd.concat([df_pdf_table, df_pdf_para], axis=0, ignore_index=True, sort=False)

    df_pdf_word_para['conf'] = df_pdf_word_para['conf'].astype(float)
    df_pdf_word_para = df_pdf_word_para[df_pdf_word_para['conf']>1.0]

    df_pdf_word_table['conf'] = df_pdf_word_para['conf'].astype(float)
    df_pdf_word_table = df_pdf_word_table[df_pdf_word_table['conf'] > 1.0]

    df_pdf_word = pd.concat([df_pdf_word_table, df_pdf_word_para], axis=0, ignore_index=True, sort=False)

    # from thulitha code
    df_pdf_context.rename(columns={'text': 'text_string'}, inplace=True)
    df_pdf_context.drop('structure', inplace=True, axis=1)

    df_pdf_word.rename(columns={'text': 'text_string'}, inplace=True)
    df_pdf_word.drop('structure', inplace=True, axis=1)


    # df_pdf_context.to_csv(Output_df_path+pdf_name+'_para.csv', index=False)
    # df_pdf_word.to_csv(Output_df_path+pdf_name+'_word.csv', index=False)



    # print('\nDataFrame cleaning Completed')
    # print(""" OCR Completed """)
    return df_pdf_context, df_pdf_word




# df_pdf_table, df_pdf_word_table = pdf_out_table(table_list, pdf_out_2)
# df, dfw = df_clean(df_pdf_table,  df_pdf_word_table, df_pdf_para, df_pdf_word_para, pdf_name_o)
# df_pdf_table, df_pdf_word_table = pdf_out_table(table_list, pdf_out_2)
# df_clean(df_pdf_table,  df_pdf_word_table, df_pdf_para, df_pdf_word_para, pdf_name_o)
