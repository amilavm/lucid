#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
# import the necessary packages
from pytesseract import Output
import cv2
import pytesseract
import os
# pytesseract.pytesseract.tesseract_ocr = r'..../lib/python3.8/site-packages/pytesseract/pytesseract.py'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import pandas as pd
# import imutils

def pdf_out_para(para_list, pdf_out):
    # print('paragraph/blockwise DataFrame Creation \n')
    column_names = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'xmin', 'ymin', 'xmax', 'ymax',
                    'conf', 'text',
                    'structure']
    column_names_para = ['page_num', 'block_num', 'xmin', 'ymin', 'xmax', 'ymax', 'structure']
    df_pdf_word_para = pd.DataFrame(columns=column_names)

    df_pdf_para = pd.DataFrame(columns=column_names_para)

    for j in para_list:

        img = cv2.imread(j)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        blur = cv2.medianBlur(gray, 5)


        results = pytesseract.image_to_data(blur, output_type=Output.DICT)

        df = pd.DataFrame(results)
        df['page_num'] = str(j)
        df['structure'] = 'para'
        df = df[(df['text'] != '') & (df['text'] != ' ')]

        num_blocks = df['block_num'].unique()

        min_conf = 50

        #         renaming the df according to the naming convention
        df.rename(columns={'top': 'ymin', 'height': 'ymax', 'left': 'xmin', 'width': 'xmax'}, inplace=True)

        for u in num_blocks:
            column_names_para = ['page_num', 'block_num', 'xmin', 'ymin', 'xmax', 'ymax']
            df_pdf_p = pd.DataFrame(columns=column_names_para, index=num_blocks)

            df_pdff = df.loc[df['block_num'] == u]
            #         words_in_block
            l = []
            t = []

            for k in range(df_pdff.shape[0]):
                l.append(df_pdff.iloc[k]['xmin'])
                t.append(df_pdff.iloc[k]['ymin'])

                if l or t:
                    top = min(t)
                    bot = (df_pdff.loc[df_pdff['ymin'] == max(t), 'ymax'].iloc[0]) + max(t)
                    left = min(l)
                    right = (df_pdff.loc[df_pdff['xmin'] == max(l), 'xmax'].iloc[0]) + max(l)
                else:
                    pass
            df_pdf_p['xmin'] = left
            df_pdf_p['ymin'] = top
            df_pdf_p['ymax'] = bot
            df_pdf_p['xmax'] = right
            df_pdf_p['page_num'] = str(j)
            df_pdf_p['block_num'] = u
            df_pdf_p['structure'] = 'para'

            df_pdf_para = pd.concat([df_pdf_para, df_pdf_p], axis=0, ignore_index=True, sort=False)

            #         df_pdf = pd.concat([df_pdf, df], axis=0, ignore_index=True, sort=False)
            #         print('\n\n', df)
            cv2.rectangle(img, (left, top), (right, bot), (255, 0, 0), 7)

        for i in range(0, len(results["text"])):

            (x, y, w, h) = (results['left'][i], results['top'][i], results['width'][i], results['height'][i])
            text = results["text"][i]
            conf = float(results["conf"][i])

            # filter out weak confidence text localizations
            if conf > min_conf:
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        df['ymax'] = df['ymin'] + df['ymax']
        df['xmax'] = df['xmin'] + df['xmax']

        df_pdf_word_para = pd.concat([df_pdf_word_para, df], axis=0, ignore_index=True, sort=False)

        # print(str(j))
        # save the detected images with contours drawn
        head, tail = os.path.split(str(j))
        tail = tail.split('.')[0]
        pdf_name_o = head.split('/')[-1]


    df_pdf_para_string = df_pdf_word_para.copy()
    df_pdf_para_string = df_pdf_para_string.groupby(['page_num', 'block_num'])['text'].apply(lambda x: ' '.join(x)).reset_index()

    df_pdf_para.drop_duplicates(inplace=True)

    df_pdf_para = pd.merge(df_pdf_para_string, df_pdf_para, how='left', left_on=['page_num', 'block_num'], right_on=['page_num', 'block_num'])

    # print('\n paragraph/blockwise DataFrame Creation completed')
    return df_pdf_para, df_pdf_word_para, pdf_name_o
