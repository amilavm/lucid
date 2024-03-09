# Import libraries
from pdf2image import convert_from_path, pdfinfo_from_path
import os
from PyPDF2 import PdfFileWriter, PdfFileReader
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dpi = 500
# parent_dir = "../OCR_pipeline/Output_files/"
# parent_dir = 'outputs/'

# def input_df():
#     # print("""OCR Process Begins\n""")
#     # print("""PDF to Image Convertion """)
#     pdf_file = input(str('Please Enter the Document Path: '))
#     head, tail = os.path.split(pdf_file)
#     tail = tail.split('.')[0]
#     # print(pdf_file)
#     print(f'Processing {tail} Document')
#     return pdf_file, tail

# ocr_out_imgs_path = os.path.join('output/pics')
img_folder_path = 'intelligent_search_output/pics'

def convert_pdf(pdf_file):

    ##########
    head, tail = os.path.split(pdf_file)
    tail = tail.split('.')[0]
    ##########

    # pages = convert_from_path(pdf_file, dpi)
    inputpdf = PdfFileReader(open(pdf_file, "rb"))
    maxPages = inputpdf.numPages

    img_out_path = os.path.join(img_folder_path, tail)

    # os.mkdir(img_out_path)
    try:
        os.makedirs(img_out_path, exist_ok = True)
        print("Directory '%s' created successfully" % img_out_path)
    except OSError as error:
        print("Directory can not be created '%s'" % error)

    for pages in range(1, maxPages + 1, 4):
        pages_imgs = convert_from_path(pdf_file, dpi=500, first_page=pages, last_page=min(pages + 4 - 1, maxPages))#, fmt='jpeg')

        # #show the image in the notebook
        saved_pages = 0
        for i, page in enumerate(pages_imgs):

            filename = img_out_path+'/' + 'page'+str(i + int(pages)) + '.jpg'
            page.save(filename, 'JPEG')
            # print('page saved: ',str(i + int(pages)))
            saved_pages += 1
        
        pages_imgs = None
    print('pages saved!')
    return img_out_path
