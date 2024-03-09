from PIL import Image
import pandas as pd
import os
import re
import cv2
import tempfile
import img2pdf
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


image_base_path = "intelligent_search_output/pics"
# csv_base_path = "assets/csv_files"
output_base_path = "intelligent_search_output/result_pdfs/"
# filename= "143438 From OSS - Lease"

temp = tempfile.TemporaryDirectory(prefix="muve_")

# image_names = os.listdir(image_base_path)

def flow_images_from_dir(dir_path, docname):
    image_names = os.listdir(os.path.join(dir_path, docname))
    image_names.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in image_names:
        img = cv2.imread(os.path.normpath(os.path.join(dir_path, docname, filename)))
        if img is not None:
            # print(filename)
            yield filename, img


def plot_pdf(outdf, category,  docname):
    outdf['page_num'] = outdf["page_num"].apply(lambda x: x.split("/")[-1])
    # print(outdf)
    for page, image in flow_images_from_dir(image_base_path, docname):
        cat_df = outdf[outdf['page_num'] == page]
        alpha = 0.3
        for row in cat_df.itertuples():
            # category = ", ".join(row.category).replace("_", " ")
            x1 = int(row.xmin)
            x2 = int(row.xmax)
            y1 = int(row.ymin)
            y2 = int(row.ymax)
            score = row.search_score
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,255), -1)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            # type_word = str(category) + ' --- Search Score: '+ str(score)
            cv2.putText(image, category, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (76, 153, 0), 5)
            cv2.putText(image, f'conf::{score}%', (x2, y2+55), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 0, 0), 5)

        image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

        cv2.imwrite(os.path.join(temp.name, page), image)

def generate_pdf(docname):
    # print(os.listdir(temp.name))
    image_names = os.listdir(temp.name)
    image_names.sort(key=lambda f: int(re.sub('\D', '', f)))
    with open(output_base_path+"out_"+docname+".pdf", "wb") as f:
        f.write(img2pdf.convert([os.path.join(temp.name, i) for i in image_names]))
    temp.cleanup()

# if i.endswith(".jpg")