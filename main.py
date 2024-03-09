import os
import urllib.request
import filetype
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd

from keyword_search_module import ocr_word_search_complete
from ocr import img_gen, Layout_det, pyteseract_para_ocr_bb, Pytesseract_table_ocr_bb
from intelligent_search_module import bert_process
from utils.generate import plot_pdf, generate_pdf
from utils.cleaning import clean_block_df

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({'message' : 'Welcome to the Lucid Search API'})

@app.route('/keyword-search', methods=['POST'])
def keyword_search():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # resp = jsonify({'message' : 'File successfully uploaded'})
        # resp.status_code = 201
        # return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

    if (request.form):
        search_keyword = str(request.form['search_keyword'])
        action_type = str(request.form['action_type'])

    output_file = Path(file_path).stem+'_output.pdf'
    if filetype.is_image(file_path):
        ocr_word_search_complete.ocr_img(
            # if 'search_str' in (args.keys()) else None
            img=None, input_file=file_path, search_str=search_keyword, action=action_type
        )
    else:
        summary, page_stats = ocr_word_search_complete.ocr_file(
            input_file=file_path, output_file=output_file, search_str=search_keyword, action=action_type
        )
    # total_matches = summary["Total matches"]
    # if int(total_matches) > 0:
    #     msg = 'Found '+ str(total_matches) + ' matches overall.'
    # else:
    #     msg = 'Found no matches!'
    resp = jsonify({
        'message' : 'Process Success.',
        'overall summary' : summary,
        'page stats' : page_stats
        })
    # resp.status_code = 200
    return resp

@app.route('/intelligent-search', methods=['POST'])
def intelligent_search():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # resp = jsonify({'message' : 'File successfully uploaded'})
        # resp.status_code = 201
        # return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

    if (request.form):
        search_input = str(request.form['search_input'])
        # action_type = str(request.form['action_type'])
    
    # save images for each page
    images_path = img_gen.convert_pdf(file_path)
    print(images_path)

    table_list, para_list, images_path = Layout_det.structure_selector(images_path)
    
    print("Performing OCR...")
    # t2 = show_animate("Generating text ...")
    df_pdf_para, df_pdf_word_para, pdf_name_o = pyteseract_para_ocr_bb.pdf_out_para(para_list, images_path)
    # stop_animate(t2)

    df_pdf_table, df_pdf_word_table = Pytesseract_table_ocr_bb.pdf_out_table(table_list, images_path)
    df, dfw = Pytesseract_table_ocr_bb.df_clean(df_pdf_table, df_pdf_word_table, df_pdf_para, df_pdf_word_para, pdf_name_o)
    print("OCR Completed Successfully!")

    # save csv files
    csv_root = 'intelligent_search_output/csv'
    filename_without_ext = filename.split('.')[0]
    csv_folder_path = os.path.join(csv_root, filename_without_ext)

    try:
        os.makedirs(csv_folder_path, exist_ok = True)
        print("Directory '%s' for csv created successfully" % csv_folder_path)
    except OSError as error:
        print("Directory can not be created '%s'" % error)
    
    # clean df
    df = clean_block_df(df)

    df.to_csv(csv_folder_path + '/' + 'para.csv')
    dfw.to_csv(csv_folder_path + '/' + 'word.csv')
    # '''
    # Perform contextual search with BERT model
    # create input context for bert model
    text_context = bert_process.build_context(df)
    search_sents, search_sents_scrs = bert_process.bert_search(text_context, search_input)
    # block_corpus, search_sent = get_corpus()
    out_df = bert_process.output_df(search_sents, search_sents_scrs, df)
    if out_df is None:
        print('Sorry, There are No Possible Matches!!!')
        msg = 'Sorry, There are No Possible Matches Found!'
    else:
        plot_pdf(out_df, search_input, filename_without_ext)
        out_df.to_csv(csv_folder_path + '/' + "search_results.csv")
        generate_pdf(filename_without_ext)
        print(f'Final PDF File Saved Successfully for {filename_without_ext}')
        msg = 'Found Matches, Final PDF File Saved Successfully'
    # '''
    # msg = 'success'

    resp = jsonify({'message' : msg})
    resp.status_code = 200
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug = True)