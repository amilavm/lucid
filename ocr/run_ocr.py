import ocr.img_gen as imggen
import ocr.Layout_det as layoutdet
import ocr.pyteseract_para_ocr_bb as ptpara
import ocr.Pytesseract_table_ocr_bb as pttable

dpi = 500
# parent_dir = "../OCR_pipeline/Output_files/"
parent_dir = 'outputs/'

def img_gen():
    pdf_file, tail = imggen.input_df()
    pdf_out_2 = imggen.convert_pdf(pdf_file, dpi, parent_dir, tail)
    return pdf_out_2

def layout_det(pdf_out_2):
    table_list, para_list, pdf_out_2 = layoutdet.structure_selector(pdf_out_2)
    return table_list, para_list, pdf_out_2

def pt_para(para_list, pdf_out_2):
    df_pdf_para, df_pdf_word_para, pdf_name_o = ptpara.pdf_out_para(para_list, pdf_out_2)
    return df_pdf_para, df_pdf_word_para, pdf_name_o

def pt_table(table_list, pdf_out_2, df_pdf_para, df_pdf_word_para, pdf_name_o):
    df_pdf_table, df_pdf_word_table = pttable.pdf_out_table(table_list, pdf_out_2)
    df, dfw = pttable.df_clean(df_pdf_table, df_pdf_word_table, df_pdf_para, df_pdf_word_para, pdf_name_o)
    return df, dfw

