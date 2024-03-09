from flask import Flask
# from flasgger import Swagger

UPLOAD_FOLDER = './assets'
OUTPUT_FOLDER = './keyword_search_output'

app = Flask(__name__)
# swagger = Swagger(app, template={
#  "swagger": "2.0",
#  "info": {
#   "title": "Inference",
#   "version": "1.0.0"
#  }
# })
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024