
# Lucid - Intelligent Document Processing & Review System ©

<br />

[![](https://img.shields.io/badge/Made_with-Python_|_Flask-blue.svg)]()
[![](https://img.shields.io/badge/Python-3.7.5-darkgreen.svg)]()
[![](https://img.shields.io/badge/Product-Lucid-important.svg)]()
<!-- [![](https://img.shields.io/badge/Powered_by-Engenuity_Ai-yellow.svg)]() -->

<!-- [![](https://img.shields.io/badge/torch-1.11.0-red.svg)]()
[![](https://img.shields.io/badge/Made_with-Flask-important.svg)]() -->

<br />

## Main Features

- OCR (Optical Character Recognition)
- Key Word Search
- Intelligent Search (Contextual Search)

<!-- ## Demo

This is a sample [demo](https://engenuitylk.sharepoint.com/:v:/s/PeoplesInsurance/EWi-FheOf9JBgkCjYY8ZaJABxMiyQmu6T3Hre8nsUd3dxA?e=EqUKeb) video of the project. -->

<br />

## Setting up the App Locally

Here is the guide for app installation in your own environments.


### Prerequisites

- Download the Sentence Transformers model Artefacts from [here.](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1/tree/main)

- Download the Layout Parser Model from [here.](https://engenuitylk.sharepoint.com/:f:/s/MuveProject/EmCZ_ZWy2QdEo1JvZzlpEhQBZCO6SS2LokhaUoFaRlmSnQ?e=WI2Rwm)

### Setup a Virtual Environment (Optional)

- Create a new virtual environment

    ```bash
    $ conda create --name <env-name> python=3.7.5
    $ # specify a convienient name for <env-name> as for the new env
    ```

- To activate the created environment

    ```bash
    $ conda activate <env-name>
    $ # replace the specified name with <env-name>
    ```

### Installation

Follow the below steps for the installation.

- Clone the repository:

    ```bash
    $ git clone https://github.com/engenuityai/lucid-engen-server.git
    ```

    or download the zip file from the [repository](https://github.com/engenuityai/lucid-engen-server.git) & extract it.

    Then navigate to the project root folder.


    ```bash
    $ cd lucid-engen-server
    ```

- Copy the Downloaded Sentence Transformer Model Artefacts and place inside the `model` folder in `intelligent_search_module` directory.


    <!-- Ex: Model Dir Path (Place in this path): `lucid-engen-server/intelligent_search_module/model/` -->

- Copy the Downloaded Layout Parser Model and place inside the `ocr` directory.

    <!-- Ex: Model Dir Path (Place in this path): `lucid-engen-server/ocr/` -->

- Install the Requirements: 

    ```bash
    $ pip install -r requirements.txt
    ```

- Run the App on localhost:

    ```bash
    $ flask run
    $ # You can change to any specific host and port to serve the app
    ```

-  App will be Running at: 
<http://127.0.0.1:5000>

    **Endpoint for the Keyword Search**
    <http://127.0.0.1:5000/keyword-search>

    **Endpoint for the Intelligent Search**
    <http://127.0.0.1:5000/intelligent-search>

<br />

## ✨ Code-base structure

The project code base structure is as below:

```bash
< PROJECT ROOT >
   |
   |-- assets/                              # Folder to store input images
   |     
   |-- intelligent_search_module/           # Module for Intelligent Search
   |    |-- model/                          # Folder containing all the Model Artefacts for Sentence Transformer 
   |    |-- __init__.py                     # Module Initialization
   |    |-- bert_process.py                 # Sentence Transformer Operations for Intelligent Search
   |
   |    
   |-- keyword_search_module/               # Module for Keyword Search    
   |    |-- __init__.py                     # Module Initialization
   |    |-- ocr_word_search_complete.py     # Keyword Search Operations
   |
   |
   |-- ocr/                                 # Module for OCR Operations
   |    |-- __init__.py                     # Module initialization
   |    |-- img_gen.py                      # Image generations
   |    |-- Layout_det.py                   # Layout Parsing
   |    |-- pyteseract_para_ocr_bb.py       # Paragraph Level OCR Processors
   |    |-- Pytesseract_table_ocr_bb.py     # Table Level Operations
   |    |-- run_ocr.py                      # Run Tasks Operations
   |    |-- test_07_14.h5                   # Layout Parser Model
   |
   |
   |-- utils/                               # Support Functions
   |    |-- __init__.py                     # Module initialization
   |    |-- cleaning.py                     # Text data preprocessing/postprocessing
   |    |-- generate.py                     # Result generation operations
   |
   |
   |-- intelligent_search_output/           # Output Result Directory for Intelligent Search   
   |    |-- csv                             # Contains reslted csv files
   |    |-- pics                            # Contains generated images
   |    |-- result_pdfs                     # Contains final pdf outputs
   |
   |
   |-- keyword_search_output/               # Output Result Directory for Keyword Search
   |
   |
   |-- requirements.txt                     # Requirements, all the required dependencies
   |-- .flaskenv                            # Flask environment configurations
   |-- Dockerfile                           # Dockerfile Script for the App
   |
   |-- app.py                               # Setup App Configuration
   |-- main.py                              # Main App Starter - WSGI gateway
   |
   |-- ************************************************************************
```
