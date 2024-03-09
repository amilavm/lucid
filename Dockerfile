 #Using the base image with python 3.7
 FROM python:3.7.5
 
 WORKDIR /app

 COPY requirements.txt /app/requirements.txt

 #Set our working directory as app
#  WORKDIR /app


 #Installing python packages pandas, scikit-learn and gunicorn
 RUN python -m pip install --upgrade pip
 RUN pip install -r requirements.txt

 COPY . /app

 RUN apt-get update
 RUN apt-get install ffmpeg libsm6 libxext6  -y
 RUN apt-get -y install tesseract-ocr
 
 
#  #Exposing the port 5000 from the container
#  EXPOSE 5000

 #Starting the python application
 CMD ["python","main.py"]