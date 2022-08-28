# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

## Table of Contents

1. Description
2. Project Components
    - ETL Pipeline
    - ML Pipeline
    - Flask Web App
3. Getting Started
    - Dependencies
    - Installing
    - Instructions
4. File Description



## 1. Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time

## 2. Project Components
There are three components of this project:

2.1. ETL Pipeline
File data/process_data.py contains data cleaning pipeline that:

- Loads the messages and categories dataset
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2.2. ML Pipeline
File models/train_classifier.py contains machine learning pipeline that:

- Loads data from the SQLite database
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

2.3. Flask Web App

- Running this command from app directory will start the web app where users can enter their query. For example massage : "There's nothing to eat and water, we starving and thirsty"

## 3. Getting Started

3.1 Dependencies
- Python 3.
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

3.2 Installing
This project requires Python 3.x and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Json
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sys
- Re
- Pickle
You will also need to have software installed to run and execute an iPython Notebook

3.3 Instructions
Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- Run the following command in the app's directory to run your web app. python run.py
- Go to http://0.0.0.0:3001/

## 4. File Description

         disaster-response-pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data                
                |-- DisasterResponse.db
                |-- categories.csv
                |-- message.csv
                |-- process_data.py
          |-- models
                |-- classifier.rar (classifier.pkl)
                |-- train_classifier.py
          |-- image     
          |-- README
