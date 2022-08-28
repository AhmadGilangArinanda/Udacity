# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

## Table of Contents

1. [Description](#description)
2. [Project Components](#project_components)
    - [ETL Pipeline](#etl)
    - [ML Pipeline](#ml_pipeline)
    - [Flask Web App](#flask)
3. [Getting Started](#getting_started)
    - [Dependencies](#dependencies)
    - [Installing](#installing)
    - [Instructions](#instructions)
4. [File Description](#file)


<a name="description"></a>
## 1. Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time

<a name="project_components"></a>
## 2. Project Components
There are three components of this project:

<a name="etl"></a>
2.1. ETL Pipeline
File data/process_data.py contains data cleaning pipeline that:

- Loads the messages and categories dataset
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

<a name="ml_pipeline"></a>
2.2. ML Pipeline
File models/train_classifier.py contains machine learning pipeline that:

- Loads data from the SQLite database
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

<a name="flask"></a>
2.3. Flask Web App

- Running this command from app directory will start the web app where users can enter their query. For example massage : "There's nothing to eat and water, we starving and thirsty"

<a name="getting_started"></a>
## 3. Getting Started

<a name="dependencies"></a>
3.1 Dependencies
- Python 3.
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
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

<a name="instructions"></a>
3.3 Instructions
Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- Run the following command in the app's directory to run your web app. python run.py
- Go to http://0.0.0.0:3001/

<a name="file"></a>
## 4. File Description

         disaster-response-pipeline
          |-- app
                |-- templates
                        |-- go.html # main page of web app
                        |-- master.html # classification result page of web app
                |-- run.py # Flask file that runs app
          |-- data                
                |-- DisasterResponse.db # database to save clean data to
                |-- categories.csv # data to process 
                |-- message.csv # data to process
                |-- process_data.py
          |-- models
                |-- classifier.rar (classifier.pkl) # saved model 
                |-- train_classifier.py
          |-- image     
          |-- README
