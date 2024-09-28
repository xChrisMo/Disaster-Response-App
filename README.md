Disaster Response Pipeline Project

![Homepage](app_pictures/Screenshot%202024-09-22%20at%2001.20.18.png)

## Project Overview
This project is part of the Data Science Nanodegree by Udacity in collaboration with Appen. The dataset contains tweets and messages from real-life disaster events. The objective is to build a model to categorize messages in real-time using Natural Language Processing techniques.

## Data:

The dataset contains over 30,000 messages drawn from events including earthquakes in Haiti and Chile, floods in Pakistan, super-storm Sandy in the U.S.A. in 2012, and news articles spanning hundreds of different disasters. The dataset was provided by [Appen](https://www.appen.com/). The data has been encoded with different categories related to disaster response, with sensitive information removed.


## Dependencies

* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
* Natural Language Processing Libraries: NLTK
* SQLite Database Libraries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly


## File Description
       disaster_materials
              |-- app
                    |-- templates
                            |-- go.html
                            |-- master.html
                    |-- run.py
              |-- data
                    |-- disaster_message.csv
                    |-- disaster_categories.csv
                    |-- DisasterResponse.db
                    |-- process_data.py
              |-- models
                    |-- classifier.pkl
                    |-- train_classifier.py
                    
       jupyter_notebooks
              |-- ETLPipelinePreparation.ipynb 
              |-- MLPipelinePreparation.ipynb
              
       app_pictures
              |-- screenshot__png*
          
## How to Use

After cloning this repository, use the `disaster_materials`, along these instructions:

* Run `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db` to clean and store the data into a database
* Run `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl` to load data from the database, run the ML pipeline with its NLP properties and then saves as a pickle file
* Run this command when in the app's directory to launch the web app. `python run.py`
* Go to `http://0.0.0.0:3001/`
* Finally, enter a message to classify, whatever comes to mind. You should have a result close to that in the image below

![app_screenshot](app_pictures/Screenshot%202024-09-22%20at%2001.22.00.png)

## Files in the Repository

#### Disaster_materials
* App folder including the templates folder and "run.py" for the web application
* Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for cleaning and database storage.
* Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.

#### Jupyter_notebooks
* **ETLPipelinePreparation.ipynb**: Data preprocessing steps used to merge, clean and store datasets (used to script `process_data.py`)
* **MLPipelinePreparation.ipynb**: Series of ML and NLP pipelines saved to create a pickle file (used to script `train_classifier.py`)


## Licensing
Copyright (c) 2024, Ayo

This project is for personal and educational purposes only. All rights reserved.
