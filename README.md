## Table of contents
* [General info](#general-info)
* [Setup](#setup)

## General info
### Disaster Response Pipeline Project
This project is a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data

Random Forest Classifier is used to classify a recieved message into one of the 36 categories. 
	
## Setup
### required libraries:
The main libraries in this project are nltk, numpy and pandas.

To run this project:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

