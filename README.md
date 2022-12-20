
# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code was developed in python 3 along with the libraries that it already has installed, being applicable in Anaconda. The libraries used were:
* sys
* Numpy
* Pandas
* sqlite3
* sqlalchemy
* re (regular expressions)
* nltk
* Scikit learn
* Pickle
* Json
* Flask
* Plotly

## Project Motivation <a name="motivation"></a>

I was interested in using real emergency messages, which would allow them to be used as an input to create a prediction model (classification) indicating on a web page, when faced with a new message, what type or category it is, speeding up the response processes of the agencies of emergency.

## File Descriptions <a name="files"></a>

- app
    + templates
        + master.html  # main page of web app
        + go.html  # classification result page of web app
    + run.py  # Flask file that runs app

- data
    + disaster_categories.csv  # data to process 
    + disaster_messages.csv  # data to process
    + process_data.py
    + data_preparation.db   # database to save clean data to

- models
    + train_classifier.py
    + NLPclassifier.pkl  # saved model 

- README.md


## Instructions <a name="instructions"></a>

To execute this app, folow the next instructions:

1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database 
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves 
        'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

Must give credit to [Appen](https://appen.com/) (formally Figure 8) for the data. Given my learning process with Udacity around data science and for which I am grateful, feel free to give me recommendations that allow me to continue growing as a data scientist, any concern or appreciation will be well received!
