# Disaster Response Pipeline Project

Here we create a web app capable of classifying new Tweets using the labeled data from Figure Eight. The newly classified data then can be used to inform relevant disaster relief agencies.

## How to use:

The project has been developed with following dependencies

 - python 3.9.7
 - nltk 3.6.2
 - pandas 1.2.4
 - flask 2.0.1
 - scikit-learn 0.24.2
 - sqlalchemy 1.4.23
 - plotly 5.3.1


### Running the Flask app

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project tree

 - `app/run.py`: flask web app
 - `data/*`: ETL pipeline and the processed dataset 
 - `models/*`: ML pipeline and the classifier  
 - `notebooks`: preliminary testings

## Acknowledgements

This project is a part of Udacity Data science Nanodegree and the dataset is provided by the Figure Eight. 