from __future__ import annotations

import sys, utils
import pandas as pd
import numpy as np
import re, unicodedata, pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, f1_score, make_scorer


def load_data(database_filepath:str)-> tuple[np.ndarray, pd.DataFrame, pd.Index]:
    """load data from SQL database and split it to X and Y

    Args:
        database_filepath (str): SQL database file path

    Returns:
        X (np.ndarray): messages data
        Y (pd.DataFrame): labels
        category_names (pd.Index): category names (labels)
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("Messages", engine)
    X = df['message'].values
    Y = df[df.columns.difference(['id', 'message', 'original', 'genre'])] # all columns except these 4
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text:str)->list[str]:
    """tokenize the message / text

    Args:
        text (str): text to be tokenized

    Returns:
        list[str]: tokens
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8') # get rid of accented chars
    text = text.lower() # convert to lowercase
    text = re.sub(utils.regex_url, "urlplaceholder", text) # remove urls
    text = re.sub(utils.regex_non_alphanumeric, " ", text) # remove everything not letters or numbers
    tokens = word_tokenize(text) # tokenize words
    words = [w for w in tokens if w not in stopwords.words("english")] # remove stopwords
    # TODO: Improve lemmatization and stemming
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words] # Lemmatize verbs by specifying pos
    stem_words = [PorterStemmer().stem(w) for w in lemmed] # Reduce words in lemmed to their stems
    
    return stem_words


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
        ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    # model score
    model_score = utils.model_score(Y_test, y_pred)
    print(model_score)


def save_model(model, model_filepath):
    pickle.dump(model, open(f'{model_filepath}', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()