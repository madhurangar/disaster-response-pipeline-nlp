#!/usr/bin/env python3 
from __future__ import annotations
import unicodedata as u
import pandas, numpy, re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer

regex_non_alphanumeric = '[^a-zA-Z0-9]'
regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
regex_html_tag = "<(\"[^\"]*\"|'[^']*'|[^'\">])*>"

def run_regex(data:list, regex:str, pout:int=1)->None:
    """run a regex on data to check if there are valied matches. Only
    prints the first occurrence by default or N times if defined. Finally,
    prints the total number of occurrences.    

    Args:
        data (list): data set to analyse
        regex (str): regex command
        N (int): number outputs to print
    """
    counter = 0
    for idx, datapoint in enumerate(data):
        detected = re.findall(regex, datapoint)
        if detected: 
            counter += 1
            if counter <= pout or pout==-1:
                print(f"Detected: {detected}: Index={idx}")
                print(f"Source: {datapoint}")
    print(f"Total: {counter}")
            

def check_accented_chars(data:list, pout:int=1)->None:
    """look for accented characters.

    Args:
        data (list): dataset
        pout (int, optional): number of outputs to print. Defaults to 1.
    """
    counter = 0
    for idx, datapoint in enumerate(data):
        # check for Normalization Form Compatibility Decomposition (NFKD)
        # state = False -> accented characters are present!  
        state = u.is_normalized("NFKD", datapoint)
        if not state: 
            counter += 1
            if counter <= pout or pout==-1:
                print(f"Detected at Index={idx}")
                print(f"Source: {datapoint}")
    print(f"Total: {counter}")
        

def remove_html_tags(text):
    # TODO: complete!
    pass


def compare_test_train(idx:int, x_test:pandas.DataFrame, y_test:pandas.DataFrame, y_predict:numpy.ndarray)->None:
    """extract an index (data point) and check its predicted value and the label

    Args:
        idx (int): index of the data set to be checked
        x_test (pandas.Dataframe): test data
        y_test (pandas.Dataframe): test data
        y_predict (numpy.ndarray): predicted data
    """
    print(f"X_test[{idx}] = {x_test[idx]}")
    print("columnname:predicted(label)")
    for i, column in enumerate(y_test.columns):
        predicted = y_predict[idx][i]
        label = y_test.iloc[idx:idx+1,i].values[0]
        if predicted or label:
            print(f"{column}:{predicted}({label})", end=" ")
    print("")
    

def rog_cmap(value:float)->str:
    """returns background-colour parameter to be use with
    pandas styles

    Args:
        value (float): value between 0 and 1

    Returns:
        str: pandas styles parameter
    """
    if value <= 0.5:
        color = '#993414'
    elif value > 0.5 and value <= 0.75:
        color = '#cc5500'
    elif value > 0.75:
        color = "#508104"
        
    return f'background-color: {color}'   


def create_scores_table(y_test:pandas.DataFrame, y_predict:numpy.ndarray)->pandas.DataFrame:
    """tabulate model scores and colorise them by the value:
    value <= 0.5         : dark red
    0.75 >= value > 0.5  : orange
    value > 0.75         : green

    Args:
        y_test (pandas.DataFrame): true values, ground truth
        y_predict (numpy.ndarray): predicted values

    Returns:
        pandas.Dataframe: coloured datafame
    """
    columns = y_test.columns.values
    df=pandas.DataFrame(index=columns, columns=["precision", "recall", "f1-score", "accuracy"])
    for idx, col in enumerate(columns):
        col_report = classification_report(y_test[col].values, y_predict[:,idx], output_dict=True, zero_division=0)
        acc_score = accuracy_score(y_test[col].values, y_predict[:,idx])
        row = {"precision": col_report['macro avg']["precision"], 
            "recall": col_report['macro avg']["recall"], 
            "f1-score": col_report['macro avg']["f1-score"],
            "accuracy": acc_score
        }
        df.loc[col] = row
        
    return df.style.applymap(rog_cmap)


def model_score(y_test:pandas.DataFrame | numpy.ndarray, y_pred:numpy.ndarray):
    """calculate model score using f1-score

    Args:
        y_test (pandas.DataFrame | numpy.ndarray): true values or the ground truth
        y_pred (numpy.ndarray): predicted values

    Returns:
        average_score(float): average f1-score
    """
    try:
        columns = y_test.columns.values
    except AttributeError:
        columns = range(len(y_test))
        
    scores_collection = numpy.array([])    
    for idx, col in enumerate(columns):
        try: 
            score = f1_score(y_test[col].values, y_pred[:,idx], average='macro')
        except IndexError:
            score = f1_score(y_test[:,idx], y_pred[:,idx], average='macro')
        scores_collection = numpy.append(scores_collection, score)

    average_score = scores_collection.mean()
    
    return average_score


def test_model(model, X, Y, tokenize, force_numpy=False, n_jobs=-1)->pandas.DataFrame:
    """test an estimator. for a rigorous test or cross validation try
     `sklearn.model_selection`

    Args:
        model (estimator): ML estimator
        X (pandas.DataFrame | numpy.ndarray): [description]
        Y (pandas.DataFrame | numpy.ndarray): [description]
        tokenize (function): tokenizer
        force_numpy (bool, optional): convert X and Y to numpy arrays
        n_jobs (int, optional): number of simultaneous jobs. Defaults to -1.

    Returns:
        pandas.DataFrame: A colored dataframe containing model scores
    """
    # TODO: convert X and Y to numpy.ndarrays
    if force_numpy:
        pass
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = MultiOutputClassifier(model, n_jobs=n_jobs)

    # train classifier
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    clf.fit(X_train_tfidf, y_train)

    # predict on test data
    X_test_counts = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)

    # display results
    scores_table = create_scores_table(y_test, y_pred)
    print(f"{model} : {model_score(y_test, y_pred)}")
    
    return scores_table

