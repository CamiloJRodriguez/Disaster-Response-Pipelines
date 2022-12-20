# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    '''
    load data from database
    
    input:
        database_filepath: The database path
    output:
        X: Dataframe with features
        Y: Target feature
    '''
    
    # Connection with database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Databse to Dataframe
    df = pd.read_sql_table('data_preparation', engine)
    
    # Features (Predictors)
    X = df.message
    
    # Target Feature
    Y = df.iloc[:,4:]
    
    # Get column names
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    In this function process text data
    
    input:
        text: data to clean (message)
    output:
        clean_tokens: is the data normalized, tokennized and lemmatized
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Funtion have a pipeline take message column and get 36 columns or features.
    Use MultiOutputClassifier to Random Forest (RFC) as Classifier.
    Last get Grid Search CV with params
    
    Output:
        model: Grid Search CV with params
    '''
    
    # ML Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])  
    
    # Params to be applied in GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [40, 50, 60],
        ##'clf__estimator__n_jobs': [2, 3, 4],
        'tfidf__use_idf': (True, False)
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model and get a report of scores
    
    input:
        model: call model with Grid Search CV with params (pipeline)
        X_test: predictor features to test
        Y_test: target feature to test
        category_names: labels
        
    output:
        scores model report
    '''
    
    # Get prediction
    y_pred = model.predict(X_test)
    
    # Report with model scores
    i = 0
    for col in Y_test:
        print('Column {}: {}'.format(i+1, col))
        print(classification_report(Y_test[col], y_pred[:, i]),  '--------------------------')
        i = i + 1


def save_model(model, model_filepath):
    '''
    Save model in pickle file
    
    input:
        model: call model with Grid Search CV with params (pipeline)
        model_filepath: the pickle file path
        
    output:
        the file saved in pickle file        
    '''
    
    with open ('NLPclassifier.pkl', 'wb') as file:
        pickle.dump(model, model_filepath)


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
        evaluate_model(model, X_test, Y_test, category_names)

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