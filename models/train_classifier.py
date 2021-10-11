import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import sqlite3 
from pathlib import Path

def load_data(database_filepath):
    """ loads data.
    keyword argument:
    database_filepath -- the path to the database 
    
    """
    database_filename = Path(database_filepath).stem
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM "+database_filename, conn)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = [str(i) for i in np.unique(Y)]
    return X, Y, category_names

def tokenize(text):
    """ tockenizes a test.
    keyword arguments:
    text -- a sentence.
    
    """
    
    text = re.sub(r'[^\w\s]','', text)
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in token]
    return clean_tokens
  
def build_model():
    """ builds the pipline with GridSearch.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ evaluates the model performance.
    Keyword arguments:
    model -- a trained model.
    X_test -- input test data.
    Y_test -- output test data.
    category_names -- expected labels for the output.
    
    """
    
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print("category "+str(i)+":", classification_report(Y_test[:,i], y_pred[:,i], target_names = category_names))

def save_model(model, model_filepath):
    """ saves the model.
    Keyword arguments:
    model -- a trained model.
    model_filepath -- the path to save the model.
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """ the main function to read in the data and split it to training and testing and train/evaluate/save the model.
    """
    
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