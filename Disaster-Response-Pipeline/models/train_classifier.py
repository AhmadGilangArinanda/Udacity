# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import re
from sqlalchemy import create_engine
import pickle

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    # Load data
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('DisasterResponse.db', engine)

    # Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)

    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns

    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    #model parameters for GridSearchCV
    parameters = { 
                    'clf__estimator__n_estimators': [20],
                    'clf__estimator__min_samples_split': [2]
              }
    cv = GridSearchCV (pipeline, param_grid= parameters, verbose =7 )

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)

def save_model_as_pickle(pipeline, pickle_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format('data/DisasterResponse.db'))
        X, y, category_names = load_data('sqlite:///D:\\New\\data\\DisasterResponse.db')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_model()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(pipeline, X_test, y_test, category_names)

        print('Saving pipeline to {} ...'.format('models/classifier.pkl'))
        save_model_as_pickle(pipeline, 'classifier.pkl')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()