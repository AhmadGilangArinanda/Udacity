# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords

# import libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)

import sys
import re
from sqlalchemy import create_engine
import pickle

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    load_data
	Load Data from the Database Function
	
	Input:
	database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
	
    Retuns:
	X -> a dataframe containing features
	Y -> a dataframe containing labels
	category_names -> List of categories names
	"""
    # Load data
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

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
    tokenize
    Tokenize the text function
    
    Input:
    text -> Text message which needs to be tokenized
    
    Returns:
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
    
    # Remove stop words
    stop_words = stopwords.words("english")
    tokens = [tok for tok in tokens if tok not in stop_words]
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def build_model():
    """
    build_model
	build machine learning pipeline
    
    Input:
    pipeline -> build Pipeline function
	
	Returns:
	A Scikit ML Pipeline that process text messages and apply a classifier.
	   
	"""
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])


    #model parameters for GridSearchCV
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    cv = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    evaluate_model
	Evaluate Model function
	
	This function applies a ML pipeline to a test set and prints out classification report
	
	Input:
	   pipeline -> A valid scikit ML Pipeline
	   X_test -> Test features
	   y_test -> Test labels
	   category_names -> target names
	"""    
    y_pred = model.predict(X_test) # predict
    class_report = classification_report(y_test, y_pred, target_names=category_names) # print classification report
    print(class_report)
    print('Accuracy: {}'.format(np.mean(y_test.values == y_pred))) # print accuracy score

def save_model_as_pickle(pipeline, pickle_filepath):
    """
    save_model_as_pickle
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Input:
    pipeline -> GridSearchCV or Scikit Pipelin object
    pickle_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))
    
def main():
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_model()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(pipeline, X_test, y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()