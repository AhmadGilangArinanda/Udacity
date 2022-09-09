# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load_data
    load data from csv files and merge to single pandas dataframe

    Input: 
    massages_filepath -> filepath to massages csv file
    categories_filepath -> filepath to categories csv file
    
    Returns:
    df -> dataframe merging messages and categories 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 

def clean_data(df):
    """
    clean_data
    clean df that containing massages and categories

    Input:
    df -> Combined data containing massages and categories

    Returns:
    df -> Combined data containing masssages and categories cleaned up
    """
    # split categories into separate category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe,
    # use this row to extract a list of new column names for categories, and
    # rename the columns of `categories`
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
         categories[column] = categories[column].str[-1] # set each value to be the last character of the string
         categories[column] = categories[column].astype(int) # convert column from string to numeric
         categories.related.replace(2,1,inplace=True) # "related" column has value 0,1,2 and we need only 0 & 1 hence replacing 2 with 1
    
    # Replace categories column in df with new category columns
    df.drop('categories' , axis = 1 , inplace = True) # drop the original categories column from `df`
    df = pd.concat([df,categories], join='inner', axis=1) # concatenate the original dataframe with the new `categories` dataframe
    df.drop_duplicates(inplace = True) # drop duplicates

    return df

def save_data(df, database_filename):
    """
    save_data
    save data to SQLite database function
    
    Input:
    df -> Combined data containing messages and categories with categories cleaned up
        
    Returns:
    database_filename -> Path to SQLite destination database
    """ 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()