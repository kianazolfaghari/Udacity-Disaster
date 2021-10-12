import sys
import pandas as pd
import sqlite3
from pathlib import Path

def load_data(messages_filepath, categories_filepath):
    """ loads data.
    Keyword arguments:
    messages_filepath -- filepath to messages
    categories_filepath -- filepath to categories
    """
    
    messages = pd.read_csv(str(messages_filepath)) 
    categories = pd.read_csv(str(categories_filepath)) 
    df = pd.merge(messages, categories, on="id") 
    return df

def clean_data(df):
    """ cleans data.
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0,:]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = [x[-1] for x in categories[column]]
        categories[column] = categories[column].astype('int')
    
    df = pd.concat([df, categories], axis = 1)
    df = df.drop(['categories'], axis = 1)
    for i in range(4, df.shape[1]):
        df = df[df.iloc[:,i].isin([0,1])]
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """ saves data.
    keyword arguments:
    df -- dataframe
    database_filename -- name of database.
    """
    
    conn = sqlite3.connect(database_filename)
    database_filename = Path(database_filename).stem
    df.to_sql(database_filename, conn, index=False, if_exists='replace') 
    
def main():
    """ main function to receive the filepath for messages, categories, and database. 
    this function loads, cleans, and saves the data 
    """
    
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