import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    First load two csv files (messages and categories) and later merge both files
    
    input:
        messages_filepath : the file path of first csv, messages, must be string
        categories_filepath : the file path of second csv, categories, must be string
    
    output:
        df: the dataframe from merge both csv files   
    '''
    
    # import data
    # This filepaths include csv files
    messages = pd.read_csv('messages_filepath') 
    categories = pd.read_csv('categories_filepath')
    
    # merge data
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    
    return df  


def clean_data(df):
    '''
    Get the categorie names from df, cleaning (remove duplicates) and return in correct datatype
    
    input:
        df: the dataframe from merge both csv files, from load_data function
     
    output:
        df: the dataframe cleaned
    '''
    
    # Get columns from categories csv, splited by semicolon
    categories = df.categories.str.split(';', expand = True)
    
    # select first row of categories dataframe
    row = categories[0:1]
    
    # Extract column names using lambda function
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:].tolist() 
    
    # Rename dataframe category names 
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].stype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    # Replace in related column values of 2 by 1, because is only 1% all data
    categories['related'] = categories['related'].replace(2,1)
    
    # Drop categories column from df
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with categories dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Clean dataframe dropping duplicates
    df.drop_duplicates(inplace = True)
    
    
    
    return df


def save_data(df, database_filename):
    '''
    Save into an sqlite database the clean dataframe
    
    input:
        df: dataframe cleaned get from clean_data function
        
    output:
        database: the database file
    '''
    
    # Send dataframe cleaned to data_preparation database
    engine = create_engine('sqlite:///data_preparation.db')
    df.to_sql('data_preparation', engine, index=False, if_exists='replace')
    


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