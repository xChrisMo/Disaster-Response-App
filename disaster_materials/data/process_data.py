import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load and merge message and category data from CSV files.

    Args:
        messages_filepath (str): Path to the messages CSV file
        categories_filepath (str): Path to the categories CSV file

    Returns:
        pd.DataFrame: Merged dataframe of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the merged dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing messages and categories

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = [col.split('-')[0] for col in categories.iloc[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df.drop(columns=['child_alone'])
    df.query('related != 2', inplace=True)           #making sure we have ONLY binary classification(has a 2 in its values)
    
    return df

def save_data(df: pd.DataFrame, database_filename: str):
    """
    Saving ourc cleaned dataframe to a SQLite database.

    Args:
        df (pd.DataFrame): Cleaned dataframe to be saved
        database_filename (str): Name of the SQLite database file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('chrismo', engine, index=False, if_exists='replace')

def main() -> None:
    """
    Main function to runs the entire ETL process.
    
    Expects three command-line arguments:
    1. Path to messages CSV file
    2. Path to categories CSV file
    3. Path to output SQLite database
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()