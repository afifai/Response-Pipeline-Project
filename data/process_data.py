import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets.
    
    Args:
        messages_filepath (str): Filepath of the messages dataset.
        categories_filepath (str): Filepath of the categories dataset.
    
    Returns:
        df (pandas.DataFrame): Merged dataset containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge the datasets based on the 'id' column
    df = messages.merge(categories, on='id')
    
    # Split the values in the 'categories' column
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # Rename the columns
    categories.columns = category_colnames
    
    # Convert category values to numeric
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int).replace(2, 0)
        categories[column] = categories[column].apply(lambda x: 1 if x > 0 else 0)

    # Drop the original categories column and concatenate the modified one
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    return df


def clean_data(df):
    """
    Clean the merged dataset by removing duplicates.
    
    Args:
        df (pandas.DataFrame): Merged dataset.
    
    Returns:
        df (pandas.DataFrame): Cleaned dataset without duplicates.
    """
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset to a SQLite database.
    
    Args:
        df (pandas.DataFrame): Cleaned dataset.
        database_filename (str): Filename of the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disasterresponse', engine, index=False, if_exists='replace')


def main():
    """
    Main function that executes the data processing steps.
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
