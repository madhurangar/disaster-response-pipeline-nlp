import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath:str, categories_filepath:str)->pd.DataFrame:
    """load messages and categories and merge them

    Args:
        messages_filepath (str): file path to messages
        categories_filepath (str): file path to categories

    Returns:
        pd.DataFrame: merged dataframe
    """
    # load csv and merge datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df:pd.DataFrame)->pd.DataFrame:
    """clean data: 
     - convert category values to 0 and 1
     - add category names to dataframe column names
     - remove duplicates

    Args:
        df (pd.DataFrame): original data: messages and labels

    Returns:
        pd.DataFrame: cleaned data
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # slice category names 
    category_colnames = row.apply(lambda x: x.split("-")[0]).values
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to numbers 0 and 1
    categories = categories.apply(lambda col: col.apply(lambda cell:int(cell.split("-")[1])))
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)  
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1) 
    
    # removing '2' from 'related' column. data must be binary!
    df.drop(df[df['related'] == 2].index, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(keep='first',  inplace=True)
    
    return df


def save_data(df:pd.DataFrame, database_filename:str)->None:
    """save cleaned dataframe as a SQL database

    Args:
        df (pd.DataFrame): cleaned data
        database_filename (str): SQL datasase file name
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False, if_exists="replace")  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {database_filepath}')
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