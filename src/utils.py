import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import dill
import numpy as np
import pickle
from dotenv import load_dotenv
import pymysql


###load_dotenv()

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
'''
host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
database = os.getenv('db')


def read_sql_data():
    logging.info("Reading SQL database started")

    try:
        mydb = pymysql.connect(
            host = host,
            user = user,
            password = password,
            db = database
        )
        logging.info("Connection Established")
        df = pd.read_sql_query('Select * from students', mydb)
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex)    

        '''