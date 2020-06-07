"""
A class to load the transactions into raw format and then transform it into an easy format for analysis and modelling.
The ability to profile the dataset is also included.
"""

import os
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class DataLoader:
    """
    Class to load the data and carry out transformations
    """

    def __init__(self, raw_path):
        self.raw_path = raw_path
        self.raw_data = None
        self.transformed_data = None

        try:
            self.load_json_data(raw_path)
        except (AttributeError, ValueError) as e:
            LOGGER.error(e)

    def load_json_data(self, path: Union[str, Path]):
        """
        Set raw_data instance variable
        Args:
            path: Union[str, Path] - path of json data
        """

        path = Path(path)  # coerce to pathlib.Path object
        if Path.exists(path):
            LOGGER.info(f'file found at path: {path}'
                        f'\nloading data as json... ')
            self.raw_data = pd.read_json(path, lines=True)
            LOGGER.info(f'data loaded')
        else:
            LOGGER.error(f'file not found at path: {path}')
            raise FileNotFoundError(f'file not found at path: {path}')

    def transform_raw_data(self, date_cols: list = None, drop_cols: bool = True) -> pd.DataFrame:
        """
        Transform raw data into dataframe suitable for analysis
        Args:
            date_col: list, name of column to coerce to datetime, if any
            drop_cols: bool, drop columns > XX% NaN values

        Returns: pd.DataFrame, transformed dataframe

        """
        LOGGER.info('transforming data...')
        df = self.raw_data

        # if set, coerce the datetime column(s) to a datetime format
        if date_cols:
            LOGGER.info(f'coercing columns to datetime format: {date_cols}')
            for date_col in date_cols:
                df[date_col] = pd.to_datetime(df[date_col])

        # replace blanks with np.nan
        LOGGER.info(f'replacing blanks with np.nan')
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # if set, drop cols with NaN > 90%
        if drop_cols:
            cols_to_drop = [column for column in df if df[column].count() / len(df) <= 0.1]
            df = df.drop(columns=cols_to_drop)
            LOGGER.info(f'df.columns dropped: {cols_to_drop}\n')
            LOGGER.info(f'df.columns remaining: \n {df.columns.values}')

        self.transformed_data = df
        LOGGER.info(f'returning transformed dataframe')

        return self.transformed_data

    def profile_dataset(self, out_path: Union[str, Path]):
        """
        Profile transformed dataset with pandas profiling
        Args:
            out_path: Union[str, Path], path to write output to
        """
        if self.transformed_data:
            try:
                profile = ProfileReport(self.transformed_data, title='Pandas Profiling Report')
                LOGGER.info(f'writing pandas profiling report to file: {out_path}')
                profile.to_file(out_path)  # recommend viewing on the .html file, preview below
            # ProfileReport can have system dependencies that cause it to fail unexpectedly
            # in general this is bad practice but necessary here
            except Exception as e:
                LOGGER.info(e)
        else:
            LOGGER.error('transformed data not set')


if __name__ == "__main__":

    # enables running in IDLE or as file
    if '__file__' in globals():
        project_dir = Path(__file__).resolve().parents[2]
    else:
        project_dir = ''

    data_path = project_dir + "data/raw/transactions.txt"
    date_col = ['transactionDateTime']

    # instantiate dataloader instance.
    dl = DataLoader(data_path)
    # transform and raw data
    dl.transform_raw_data(date_cols=date_col, drop_cols=True)
    # view
    print(dl.transformed_data.head())

    dl.profile_dataset(out_path="transactions_pandas_profiling.html")

