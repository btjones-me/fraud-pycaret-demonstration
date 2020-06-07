"""
Define quick functions to be used by the notebooks
"""

import pandas as pd


def calculate_target_percentage_by_group(df_in, group, target='isFraud', filter_true=False):
    """
    Group_by on the group on our dataframe and find the count and relative % of fraud per group by category

    Args:
        df_in: pd.DataFrame
        group: str, categorical feature to groupby
        target: str, target_variable, defaults to 'isFraud'
        filter_true: filter on only target=True

    Returns:

    """

    df = df_in.copy()
    # calculate our metrics for each group
    df_count = df.groupby(group)[target].value_counts().rename('count').reset_index()
    df_percent = df.groupby(group)[target].value_counts(normalize='True').mul(100).rename('percent').reset_index()
    df_total = df.groupby(group).size().rename('total').reset_index()

    # join our metrics together
    df_out = pd.merge(df_count, df_total, left_on=group, right_on=group).merge(df_percent)
    # sort by percent
    df_out = df_out.sort_values(['percent'], ascending=[False])
    # filter on true only
    if filter_true:
        df_out = df_out[df_out[target] == True]

    return df_out


