# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

import pandas as pd
import numpy as np

def JointHistogram(df, findex, **kwargs):
    """
    Compute joint histograms using pandas by counting unique rows and their frequencies.
    Also returns a list of indices mapping each row in the input dataframe to the unique_rows dataframe if compute_idx is True.
    
    Parameters:
    - df: pandas DataFrame
    - findex: list of column indices to compute the joint histogram on
    - **kwargs: keyword arguments; specify compute_idx=True if idx is desired as output

    Returns:
    - unique_rows: DataFrame of unique rows in the specified columns
    - counts: Series with frequencies of each unique row
    - idx (optional): List of indices mapping each row in df to the row in unique_rows
    """
    # Subset the DataFrame to only include columns specified by findex
    subset_df = df.iloc[:, findex]
    subset_df = pd.DataFrame(subset_df)
    # Group by unique rows and count frequencies
    counts_series = subset_df.groupby(list(subset_df.columns)).size()
    # counts_df = subset_df.groupby(list(subset_df.columns)).size().reset_index(name='Frequency')
    counts = counts_series.values
    unique_rows = counts_series.index.to_frame(index=False)
    # If compute_idx is True, map each row in subset_df to its corresponding row index in unique_rows
    if kwargs.get('compute_idx', False):
        merged_df = pd.merge(subset_df, counts, 
                             on=subset_df.columns.tolist(), how='left')
        idx = merged_df['index'].tolist()
        return unique_rows, counts, idx
    else:
        return unique_rows, counts
    
def JointEntropy(df, findex):
    unique_rows, counts = JointHistogram(df, findex, compute_idx=False)
    pf = counts/df.shape[0]
    hf = -pf*np.log2(pf)
    Hf = sum(hf)
    
    return Hf
