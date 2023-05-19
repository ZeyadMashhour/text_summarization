import pandas as pd
import numpy as np
<<<<<<< Updated upstream


=======
>>>>>>> Stashed changes
def extract_number(string_with_number):
    """
    This function removes alpahanumeric values and returns float
    NOTE string must contain a number
    """
    if isinstance(string_with_number, str):
        numeric_string  = ''.join(filter(lambda x: x.isdigit() or x == '.', string_with_number))
        if numeric_string.isdigit():
            return float(numeric_string)
        else:
            try:
                output_float = float(numeric_string)
                return output_float
            except ValueError:
                return string_with_number
    return float(string_with_number)

def extracted_number_from_df(df):
    """
    This function runs on a dataframe and removes alaphanumeric values
    NOTE all cells must contain a number in the dataframe
    """
    for column_name in df:
        for i,cell in enumerate(df[column_name]):
            df[column_name][i] = extract_number(cell)
    return df


def get_max_values(df, remove_first_two_columns=False):
    """
    This function returns the maximum values in every row of a DataFrame.
    It optionally removes the first two columns if the `remove_first_two_columns` parameter is True.
    """
    if remove_first_two_columns:
        max_values = df.iloc[:, 2:].max(axis=1)
    else:
        max_values = df.iloc[:, 3:].max(axis=1)
    
    return max_values



def create_max_values_dataframe(df, max_values):
    """
    This function create dataframe of max_values of each row
    """
    output_list = []
    for i in range(len(df.index)):
        output_list.append([df.index[i]
            ,max_values[i]
            ,df.columns[list(np.where(df.loc[df.index[i]] == max_values[i]))[0][0]]])
            #[0][0] refers to accessing output inside a np array 
    columns_names= ["metric", "max_score", "algorithm"]
    output_dataframe = pd.DataFrame(output_list, columns=columns_names)
    return output_dataframe

def get_sorted_values(df, remove_first_two_columns=False):
    """
    This function returns a dictionary of DataFrames, where each key is an
    index and the corresponding value is a DataFrame containing the sorted
    column values for that index.
    It optionally removes the first two columns if the `remove_first_two_columns` parameter is True.
    """
    if remove_first_two_columns:
        values_df = df.iloc[:, 2:]
    else:
        values_df = df.iloc[:, 3:]
    
    max_values = values_df.max(axis=1)
    sorted_dict = {}
    
    for i, row in df.iterrows():
        sorted_row = row.iloc[2:].sort_values(ascending=False)
        sorted_dict[i] = pd.DataFrame({'value': sorted_row.values}, index=sorted_row.index)
    
    sorted_dict = {k: v for k, v in sorted(sorted_dict.items(), key=lambda x: max_values[x[0]], reverse=True)}
    return sorted_dict



def concatenate_dataframes(df1, df2):
    """
    Concatenates two dataframes along the columns axis
    
    Parameters:
        df1 (pandas.DataFrame): The first dataframe.
        df2 (pandas.DataFrame): The second dataframe.
        
    Returns:
        pandas.DataFrame: The concatenated dataframe.
    """
    return pd.concat([df1, df2], axis=1)