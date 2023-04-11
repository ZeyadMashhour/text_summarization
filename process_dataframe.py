# def extract_number(string_with_number):
#     """
#     This function removes alpahanumeric values and returns float
#     NOTE string must contain a number
#     """
#     if isinstance(string_with_number, str):
#         return float(''.join(filter(lambda x: x.isdigit() or x == '.', string_with_number)))
#     return float(string_with_number)


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


def get_max_values(df):
    """
    This function return the max values in every row of a data frame
    note this function is used in Hamed format
    """
    #we dun take the first 2 columns because they are the originals
    max_values = df.iloc[:, 3:].max(axis=1)
    return max_values


def create_dataframe(df, max_values):
    """
    
    """
    output_list = []
    for i in range(len(df.index)):
        output_list.append([df.index[i]
            ,max_values[i]
            ,df.columns[output_list(np.where(df.loc[df.index[i]] == max_values[i]))[0][0]]])
    columns_names= ["metric", "max_score", "algorithm"]
    output_dataframe = pd.DataFrame(output_list, columns=columns_names)
    return output_dataframe
