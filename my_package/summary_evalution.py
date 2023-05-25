from torchmetrics.text.rouge import ROUGEScore
rouge = ROUGEScore()
from pprint import pprint
import pandas as pd
from tqdm import tqdm


def calculate_efficiency(predicted_summary: str, original_summary: str) -> dict:
    """
    Calculate the efficiency score between the predicted summary and the original summary
    using the ROUGE metric.

    :param predicted_summary: The predicted summary.
    :type predicted_summary: str
    :param original_summary: The original summary.
    :type original_summary: str
    :return: A dictionary with the efficiency score for ROUGE-1, ROUGE-2, and ROUGE-L.
    :rtype: dict
    """
    rouge_score = ROUGEScore()
    efficiency_dict = rouge_score(predicted_summary, original_summary)
    return efficiency_dict

def refined_calculate_efficiency(predicted_summary: str, original_summary: str) -> dict:
    """
    Calculate the efficiency score between the predicted summary and the original summary
    using the ROUGE metric.

    :param predicted_summary: The predicted summary.
    :type predicted_summary: str
    :param original_summary: The original summary.
    :type original_summary: str
    :return: A dictionary with the efficiency score for ROUGE-1, ROUGE-2, and ROUGE-L.
    :rtype: dict
    """
    rouge_score = ROUGEScore()
    efficiency_dict = rouge_score(remove_escape_characters(predicted_summary), remove_escape_characters(original_summary))
    return efficiency_dict



def calculate_rouge_scores_df(reference_df, system_df, reference_column='Original Summary', system_column=None):
    """
    Calculate the ROUGE scores for each pair of reference and system summaries.
    
    Parameters:
        reference_df (pandas.DataFrame): The DataFrame containing the reference summaries.
        system_df (pandas.DataFrame): The DataFrame containing the system-generated summaries.
        reference_column (str): The name of the column in the reference_df containing the reference summaries.
        system_column (str): The name of the column in the system_df containing the system-generated summaries. If None,
            the function will assume that the system_df has only one column containing the summaries.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the ROUGE scores for each pair of reference and system summaries.
    """
    if system_column is None:
        system_column = system_df.columns[0]
    
    rouge = ROUGEScore()
    rouge_scores = []
    for i in tqdm(range(len(reference_df))):
        reference_summary = reference_df[reference_column][i]
        system_summary = system_df[system_column][i]
        scores = rouge(reference_summary, system_summary)
        rouge_scores.append(scores)
    
    return pd.DataFrame(rouge_scores)

# <<<<<<< HEAD
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

# =======
# >>>>>>> 5ac36fab0da9ef6ec6dbdc4e06b88f8f01951783


def calculate_average_by_column_df(df):
    """
    Calculates the average of each column in the dataframe
    
    Parameters:
        df (pandas.DataFrame): The dataframe.
    
    Returns:
        pandas.Series: A series containing the average of each column.
    """
    return df.mean()




def get_average_scores_df(reference_summary_dataset, system_summaries):
    """
    Calculates average scores for all algorithms and their combinations.
    
    Parameters:
        reference_summary_dataset (pandas.DataFrame): The reference summary dataset.
        system_summaries (pandas.DataFrame): The system summaries.
    
    Returns:
        pandas.DataFrame: A dataframe containing the average scores.
    """
    avg_scores = []
    combinations = list(system_summaries.columns)
    original_summary = pd.DataFrame(reference_summary_dataset["Original Summary"])

    for combination in tqdm(combinations):
        combination_df = pd.DataFrame(system_summaries[combination])
        while True:
            try:
                ensemble_scores = calculate_rouge_scores_df(df=original_summary, algorithm_summary_df=combination_df)
                combination_avg = calculate_average_by_column_df(ensemble_scores)
                avg_scores.append(combination_avg)
                break
            except Exception as e:
                print(f"Error occurred in function: {e}")
                print("Retrying the function...")

    avg_scores_df = pd.DataFrame(avg_scores).T
    avg_scores_df.columns = combinations
    return avg_scores_df


def remove_escape_characters(data):
    """
    Removes all escaped characters, including newline characters, from text, a list, or a list of lists.

    Args:
    - data (str, bytes, list, or list of lists): The data to remove escaped characters from.

    Returns:
    - The data with escaped characters removed, in the same format as the input.
    """
    # Helper function to remove escaped characters from a single string
    def remove_escape_characters_single_string(s):
        return re.sub(r'(\\[rnt])+|[\r\n]+', ' ', s)

    # Handle different data types
    if isinstance(data, (str, bytes)):
        # Decode bytes to string if necessary
        if isinstance(data, bytes):
            data = data.decode()

        # Remove escaped characters from a single string
        return remove_escape_characters_single_string(data)
    elif isinstance(data, list):
        # Remove escaped characters from a list of strings or a list of lists of strings
        cleaned_data = []
        for item in data:
            cleaned_item = remove_escape_characters(item)
            cleaned_data.append(cleaned_item)
        return cleaned_data
    else:
        raise ValueError("Invalid input type. The data should be a string, bytes, list, or list of lists.")