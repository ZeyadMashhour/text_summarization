from torchmetrics.text.rouge import ROUGEScore
rouge = ROUGEScore()
from pprint import pprint
import pandas as pd
from tqdm import tqdm


def calculate_efficiency(predicted_summary,original_summary):
    rouge = ROUGEScore()
    return rouge(predicted_summary, original_summary)


def rouge_scores_df(df,algorithm_summary_df):
    #compute rouge scores between df['Original Summary'] and algorithm_summary_df[algorithm_summary_df_name][i]
    sentences_efficiency = []
    length = len(df)
    algorithm_summary_df_name = algorithm_summary_df.columns[0]
    for i in tqdm(range(length)):
        predicted_summary = algorithm_summary_df[algorithm_summary_df_name][i]
        original_summary = df['Original Summary'][i]
        efficiency_dict = calculate_efficiency(predicted_summary,original_summary)
        sentences_efficiency.append(efficiency_dict)
            
    dataframe = pd.DataFrame(sentences_efficiency)
    return dataframe


def show_scores_df(summary_df, scores_df):
    return pd.concat([summary_df,scores_df],axis = 1)


def df_avg_by_column(df):
    row,col = df.shape
    return df.sum()/row


def get_avg_scores_df(reference_summary_dataset,system_summaries):
    """
    This gets avg_scores for all algorithms and their combinations
    """

    avg_scores = []
    combinations = list(system_summaries.columns)
    original_summary = pd.DataFrame(reference_summary_dataset["Original Summary"])
    #df = pd.DataFrame(system_summaries[combination])
    for combination in tqdm(combinations):
        combination_df = pd.DataFrame(system_summaries[combination])#create dataframe
        ensemble_scores = rouge_scores_df(df=original_summary, algorithm_summary_df=combination_df)
        combination_avg = df_avg_by_column(ensemble_scores)
        avg_scores.append(combination_avg)
    avg_scores_df = pd.DataFrame(avg_scores).T
    avg_scores_df.columns = combinations
    return avg_scores_df