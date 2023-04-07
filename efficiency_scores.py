from torchmetrics.text.rouge import ROUGEScore
rouge = ROUGEScore()
from pprint import pprint
import pandas as pd
from tqdm import tqdm


def calculate_efficiency(predicted_summary,original_summary):
    rouge = ROUGEScore()
    return rouge(predicted_summary, original_summary)


def rouge_scores_df(df,algorithm_summary_df):
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