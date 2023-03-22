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
    rows, column = algorithm_summary_df.shape
    for row in tqdm(range(rows)):
        predicted_summary = algorithm_summary_df.iloc[row,0]
        original_summary = df.iloc[row,1]
        efficiency_dict = calculate_efficiency(predicted_summary,original_summary)
        sentences_efficiency.append(efficiency_dict)
              
    dataframe = pd.DataFrame(sentences_efficiency)
    return dataframe


def show_scores_df(summary_df, scores_df):
    return pd.concat([summary_df,scores_df],axis = 1)


def df_avg_by_column(df):
    row,col = df.shape
    return df.sum()/row