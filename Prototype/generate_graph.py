import os
import random
import datetime
import matplotlib
import pandas as pd
matplotlib.use('Agg') # python framework problem
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc # candlestick chart


def graph_section(df, catagory, filename, path, l_period):
    """
    :param df: the section of dataframe to be graphed
    :param catagory: the category/label that the image belong to
    :param filename: a unique string to identify each file
    :param path: the path to save the images
    :param l_period: period of time in this graph
    :return: graphs saved in /train/label or /test/label
    """
    # copy the dataset
    ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()

    # create subplot
    f1, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    # plot the trendline
    ax.plot(df['Date'], df['Close'],c = "k", linewidth= 1.0)

    # plot the candlesticks
    candlestick_ohlc(ax, ohlc.values, width=.6, colorup='grey', colordown='black')

    # save
    if random.randint(1, 101) < 75: # change this value to control the training-testing split
        func = "train"
    else:
        func = "test"
    directory = path + func + "/" + catagory
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = directory + "/" + filename + ".png"
    plt.savefig(filepath,bbox_inches='tight')
    plt.close('all')

def catagorizer(x, criteria):
    """
    :param x: a float that passed in to test against the criteris
    :param criteria: user defined dictionary with keys as label names and
           values as a list of cutoffs e.g. [0.1,0.2] between -1 and 1
    :return: the label that x belongs to based on the criteria
    """
    for i in criteria:
        if x > criteria[i][0] and x <= criteria[i][1]:
            return i
        else:
            pass


def make_data(df, l_period, p_period, criteria):
    """
    :param df: the df that we hope to segment into different graphs
    :param l_period: learning period. look forward how long
    :param p_period: prediction period. the difference of the time periodo that we want to predict
    :param criteria: user defined dictionary with keys as label names and
           values as a list of cutoffs e.g. [0.1,0.2] between -1 and 1
    :return: graphs saved in /train/label or /test/label
    """
    #print(l_period + 1, df.shape[0], p_period)
    if df.shape[0] < l_period + 1:
        raise ValueError('Insufficient number of rows.')
    for i in range(l_period + 1, df.shape[0] - p_period, p_period):
        diff = (df.iloc[i + p_period]["Close"] - df.iloc[i]["Close"]) / float(df.iloc[i]["Close"])
        cata_name = catagorizer(diff, criteria)
        print(round(float(i)/ (df.shape[0] - p_period),2), cata_name)
        graph_section(df.iloc[(i - l_period - 1):(i - 1)], cata_name, str(df.iloc[i]["Date"]), "Output/", l_period)

if __name__ == '__main__':

    data = pd.read_csv("rawdata/aapl_us_d.csv")
    stock_price_df = data
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df["Date"] = stock_price_df["Date"].apply(mdates.date2num)

    l_period = 20
    p_period = 5
    criteria = {
        "down": [-1, -.05],
        "same": [-.05, 0.04],
        "up": [.04, 1]
    }

    # running batches to avoid memory
    df = stock_price_df
    make_data(df, l_period, p_period, criteria)

