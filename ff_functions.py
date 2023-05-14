# Description: This program will be a library of functions to be used in Fantasy Football Player Valuation.py
#
# Import libraries

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

def player_scrape(begin_year, end_year):
    """
    This function will scrape data from pro-football-reference.com, loop through the begin_year to end_year seasons
    and returns a 4 dataframe with the data. The dataframes are qb_df, rb_df, wr_df, and te_df.

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int 
        The last year to scrape data from

    Returns:
    -------
    dataframe
        4 dataframes with the data scraped from pro-football-reference.com
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/fantasy.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'id': 'fantasy'})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['Rk', 'Player', 'Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'Att_pass', 'Yds_pass', 'TD_pass', 'Int_pass', 'Att_rush', 'Yds_rush', 'Y/A', 'TD_rush', 'Tgt', 'Rec', 'Yds_rec', 'Y/R', 'TD_rec', 'Fmb', 'FL', 'TD_total', '2PM', '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[1:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_all = df
        else:
            df_all = df_all.append(df)
    df_all['Tm'] = df_all['Tm'].replace('GNB', 'GB')
    df_all['Tm'] = df_all['Tm'].replace('KAN', 'KC')
    df_all['Tm'] = df_all['Tm'].replace('NOR', 'NO')
    df_all['Tm'] = df_all['Tm'].replace('SFO', 'SF')
    df_all['Tm'] = df_all['Tm'].replace('TAM', 'TB')
    df_all['Tm'] = df_all['Tm'].replace('LVR', 'LV')
    df_all['Tm'] = df_all['Tm'].replace('NWE', 'NE')
    df_all = df_all.apply(pd.to_numeric, errors='ignore')

    # Grouping data by team and year to get sum of TD and Tgt
    df_all['Team_TD'] = df_all.groupby(['Tm', 'Year'])['TD_total'].transform('sum')
    df_all['Team_Tgt'] = df_all.groupby(['Tm', 'Year'])['Tgt'].transform('sum')

    df_qb = df_all[df_all['FantPos'] == 'QB']
    df_qb = df_qb[['Player', 'Tm', 'Age', 'Yds_pass', 'Yds_rush', 'TD_pass', 'TD_rush', 'VBD', 'PPR', 'Year']]
    df_qb = df_qb.reset_index(drop=True)
    df_rb = df_all[df_all['FantPos'] == 'RB']
    df_rb = df_rb[['Player', 'Tm', 'Age', 'Att_rush', 'Tgt', 'TD_rush', 'TD_rec', 'Team_TD', 'Team_Tgt', 'VBD', 'PPR', 'Year']]
    df_rb = df_rb.reset_index(drop=True)
    df_wr = df_all[df_all['FantPos'] == 'WR']
    df_wr = df_wr[['Player', 'Tm', 'Age', 'Tgt', 'TD_rec', 'Team_TD', 'Team_Tgt', 'VBD', 'PPR', 'Year']]
    df_wr = df_wr.reset_index(drop=True)
    df_te = df_all[df_all['FantPos'] == 'TE']
    df_te = df_te[['Player', 'Tm', 'Age', 'Tgt', 'TD_rec', 'Team_TD', 'Team_Tgt', 'VBD', 'PPR', 'Year']]
    df_te = df_te.reset_index(drop=True)

    # Removing the symbols "*" and "+" from the player names using a loop
    for i in range(len(df_qb)):
        df_qb['Player'][i] = df_qb['Player'][i].replace('*', '')
        df_qb['Player'][i] = df_qb['Player'][i].replace('+', '')
    for i in range(len(df_rb)):
        df_rb['Player'][i] = df_rb['Player'][i].replace('*', '')
        df_rb['Player'][i] = df_rb['Player'][i].replace('+', '')
    for i in range(len(df_wr)):
        df_wr['Player'][i] = df_wr['Player'][i].replace('*', '')
        df_wr['Player'][i] = df_wr['Player'][i].replace('+', '')
    for i in range(len(df_te)):
        df_te['Player'][i] = df_te['Player'][i].replace('*', '')
        df_te['Player'][i] = df_te['Player'][i].replace('+', '')

    return df_qb, df_rb, df_wr, df_te

def team_def_scrape(begin_year, end_year):
    """
    This function will scrape data from pro-football-reference.com for team defense data, loop through the begin_year to end_year seasons
    and returns a dataframe with the data

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from
    
    Returns:
    -------
    dataframe
        A dataframe with the data scraped from pro-football-reference.com
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/opp.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table')
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['Rk', 'Tm', 'G', 'PF', 'Yds', 'Ply', 'Y/P', 'TO', 'FL', '1stD', 'Cmp', 'Att', 'Yds_pass', 'TD_pass', 'Int_pass', 'NY/A', '1stD_pass', 'Att_rush', 'Yds_rush', 'TD_rush', 'Y/A', '1stD_rush', 'Pen', 'Yds_pen', '1stPy', 'Sc%', 'TO%', 'EXP']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[1:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_def = df
        else:
            df_def = df_def.append(df)
    df_def['Tm'] = df_def['Tm'].replace('Los Angeles Rams', 'LAR')
    df_def['Tm'] = df_def['Tm'].replace('Los Angeles Chargers', 'LAC')
    df_def['Tm'] = df_def['Tm'].replace('New England Patriots', 'NE')
    df_def['Tm'] = df_def['Tm'].replace('New Orleans Saints', 'NO')
    df_def['Tm'] = df_def['Tm'].replace('New York Giants', 'NYG')
    df_def['Tm'] = df_def['Tm'].replace('New York Jets', 'NYJ')
    df_def['Tm'] = df_def['Tm'].replace('San Francisco 49ers', 'SF')
    df_def['Tm'] = df_def['Tm'].replace('Tampa Bay Buccaneers', 'TB')
    df_def['Tm'] = df_def['Tm'].replace('Tennessee Titans', 'TEN')
    df_def['Tm'] = df_def['Tm'].replace('Washington Football Team', 'WSH')
    df_def['Tm'] = df_def['Tm'].replace('Arizona Cardinals', 'ARI')
    df_def['Tm'] = df_def['Tm'].replace('Atlanta Falcons', 'ATL')
    df_def['Tm'] = df_def['Tm'].replace('Baltimore Ravens', 'BAL')
    df_def['Tm'] = df_def['Tm'].replace('Buffalo Bills', 'BUF')
    df_def['Tm'] = df_def['Tm'].replace('Carolina Panthers', 'CAR')
    df_def['Tm'] = df_def['Tm'].replace('Chicago Bears', 'CHI')
    df_def['Tm'] = df_def['Tm'].replace('Cincinnati Bengals', 'CIN')
    df_def['Tm'] = df_def['Tm'].replace('Cleveland Browns', 'CLE')
    df_def['Tm'] = df_def['Tm'].replace('Dallas Cowboys', 'DAL')
    df_def['Tm'] = df_def['Tm'].replace('Denver Broncos', 'DEN')
    df_def['Tm'] = df_def['Tm'].replace('Detroit Lions', 'DET')
    df_def['Tm'] = df_def['Tm'].replace('Green Bay Packers', 'GB')
    df_def['Tm'] = df_def['Tm'].replace('Houston Texans', 'HOU')
    df_def['Tm'] = df_def['Tm'].replace('Indianapolis Colts', 'IND')
    df_def['Tm'] = df_def['Tm'].replace('Jacksonville Jaguars', 'JAX')
    df_def['Tm'] = df_def['Tm'].replace('Kansas City Chiefs', 'KC')
    df_def['Tm'] = df_def['Tm'].replace('Miami Dolphins', 'MIA')
    df_def['Tm'] = df_def['Tm'].replace('Minnesota Vikings', 'MIN')
    df_def['Tm'] = df_def['Tm'].replace('Oakland Raiders', 'LV')
    df_def['Tm'] = df_def['Tm'].replace('Philadelphia Eagles', 'PHI')
    df_def['Tm'] = df_def['Tm'].replace('Pittsburgh Steelers', 'PIT')
    df_def['Tm'] = df_def['Tm'].replace('Seattle Seahawks', 'SEA')
    df_def['Tm'] = df_def['Tm'].replace('Washington Commanders', 'WSH')
    df_def['Tm'] = df_def['Tm'].replace('Las Vegas Raiders', 'LV')
    df_def['Tm'] = df_def['Tm'].replace('San Diego Chargers', 'LAC')
    df_def['Tm'] = df_def['Tm'].replace('St. Louis Rams', 'LAR')
    df_def['Tm'] = df_def['Tm'].replace('Washington Redskins', 'WSH')

    df_def = df_def[['Tm','EXP', 'TO', 'NY/A', 'Y/A', 'Year']]

    # Normalize the data
    df_def['Pass_QB_def_norm'] = df_def.groupby('Year')['NY/A'].transform(lambda x: (x - x.mean()) / x.std())
    df_def['Rush_def_norm'] = df_def.groupby('Year')['Y/A'].transform(lambda x: (x - x.mean()) / x.std())

    return df_def

def nfl_schedule(begin_year, end_year):
    """
    This function looks in the file path for xlsx files that meet the criteria and compiles them into one dataframe

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        A dataframe with the compiled schedule data
    """
    for x in range(begin_year, end_year + 1):
        df = pd.read_excel('nfl_schedule_' + str(x) + '.xlsx')
        df['Year'] = x
        if x == begin_year:
            df_schedule = df
        else:
            df_schedule = df_schedule.append(df)
    return df_schedule

def qb_adv_stats(begin_year, end_year):
    """
    This function scrapes https://www.pro-football-reference.com/years/2022/passing_advanced.htm for advanced stats for QBs

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        2 dataframes, one with qb advanced stats and one with team qb advanced stats
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/passing_advanced.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'id': 'advanced_accuracy'})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['Player', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'Bats', 'ThAwy', 'Spikes', 'Drops', 'Drop%', 'BadTh', 'Bad%', 'OnTgt', 'OnTgt%']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[0:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_qb_adv = df
        else:
            df_qb_adv = df_qb_adv.append(df)
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('GNB', 'GB')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('KAN', 'KC')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('NOR', 'NO')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('SFO', 'SF')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('TAM', 'TB')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('LVR', 'LV')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('NWE', 'NE')
    df_qb_adv = df_qb_adv[df_qb_adv['Pos'] == 'QB']
    # Gettting the sum of OnTgt and Att per team and year
    # Higher the number the more accurate passes the team has
    df_qb_adv['Team_OnTgt_sum'] = df_qb_adv.groupby(['Tm', 'Year'])['OnTgt'].transform('sum')
    df_qb_adv['Team_OnTgt_norm'] = df_qb_adv.groupby('Year')['Team_OnTgt_sum'].transform(lambda x: (x - x.mean()) / x.std())

    # Creating a new dataframe from df_qb_adv that is grouped by team and year with the Team_OnTgt_norm column
    df_qb_adv_team = df_qb_adv.groupby(['Tm', 'Year'])['Team_OnTgt_norm'].mean().reset_index()

    return df_qb_adv, df_qb_adv_team

def evaluate_model(df, df_testing_year, df_training_year, target, model):
    """
    This function evaluates a model by calculating r2 and mse. It will also create a df with the features and their coefficients

    Parameters:
    ----------
    df : DataFrame
        The dataframe to use for the model
    df_testing_year : int
        The year to use for testing
    df_training_year : int
        The year to use for training
    target : str    
        The target variable
    model : object
        The model to use for the evaluation
    """
    # Split the data into training and testing sets based on the year
    df_train = df[df['Year'] == df_training_year]
    df_test = df[df['Year'] == df_testing_year]
    
    # Fit the model to the training data
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    y_pred = model.predict(X_test)

    # Calculate r2 and mse score
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Add the predicted values to the testing data frame
    df_test['Predicted'] = y_pred

    # Create a data frame with the features and their coefficients
    df_coef = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.feature_importances_})
    
    # Return the testing data frame and evaluation metrics
    return df_test, r2, mse, df_coef


#testing the function
# q: how to get r2 score?
# a: https://stackoverflow.com/questions/42324419/getting-r2-score-in-cross-validation-in-scikit-learn
