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
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
    df_all['Team_Yds'] = df_all.groupby(['Tm', 'Year'])['Yds_pass'].transform('sum') + df_all.groupby(['Tm', 'Year'])['Yds_rush'].transform('sum')
    df_all['Team_Tgt'] = df_all.groupby(['Tm', 'Year'])['Tgt'].transform('sum')

    df_qb = df_all[df_all['FantPos'] == 'QB']
    df_qb = df_qb[['Player', 'Tm', 'Age', 'Yds_pass', 'Yds_rush', 'TD_pass', 'TD_rush', 'VBD', 'PPR', 'Year']]
    df_qb = df_qb.reset_index(drop=True)
    df_rb = df_all[df_all['FantPos'] == 'RB']
    df_rb = df_rb[['Player', 'Tm', 'Age', 'Att_rush', 'Tgt', 'TD_rush', 'TD_rec', 'Team_Yds', 'Team_Tgt', 'VBD', 'PPR', 'Year']]
    df_rb = df_rb.reset_index(drop=True)
    df_wr = df_all[df_all['FantPos'] == 'WR']
    #df_wr = df_wr[['Player', 'Tm', 'Age', 'Tgt', 'Rec', 'TD_rec', 'Yds_rec', 'Team_Yds', 'Team_Tgt', 'VBD', 'PPR', 'Year']]
    df_wr = df_wr.reset_index(drop=True)
    df_te = df_all[df_all['FantPos'] == 'TE']
    #df_te = df_te[['Player', 'Tm', 'Age', 'Tgt', 'TD_rec', 'Team_Yds', 'Team_Tgt', 'VBD', 'PPR', 'Year']]
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

    # Write the dataframes to csv files
    df_qb.to_csv('qb.csv', index=False)
    df_rb.to_csv('rb.csv', index=False)
    df_wr.to_csv('wr.csv', index=False)
    df_te.to_csv('te.csv', index=False)

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

    # Write the dataframe to a csv file
    df_def.to_csv('team_def.csv', index=False)

    return df_def

def redzone_scrape(begin_year, end_year):
    """
    This function scrapes https://www.pro-football-reference.com/years/ for redzone passing, rushing, receiving and defense data

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        A dataframe with the compiled redzone data
    """
    # Creating a for loop to go through passing, rushing and receiving data
    for y in ['passing', 'rushing', 'receiving']:
        if y == 'passing':
            column_headers = ['Tm', 'Cmp_20', 'Att_20', 'Cmp%_20', 'Yds_20', 'TD_20', 'Int_20', 'Cmp_10', 'Att_10', 'Cmp%_10', 'Yds_10', 'TD_10', 'Int_10', 'Highlights']
        elif y == 'rushing':
            column_headers = ['Tm', 'Att_20', 'Yds_20', 'TD_20', '%Rush_20', 'Att_10', 'Yds_10', 'TD_10', '%Rush_10', 'Att_5', 'Yrds_5', 'TD_5', '%Rush_5', 'Highlights']
        else:
            column_headers = ['Tm', 'Tgt_20', 'Rec_20', 'Ctch%_20', 'Yds_20', 'TD_20', '%Tgt_20', 'Tgt_10', 'Rec_10', 'Ctch%_10', 'Yds_10', 'TD_10', '%Tgt_10', 'Highlights']
        
        for x in range(begin_year, end_year + 1):
            url = 'https://www.pro-football-reference.com/years/' + str(x) + '/redzone-' + str(y) + '.htm'
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')
            table = soup.find('table', {'id': 'fantasy_rz'})
            table_head = table.find('thead')
            table_body = table.find('tbody')
            table_head_rows = table_head.find_all('tr')
            table_body_rows = table_body.find_all('tr')
            data_rows_player = [[td.getText() for td in table_body_rows[i].find_all('th')] for i in range(len(table_body_rows))]
            data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
            df = pd.DataFrame(data_rows, columns=column_headers[0:])
            df_player = pd.DataFrame(data_rows_player, columns=['Player'])
            df['Player'] = df_player['Player']
            df = df.dropna()
            df = df.replace('', 0)
            df = df.apply(pd.to_numeric, errors='ignore')
            df['Year'] = x
            if x == begin_year:
                df_name = df
            else:
                df_name = df_name.append(df)
            df_name['Tm'] = df_name['Tm'].replace('GNB', 'GB')
            df_name['Tm'] = df_name['Tm'].replace('KAN', 'KC')
            df_name['Tm'] = df_name['Tm'].replace('NOR', 'NO')
            df_name['Tm'] = df_name['Tm'].replace('SFO', 'SF')
            df_name['Tm'] = df_name['Tm'].replace('TAM', 'TB')
            df_name['Tm'] = df_name['Tm'].replace('LVR', 'LV')
            df_name['Tm'] = df_name['Tm'].replace('NWE', 'NE')
        if y == 'passing':
            df_rz_pass = df_name
        elif y == 'rushing':
            df_rz_rush = df_name
        else:   
            df_rz_rec = df_name

    # Write the dataframes to csv files
    df_rz_pass.to_csv('rz_pass.csv', index=False)
    df_rz_rush.to_csv('rz_rush.csv', index=False)
    df_rz_rec.to_csv('rz_rec.csv', index=False)

    return df_rz_pass, df_rz_rush, df_rz_rec

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

    # Getting rid of all special characters in the player names
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('*', '')
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('+', '')
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('\\', '')
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('\'', '')

    # Gettting the sum of OnTgt and Att per team and year
    df_qb_adv_team = df_qb_adv.groupby(['Tm', 'Year'])['OnTgt', 'Att'].sum()
    df_qb_adv_team = df_qb_adv_team.reset_index()
    df_qb_adv_team = df_qb_adv_team.rename(columns={'OnTgt': 'Team_OnTgt', 'Att': 'Team_Att'})
    df_qb_adv_team['Team_OnTgt_norm'] = df_qb_adv_team.groupby('Year')['Team_OnTgt'].transform(lambda x: (x - x.mean()) / x.std())

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
    # Setting player as the index
    df = df.set_index('Player')

    # Split the data into training and testing sets based on the year

    # Compiling the training data if a list of years is passed
    if type(df_training_year) == list:
        df_train = df[df['Year'].isin(df_training_year)]
    else:
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
    
def nfl_schedule_scrape(begin_year, end_year):
    """
    This function scraped the nfl schedule from https://thehuddle.com/2019/04/18/2019-nfl-schedule-team-by-week-grid/

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
        url = 'https://thehuddle.com/' + str(x) + '/04/18/' + str(x) + '-nfl-schedule-team-by-week-grid/'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'class':'table-responsive-inner'})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['TEAM', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9', 'week_10', 'week_11', 'week_12', 'week_13', 'week_14', 'week_15', 'week_16', 'week_17']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[0:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_schedule = df
        else:
            df_schedule = df_schedule.append(df)

def model_evaluation(df, df_testing_year, target, model):
    """
    This function evaluates a model by calculating r2 and mse. It will also create a df with the features and their coefficients

    Parameters:
    ----------
    df : DataFrame
        The dataframe to use for the model
    df_testing_year : int
        The year to use for testing
    target : str    
        The target variable
    model : object
        The model to use for the evaluation
    """
    # Setting player as the index
    df = df.set_index('Player')

    # Setting the training data to be all years except the testing year
    df_train = df[df['Year'] != df_testing_year]
    df_test = df[df['Year'] == df_testing_year]

    # Drop the year column
    df_train = df_train.drop('Year', axis=1)
    df_test = df_test.drop('Year', axis=1)

    # Drop rows that have NaN values
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    # Fit the model to the training data
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]

    # Create a pipeline for RandomForestRegressor
    pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor())])

    # Creating grid search parameters
    param_grid = [{'model': [RandomForestRegressor()], 
                    'model__n_estimators': [25, 50, 100, 300], 
                    'model__max_depth': [20, 30, 40],
                    'model__min_samples_split': [4, 5, 6],
                    'model__min_samples_leaf': [6, 8, 10],
                    'model__max_features': ['log2']}]
    
    # Create grid search object
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model
    model = grid.fit(X_train, y_train)
    
    # Make predictions on the testing data
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    y_pred = model.predict(X_test)

    # Calculate r2 and mse score
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Add the predicted values to the testing data frame
    df_test['Predicted'] = y_pred

    # Create a data frame with the features and their coefficients
    df_coef = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.best_estimator_.named_steps['model'].feature_importances_})

    # Print the best parameters
    best = grid.best_params_
    
    # Return the testing data frame and evaluation metrics
    return df_test, r2, mae, df_coef, best

def draft_scrape(begin_year, end_year):
    """
    This function will scrape https://www.pro-football-reference.com/years/2023/draft.htm for draft data

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        A dataframe with the compiled draft data
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/draft.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'id': 'drafts'})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['Pick', 'Team', 'Player', 'Position', 'Age', 'AP1', 'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Att', 'Yds_pass', 'TD_pass', 'Int_pass', 'Att_rush', 'Yds_rush', 'TD_rush', 'Rec', 'Yds_rec', 'TD_rec', 'Solo', 'Int_def', 'Sk', 'x', 'College/Univ', 'y']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[0:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        # Drop all columns that arent Pick, Team, Player, Position, College/Univ and Year
        df = df.drop(['Age', 'AP1', 'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Att', 'Yds_pass', 'TD_pass', 'Int_pass', 'Att_rush', 'Yds_rush', 'TD_rush', 'Rec', 'Yds_rec', 'TD_rec', 'Solo', 'Int_def', 'Sk', 'x', 'y'], axis=1)
        if x == begin_year:
            df_draft = df
        else:
            df_draft = df_draft.append(df)
        df['Team'] = df['Team'].replace('GNB', 'GB')
        df['Team'] = df['Team'].replace('KAN', 'KC')
        df['Team'] = df['Team'].replace('NOR', 'NO')
        df['Team'] = df['Team'].replace('SFO', 'SF')
        df['Team'] = df['Team'].replace('TAM', 'TB')
        df['Team'] = df['Team'].replace('LVR', 'LV')
        df['Team'] = df['Team'].replace('NWE', 'NE')
        
    # Write the dataframe to a csv file
    df_draft.to_csv('draft.csv', index=False)
