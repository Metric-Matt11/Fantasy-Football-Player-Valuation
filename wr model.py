# Description: This program will set a value for each player 
#              based on their performance in the previous 3 season.
#              The idea for this model is to create a ranking for a fantasy football draft.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import ff_functions as ff

# Pulling wr, qb and rec redzone data from csv files
df_wr = pd.read_csv('wr.csv')
df_qb = pd.read_csv('qb.csv')
df_rz_rec = pd.read_csv('rz_rec.csv')
df_signings = pd.read_excel('FA_Signings.xlsx')
df_def = pd.read_csv('team_def.csv')
df_draft = pd.read_csv('draft.csv')

# Create a df that has all teams for all years as a columns and has a flag for each position and if the team had a top 10 pick in that postion
df_draft = df_draft[df_draft['Pick'] <= 10]
df_draft = df_draft[~df_draft['Position'].isin(['CB', 'DB', 'DE', 'DT', 'G', 'ILB', 'LB', 'OL', 'S', 'T'])]
df_draft = df_draft.groupby(['Team', 'Year', 'Position']).count().reset_index()
df_draft = df_draft.pivot(index=['Team', 'Year'], columns='Position', values='Player')
df_draft = df_draft.reset_index()
df_draft = df_draft.fillna(0)
df_draft['QB_top10'] = np.where(df_draft['QB'] > 0, 1, 0)
df_draft['RB_top10'] = np.where(df_draft['RB'] > 0, 1, 0)
df_draft['WR_top10'] = np.where(df_draft['WR'] > 0, 1, 0)
df_draft['TE_top10'] = np.where(df_draft['TE'] > 0, 1, 0)
df_draft = df_draft.drop(columns=['QB', 'RB', 'WR', 'TE'])

df_wr = df_wr.merge(df_draft[['Team', 'Year', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']], how='left', left_on=['Tm', 'Year'], right_on=['Team', 'Year'])
df_wr = df_wr.merge(df_rz_rec[['Player', 'Year', 'Yds_20', 'Yds_10']], how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'])
df_wr = df_wr.merge(df_signings[['Tm', 'Year', 'Tier_1_Sign']], how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_wr = df_wr.merge(df_def[['Tm', 'Year', 'Pass_QB_def_norm', 'Rush_def_norm']], how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_wr['Tgt_share'] = df_wr['Tgt'] / df_wr['Team_Tgt']

# Join the qb data to the wr data on Team and Year and only pull in the PPR column
# put qb_ as a prefix for all columns in the qb dataframe
# Only join QB with the biggers PPR value
df_qb = df_qb.groupby(['Tm', 'Year']).max().reset_index()
df_qb = df_qb.add_prefix('qb_')
df_wr = df_wr.merge(df_qb[['qb_Tm', 'qb_Year', 'qb_PPR']], how='left', left_on=['Tm', 'Year'], right_on=['qb_Tm', 'qb_Year'])
df_wr = df_wr.drop(columns=['qb_Tm', 'qb_Year'])

# Getting a count of number of years a player has played in the NFL (excluding 2022)
df_wr['Year_count'] = df_wr.groupby('Player')['Year'].transform('count') - 1

# Dropping all players with less than 3 years of data
df_wr = df_wr[df_wr['Year_count'] >= 3]

# Shifting the data for each player by 1 year, changing all columns to have a _prev suffix
df_wr = df_wr.sort_values(by=['Player', 'Year'])
df_wr = df_wr.reset_index(drop=True)
df_wr = df_wr.rename(columns={'Player': 'Player'})
df_wr['Tgt_prev'] = df_wr.groupby('Player')['Tgt'].shift(1)
df_wr['Tgt_share_prev'] = df_wr.groupby('Player')['Tgt_share'].shift(1)
#df_wr['Tgt_share_2yr_prev'] = df_wr.groupby('Player')['Tgt_share'].shift(2)
#df_wr['Yds_rec/Att_prev'] = df_wr.groupby('Player')['Yds_rec/Att'].shift(1)
df_wr['TD_rec_prev'] = df_wr.groupby('Player')['TD_rec'].shift(1)
df_wr['Yds_20_prev'] = df_wr.groupby('Player')['Yds_20'].shift(1)
df_wr['Yds_10_prev'] = df_wr.groupby('Player')['Yds_10'].shift(1)
df_wr['Yds_rec_prev'] = df_wr.groupby('Player')['Yds_rec'].shift(1)
df_wr['Yds_rec_2yr_prev'] = df_wr.groupby('Player')['Yds_rec'].shift(2)
df_wr['Y/R_prev'] = df_wr.groupby('Player')['Y/R'].shift(1)
df_wr['Fmb_prev'] = df_wr.groupby('Player')['Fmb'].shift(1)
df_wr['TD_rec_prev'] = df_wr.groupby('Player')['TD_rec'].shift(1)
df_wr['G_prev'] = df_wr.groupby('Player')['G'].shift(1)
df_wr['GS_prev'] = df_wr.groupby('Player')['GS'].shift(1)
df_wr['PPR_prev'] = df_wr.groupby('Player')['PPR'].shift(1)
df_wr['PPR_2yr_prev'] = df_wr.groupby('Player')['PPR'].shift(2)
df_wr['PosRank_prev'] = df_wr.groupby('Player')['PosRank'].shift(1)
df_wr['PosRank_2yr_prev'] = df_wr.groupby('Player')['PosRank'].shift(2)
df_wr['OvRank_prev'] = df_wr.groupby('Player')['OvRank'].shift(1)
df_wr['VBD_prev'] = df_wr.groupby('Player')['VBD'].shift(1)
df_wr['FantPt_prev'] = df_wr.groupby('Player')['FantPt'].shift(1)
df_wr['DKPt_prev'] = df_wr.groupby('Player')['DKPt'].shift(1)
df_wr['FDPt_prev'] = df_wr.groupby('Player')['FDPt'].shift(1)
df_wr['qb_PPR_prev'] = df_wr.groupby('Player')['qb_PPR'].shift(1)
df_wr['Team_Off_prev'] = df_wr.groupby('Player')['Team_Yds'].shift(1)
df_wr = df_wr.fillna(0)

# Recursively looking at each players history in Yds_rec and creating a historical average for each player excluding the current and prior year data
df_wr['Yds_rec_historical_avg'] = df_wr.groupby('Player')['Yds_rec'].transform(lambda x: x.expanding().mean().shift(2))
#df_wr['GS_historical_avg'] = df_wr.groupby('Player')['GS'].transform(lambda x: x.expanding().mean().shift(2))
df_wr['PosRank_historical_avg'] = df_wr.groupby('Player')['PosRank'].transform(lambda x: x.expanding().mean().shift(2))
df_wr['PPR_historical_avg'] = df_wr.groupby('Player')['PPR'].transform(lambda x: x.expanding().mean().shift(2))
#df_wr['Tgt_historical_avg'] = df_wr.groupby('Player')['Tgt'].transform(lambda x: x.expanding().mean().shift(2))
#df_wr['Tgt_share_historical_avg'] = df_wr.groupby('Player')['Tgt_share'].transform(lambda x: x.expanding().mean().shift(2))

# Dropping all columns that dont have a _prev suffix except for Player, Year and PPR
# Use a loop
for col in df_wr.columns:
    if '_prev' not in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Tier_1_Sign', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg']:
        df_wr = df_wr.drop(columns=col)

# Normalizing all columns that have a _prev suffix except any column that has rank in the name
for col in df_wr.columns:
    if col not in ['Player', 'Year', 'PPR', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']:
        df_wr[col] = (df_wr[col] - df_wr[col].mean()) / df_wr[col].std()

# Dropping the smallest 2 years for each player
df_wr = df_wr.sort_values(by=['Player', 'Year'])
df_wr = df_wr.reset_index(drop=True)
df_wr = df_wr.groupby('Player').apply(lambda x: x.iloc[2:]).reset_index(drop=True)

# Testing to see if the model is better with or without certain columns
df_wr = df_wr.drop(columns=['FDPt_prev'])
df_wr = df_wr.drop(columns=['qb_PPR'])
df_wr = df_wr.drop(columns=['Year_count'])
df_wr = df_wr.drop(columns=['FantPt_prev'])
df_wr = df_wr.drop(columns=['DKPt_prev'])
df_wr = df_wr.drop(columns=['Fmb_prev'])
df_wr = df_wr.drop(columns=['Y/R_prev'])
df_wr = df_wr.drop(columns=['OvRank_prev'])
#-----------------
#df_wr = df_wr.drop(columns=['qb_PPR_prev'])
#df_wr = df_wr.drop(columns=['Yds_10_prev'])
#df_wr = df_wr.drop(columns=['Tgt_share_prev'])
#df_wr = df_wr.drop(columns=['Tier_1_Sign'])
df_wr = df_wr.drop(columns=['PPR_prev'])
#df_wr = df_wr.drop(columns=['PPR_2yr_prev'])

# Run model evaluation function 10 times and average the and r2 values and add average PPR score to result dataframe
result, r2, mae, wr_coef, best = ff.model_evaluation(df_wr, 2022, 'PPR', RandomForestRegressor())

print('Average R2: ', r2)
print('Average MSE: ', mae)
print('Best R2: ', best)

# add a column that is the difference between the players PPR and the predicted PPR
result['PPR_diff'] = result['PPR'] - result['Predicted']

X = df_wr.drop('PPR', axis=1)
y = df_wr['PPR']

#q: how to fix error "not in index"
#a: https://stackoverflow.com/questions/41286569/pandas-iloc-error-not-in-index