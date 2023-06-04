#TE Model
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import ff_functions as ff

# Pulling te, qb and rec redzone data from csv files
df_te = pd.read_csv('te.csv')
df_qb = pd.read_csv('qb.csv')
df_rz_rec = pd.read_csv('rz_rec.csv')
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

df_te = df_te.merge(df_draft[['Team', 'Year', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']], how='left', left_on=['Tm', 'Year'], right_on=['Team', 'Year'])
df_te = df_te.merge(df_rz_rec[['Player', 'Year', 'Yds_20', 'Yds_10']], how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'])
df_te = df_te.merge(df_def[['Tm', 'Year', 'Pass_QB_def_norm', 'Rush_def_norm']], how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_te['Tgt_share'] = df_te['Tgt'] / df_te['Team_Tgt']

# Join the qb data to the wr data on Team and Year and only pull in the PPR column
# put qb_ as a prefix for all columns in the qb dataframe
# Only join QB with the biggers PPR value
df_qb = df_qb.groupby(['Tm', 'Year']).max().reset_index()
df_qb = df_qb.add_prefix('qb_')
df_te = df_te.merge(df_qb[['qb_Tm', 'qb_Year', 'qb_PPR']], how='left', left_on=['Tm', 'Year'], right_on=['qb_Tm', 'qb_Year'])
df_te = df_te.drop(columns=['qb_Tm', 'qb_Year'])

# Getting a count of number of years a player has played in the NFL (excluding 2022)
df_te['Year_count'] = df_te.groupby('Player')['Year'].transform('count') - 1

# Dropping all players with less than 3 years of data
df_te = df_te[df_te['Year_count'] >= 3]

# Shifting the data for each player by 1 year, changing all columns to have a _prev suffix
df_te = df_te.sort_values(by=['Player', 'Year'])
df_te = df_te.reset_index(drop=True)
df_te = df_te.rename(columns={'Player': 'Player'})
df_te['Tgt_prev'] = df_te.groupby('Player')['Tgt'].shift(1)
df_te['Tgt_share_prev'] = df_te.groupby('Player')['Tgt_share'].shift(1)
#df_te['Tgt_share_2yr_prev'] = df_te.groupby('Player')['Tgt_share'].shift(2)
#df_te['Yds_rec/Att_prev'] = df_te.groupby('Player')['Yds_rec/Att'].shift(1)
df_te['TD_rec_prev'] = df_te.groupby('Player')['TD_rec'].shift(1)
df_te['Yds_20_prev'] = df_te.groupby('Player')['Yds_20'].shift(1)
df_te['Yds_10_prev'] = df_te.groupby('Player')['Yds_10'].shift(1)
df_te['Yds_rec_prev'] = df_te.groupby('Player')['Yds_rec'].shift(1)
#df_te['Yds_rec_2yr_prev'] = df_te.groupby('Player')['Yds_rec'].shift(2)
df_te['Y/R_prev'] = df_te.groupby('Player')['Y/R'].shift(1)
df_te['Fmb_prev'] = df_te.groupby('Player')['Fmb'].shift(1)
df_te['TD_rec_prev'] = df_te.groupby('Player')['TD_rec'].shift(1)
df_te['G_prev'] = df_te.groupby('Player')['G'].shift(1)
df_te['GS_prev'] = df_te.groupby('Player')['GS'].shift(1)
df_te['PPR_prev'] = df_te.groupby('Player')['PPR'].shift(1)
#df_te['PPR_2yr_prev'] = df_te.groupby('Player')['PPR'].shift(2)
df_te['PosRank_prev'] = df_te.groupby('Player')['PosRank'].shift(1)
#df_te['PosRank_2yr_prev'] = df_te.groupby('Player')['PosRank'].shift(2)
df_te['OvRank_prev'] = df_te.groupby('Player')['OvRank'].shift(1)
df_te['VBD_prev'] = df_te.groupby('Player')['VBD'].shift(1)
df_te['FantPt_prev'] = df_te.groupby('Player')['FantPt'].shift(1)
df_te['DKPt_prev'] = df_te.groupby('Player')['DKPt'].shift(1)
df_te['FDPt_prev'] = df_te.groupby('Player')['FDPt'].shift(1)
df_te['qb_PPR_prev'] = df_te.groupby('Player')['qb_PPR'].shift(1)
df_te['Team_Off_prev'] = df_te.groupby('Player')['Team_Yds'].shift(1)
df_te = df_te.fillna(0)

# Recursively looking at each players history in Yds_rec and creating a historical average for each player excluding the current and prior year data
df_te['PosRank_historical_avg'] = df_te.groupby('Player')['PosRank'].transform(lambda x: x.expanding().mean().shift(2))
df_te['PPR_historical_avg'] = df_te.groupby('Player')['PPR'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Yds_rec_historical_avg'] = df_te.groupby('Player')['Yds_rec'].transform(lambda x: x.expanding().mean().shift(2))

df_te['GS_historical_avg'] = df_te.groupby('Player')['GS'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Tgt_historical_avg'] = df_te.groupby('Player')['Tgt'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Tgt_share_historical_avg'] = df_te.groupby('Player')['Tgt_share'].transform(lambda x: x.expanding().mean().shift(2))

# Dropping all columns that dont have a _prev suffix except for Player, Year and PPR
# Use a loop
for col in df_te.columns:
    if '_prev' not in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg']:
        df_te = df_te.drop(columns=col)

# Normalizing all columns that have a _prev suffix except any column that has rank in the name
for col in df_te.columns:
    if col not in ['Player', 'Year', 'PPR', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']:
        df_te[col] = (df_te[col] - df_te[col].mean()) / df_te[col].std()

# Dropping the smallest 2 years for each player
df_te = df_te.sort_values(by=['Player', 'Year'])
df_te = df_te.reset_index(drop=True)
df_te = df_te.groupby('Player').apply(lambda x: x.iloc[2:]).reset_index(drop=True)

# Testing to see if the model is better with or without certain columns
df_te = df_te.drop(columns=['FDPt_prev'])
df_te = df_te.drop(columns=['qb_PPR'])
df_te = df_te.drop(columns=['Year_count'])
df_te = df_te.drop(columns=['FantPt_prev'])
df_te = df_te.drop(columns=['DKPt_prev'])
df_te = df_te.drop(columns=['Fmb_prev'])
df_te = df_te.drop(columns=['Y/R_prev'])
df_te = df_te.drop(columns=['OvRank_prev'])
#-----------------
#df_te = df_te.drop(columns=['qb_PPR_prev'])
#df_te = df_te.drop(columns=['Yds_10_prev'])
#df_te = df_te.drop(columns=['Tgt_share_prev'])
#df_te = df_te.drop(columns=['Tier_1_Sign'])
#df_te = df_te.drop(columns=['PPR_prev'])
#df_te = df_te.drop(columns=['PPR_2yr_prev'])

# Run model evaluation function 10 times and average the and r2 values and add average PPR score to result dataframe
result, r2, mae, te_coef, best = ff.model_evaluation(df_te, 2022, 'PPR', RandomForestRegressor())

print('Average R2: ', r2)
print('Average MSE: ', mae)
print('Best R2: ', best)

# add a column that is the difference between the players PPR and the predicted PPR
result['PPR_diff'] = result['PPR'] - result['Predicted']

X = df_te.drop('PPR', axis=1)
y = df_te['PPR']