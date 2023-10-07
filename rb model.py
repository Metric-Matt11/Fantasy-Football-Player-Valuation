# RB Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import ff_functions as ff

# Pulling rb, qb and rec redzone data from csv files
df_rb = pd.read_csv('rb.csv')
df_qb = pd.read_csv('qb.csv')
df_rz_rec = pd.read_csv('rz_rec.csv')
df_rz_rush = pd.read_csv('rz_rush.csv')
df_def = pd.read_csv('team_def.csv')
# Setting df_def Tm to WAS if WSH
df_def['Tm'] = np.where(df_def['Tm'] == 'WSH', 'WAS', df_def['Tm'])

# Setting df_rb Tm to OAK if LV and year less than 2020
df_rb['Tm'] = np.where((df_rb['Tm'] == 'LV') & (df_rb['Year'] < 2020), 'OAK', df_rb['Tm'])

#df_draft = pd.read_csv('draft.csv')
df_adv_rush = pd.read_csv('adv_rush.csv')
df_adv_rec = pd.read_csv('adv_rec.csv')

#Pulling OL_Rank text file and converting to a dataframe
with open('OL_Rank.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]
df_ol = pd.DataFrame(content)
df_ol = df_ol[0].str.split(',', expand=True)
df_ol.columns = df_ol.iloc[0]
df_ol = df_ol[1:]

df_ol['Year'] = df_ol['Year'].astype(int)

#Renaming Rank to OL_Rank and top_10 to OL_top_10
df_ol = df_ol.rename(columns={'Rank': 'OL_Rank', 'top_10': 'OL_top_10'})

# Create a df that has all teams for all years as a columns and has a flag for each position and if the team had a top 10 pick in that postion
#df_draft = df_draft[df_draft['Pick'] <= 10]
#df_draft = df_draft[~df_draft['Position'].isin(['CB', 'DB', 'DE', 'DT', 'G', 'ILB', 'LB', 'OL', 'S', 'T'])]
#df_draft = df_draft.groupby(['Team', 'Year', 'Position']).count().reset_index()
#df_draft = df_draft.pivot(index=['Team', 'Year'], columns='Position', values='Player')
#df_draft = df_draft.reset_index()
#df_draft = df_draft.fillna(0)
#df_draft['QB_top10'] = np.where(df_draft['QB'] > 0, 1, 0)
#df_draft['RB_top10'] = np.where(df_draft['RB'] > 0, 1, 0)
#df_draft['WR_top10'] = np.where(df_draft['WR'] > 0, 1, 0)
#df_draft['TE_top10'] = np.where(df_draft['TE'] > 0, 1, 0)
#df_draft = df_draft.drop(columns=['QB', 'RB', 'WR', 'TE'])

#df_rb = df_rb.merge(df_draft[['Team', 'Year', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']], how='left', left_on=['Tm', 'Year'], right_on=['Team', 'Year'])
df_rb = df_rb.merge(df_rz_rec[['Player', 'Year', 'Yds_20', 'Yds_10']], how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'])
df_rb = df_rb.merge(df_rz_rush[['Player', 'Year', 'Yds_20', 'Yds_10']], how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'], suffixes=('', '_rush'))
df_rb = df_rb.merge(df_def[['Tm', 'Year', 'Pass_QB_def_norm', 'Rush_def_norm']], how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_rb = df_rb.merge(df_ol[['Team', 'Year', 'OL_Rank', 'OL_top_10']], how='left', left_on=['Tm', 'Year'], right_on=['Team', 'Year'])
df_rb['Tgt_share'] = df_rb['Tgt'] / df_rb['Team_Tgt']
df_rb['Rush_share'] = df_rb['Att_rush'] / df_rb['Team_Att_rush']
df_qb = df_qb[df_qb['Year'] >= 2018]

# Join the qb data to the wr data on Team and Year and only pull in the PPR column
# put qb_ as a prefix for all columns in the qb dataframe
# Only join QB with the biggers PPR value
df_qb = df_qb.groupby(['Tm', 'Year']).max().reset_index()
df_qb = df_qb.add_prefix('qb_')
df_rb = df_rb.merge(df_qb[['qb_Tm', 'qb_Year', 'qb_PPR']], how='left', left_on=['Tm', 'Year'], right_on=['qb_Tm', 'qb_Year'])
df_rb = df_rb.drop(columns=['qb_Tm', 'qb_Year'])

df_adv_rush = df_adv_rush.drop(columns=['Tm'])
df_rb = df_rb.merge(df_adv_rush, how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'], suffixes=('', '_rush'))

df_adv_rec = df_adv_rec.drop(columns=['Tm'])
df_rb = df_rb.merge(df_adv_rec, how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'], suffixes=('', '_rec'))

# Getting a count of number of years a player has played in the NFL (excluding 2022)
df_rb['Year_count'] = df_rb.groupby('Player')['Year'].transform('count') - 1

# Dropping all players with less than 3 years of data
df_rb = df_rb[df_rb['Year_count'] >= 3]
#df_rb = df_rb.drop(columns=['FantPos', 'Team', '2PP'])

# Shifting the data for each player by 1 year, changing all columns to have a _prev suffix
df_rb = df_rb.sort_values(by=['Player', 'Year'])
df_rb = df_rb.reset_index(drop=True)
df_rb = df_rb.rename(columns={'Player': 'Player'})

for col in df_rb.columns:
    if col not in ['Player', 'Year', 'Tm', 'Pos', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Pass_QB_def_norm', 'Rush_def_norm', 'OL_Rank', 'OL_top_10']:
        df_rb[col + '_prev'] = df_rb.groupby('Player')[col].shift(1)
        df_rb[col + '_2yr'] = df_rb.groupby('Player')[col].shift(1)
        df_rb[col + '_2yr_prev'] = df_rb.groupby('Player')[col].shift(2)

# Drop Team_x, Team_y, FantPos_prev, FantPos_2yr_prev, Team_x_prev, Team_y_prev, Team_x_2yr_prev, Team_y_2yr_prev
df_rb = df_rb.drop(columns=['FantPos', 'Tm', 'FantPos_prev', 'FantPos_2yr', 'FantPos_2yr_prev', 'Team_prev', 'Team_2yr_prev', 'Team_2yr'])

#Set Null values in the OL_Rank column to 15
df_rb['OL_Rank'] = df_rb['OL_Rank'].fillna(15)
df_rb['OL_top_10'] = df_rb['OL_top_10'].fillna(0)
df_rb = df_rb.fillna(0)

#Setting OL_Rank and OL_top_10 to int
df_rb['OL_Rank'] = df_rb['OL_Rank'].astype(int)
df_rb['OL_top_10'] = df_rb['OL_top_10'].astype(int)

#Creating a new column called OL_top_5 that is a flag for if the player played on a team with a top 5 OL
df_rb['OL_top_5'] = np.where(df_rb['OL_Rank'] <= 5, 1, 0)

# Dropping the smallest 2 years for each player
df_rb = df_rb.sort_values(by=['Player', 'Year'])
df_rb = df_rb.reset_index(drop=True)
df_rb = df_rb.groupby('Player').apply(lambda x: x.iloc[2:]).reset_index(drop=True)

# Recursively looking at each players history in Yds_rec and creating a historical average for each player excluding the current and prior year data
df_rb['PosRank_historical_avg'] = df_rb.groupby('Player')['PosRank'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['PPR_historical_avg'] = df_rb.groupby('Player')['PPR'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Yds_rec_historical_avg'] = df_rb.groupby('Player')['Yds_rec'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['GS_historical_avg'] = df_rb.groupby('Player')['GS'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Tgt_historical_avg'] = df_rb.groupby('Player')['Tgt'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Tgt_share_historical_avg'] = df_rb.groupby('Player')['Tgt_share'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Att_rush_historical_avg'] = df_rb.groupby('Player')['Att_rush'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Yds_rush_historical_avg'] = df_rb.groupby('Player')['Yds_rush'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Yds_rec_historical_avg'] = df_rb.groupby('Player')['Yds_rec'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['PPR_historical_avg'] = df_rb.groupby('Player')['PPR'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Yds_10_historical_avg'] = df_rb.groupby('Player')['Yds_10'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['1D_historical_avg'] = df_rb.groupby('Player')['1D'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Team_Yds_rush_historical_avg'] = df_rb.groupby('Player')['Team_Yds_rush'].transform(lambda x: x.expanding().mean().shift(2))
df_rb['Rec/Br_historical_avg'] = df_rb.groupby('Player')['Rec/Br'].transform(lambda x: x.expanding().mean().shift(2))

df_rb = df_rb[df_rb['Year'] >= 2018]

#Creating a table that has all columns without the prev suffix exluding QB_top10, RB_top10, WR_top10, TE_top10
df_rb_current = df_rb.loc[:, ~df_rb.columns.str.contains('QB_top10') | ~df_rb.columns.str.contains('RB_top10') | ~df_rb.columns.str.contains('WR_top10') | ~df_rb.columns.str.contains('TE_top10')]

#Adding _prev to all columns in df_rb_current except for all columns with a _historical_avg suffix and Player, Team, Year and POS
for col in df_rb_current.columns:
    if '_prev' in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Tier_1_Sign', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg']: #, 'Yds_10_historical_avg']: #, '1D_historical_avg', 'Team_Yds_rush_historical_avg', 'Rec/Br_historical_avg']:
        df_rb_current = df_rb_current.drop(columns=col)

#Adding _prev suffix to all columns in df_rb_current except for all columns with a _historical_avg suffix and Player, Team, Year and POS
for col in df_rb_current.columns:
    if '_prev' not in col and '_historical_average' not in col and col not in ['Player', 'Tm', 'Year', 'qb_PPR', 'Year_count', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg', 'OL_top_5', 'OL_top_10', 'OL_Rank', 'Att_rush_historical_avg', 'Yds_rush_historical_avg']: #, 'Yds_10_historical_avg']: #, '1D_historical_avg', 'Team_Yds_rush_historical_avg', 'Rec/Br_historical_avg']:
        df_rb_current = df_rb_current.rename(columns={col: col + '_prev'})

# Setting all Null values to the median of the column for that year
for col in df_rb.columns:
    if col not in ['Player', 'Year', 'FantPos', 'Tm', 'Team', 'Pos', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Pass_QB_def_norm', 'Rush_def_norm']:
        df_rb[col] = df_rb.groupby('Year')[col].transform(lambda x: x.fillna(x.median()))

# Dropping all columns that dont have a _prev suffix except for Player, Year and PPR
# Use a loop
for col in df_rb.columns:
    if '_prev' not in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg', 'Att_rush_historical_avg', 'Yds_rush_historical_avg', 'OL_Rank', 'OL_top_10', 'OL_top_5']:
        df_rb = df_rb.drop(columns=col)

# Normalizing all columns that have a _prev suffix except any column that has rank in the name
for col in df_rb.columns:
    if col not in ['Player', 'Year', 'PPR', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'OL_Rank', 'OL_top_10', 'OL_top_5']:
        df_rb[col] = (df_rb[col] - df_rb[col].mean()) / df_rb[col].std()

df_rb_current = df_rb_current[df_rb_current['Year'] == 2022]

# Testing to see if the model is better with or without certain columns
df_rb = df_rb.drop(columns=['FDPt_2yr_prev'])
df_rb = df_rb.drop(columns=['FDPt_prev'])
df_rb = df_rb.drop(columns=['DKPt_2yr_prev'])
df_rb = df_rb.drop(columns=['qb_PPR'])
df_rb = df_rb.drop(columns=['Year_count'])
df_rb = df_rb.drop(columns=['FantPt_prev'])
df_rb = df_rb.drop(columns=['DKPt_prev'])
df_rb = df_rb.drop(columns=['VBD_prev'])
df_rb = df_rb.drop(columns=['Y/R_prev'])
df_rb = df_rb.drop(columns=['OvRank_prev'])
df_rb = df_rb.drop(columns=['qb_PPR_prev'])
df_rb = df_rb.drop(columns=['Yds_10_prev'])
df_rb = df_rb.drop(columns=['Tgt_share_prev'])
df_rb = df_rb.drop(columns=['Year_count_prev'])
df_rb = df_rb.drop(columns=['Year_count_2yr_prev'])
df_rb = df_rb.drop(columns=['2PP_prev'])
df_rb = df_rb.drop(columns=['2PP_2yr_prev'])
#df_rb = df_rb.drop(columns=['Cmp_prev'])
#df_rb = df_rb.drop(columns=['Cmp_2yr_prev'])
#df_rb = df_rb.drop(columns=['Att_pass_prev'])
#df_rb = df_rb.drop(columns=['Att_pass_2yr_prev'])

#df_rb = df_rb.drop(columns=['TD_pass_prev'])
#df_rb = df_rb.drop(columns=['TD_pass_2yr_prev'])
#df_rb = df_rb.drop(columns=['Int_prev'])
#df_rb = df_rb.drop(columns=['Int_2yr_prev'])
#--------------------
#df_rb = df_rb.drop(columns=['Tgt_share_prev'])
#df_rb = df_rb.drop(columns=['Tier_1_Sign'])
#df_rb = df_rb.drop(columns=['PPR_prev'])
#df_rb = df_rb.drop(columns=['PPR_2yr_prev'])
#df_rb = df_rb.drop(columns=['PosRank_prev'])

# Run model evaluation function 10 times and average the and r2 values and add average PPR score to result dataframe
result_all, r2_all, mae, rb_coef, best, test = ff.evaluate_model(df_rb, 2022, 'PPR', 'RandomForestRegressor')
#x_result, x_r2, x_mae, x_rb_coef, x_best, x_model = ff.evaluate_model(df_rb, 2022, 'PPR', 'XGBRegressor')
#s_result, s_r2, s_mae, s_rb_coef, s_best, s_model = ff.evaluate_model(df_rb, 2022, 'PPR', 'SVR')

#Creating a PPR column and setting them all to 0
df_rb_current['PPR'] = 0

#Putting columns in current dataframe in same order as training dataframe
df_rb_current = df_rb_current[df_rb.columns]

#Drop the following columns from the wr_df_current dataframe: 'Tm', 'Year', 'PPR'
df_rb_current = df_rb_current.drop(columns=['Year', 'PPR'])
df_rb_current = df_rb_current.set_index('Player')

#Filling NA values with the median for each column
#column_medians = df_rb_current.median()

#rb_coef = rb_coef.sort_values(by='Coefficient', ascending=False)
#rb_coef = rb_coef[rb_coef['Coefficient'] >= 0.002]
#rb_coef = rb_coef.reset_index(drop=True)

#adding the player column to the list
#rb_coef = rb_coef.append({'Feature': 'Player'}, ignore_index=True)
#rb_coef = rb_coef.append({'Feature': 'Year'}, ignore_index=True)
#rb_coef = rb_coef.append({'Feature': 'PPR'}, ignore_index=True)

#Reruning the model with only the top 20 columns
#df_rb = df_rb[rb_coef['Feature']]

#result_20, r2_20, mae_20, rb_coef_20, best, model_20 = ff.evaluate_model(df_rb, 2022, 'PPR', 'RandomForestRegressor')
#x_result_20, x_r2_20, x_mae, x_rb_coef, x_best, x_model_20 = ff.evaluate_model(df_rb, 2022, 'PPR', 'XGBRegressor')

#df_wr_current = df_rb_current[df_rb.columns]

# Iterate over each column in the DataFrame
for column in df_rb_current.columns:
    # Check if any NA values exist in the column
    if df_rb_current[column].isna().any():
        # Fill NA values with the corresponding column median
        df_rb_current[column].fillna(column_medians[column], inplace=True)

#Making CURRENT PREDICTION
ppr_pred_2023 = test.predict(df_rb_current)
df_rb_current['PPR_prediction_2023'] = ppr_pred_2023

#Ordering the dataframe by the PPR_prediction_2023 column descending
df_rb_current = df_rb_current.sort_values(by='PPR_prediction_2023', ascending=False)

#Putting PPR_prediction_2023 column in front of the dataframe
cols = df_rb_current.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_rb_current = df_rb_current[cols]

#Sending the dataframe to a csv file
df_rb_current.to_csv('rb_predictions_2023.csv')

d = 0