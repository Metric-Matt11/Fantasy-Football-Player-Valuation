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
df_adv_rec = pd.read_csv('adv_rec.csv')

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

df_adv_rec = df_adv_rec.drop(columns=['Tm'])
df_te = df_te.merge(df_adv_rec, how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'], suffixes=('', '_rec_adv'))

# Getting a count of number of years a player has played in the NFL (excluding 2022)
df_te['Year_count'] = df_te.groupby('Player')['Year'].transform('count') - 1

# Dropping all players with less than 3 years of data
#df_te = df_te[df_te['Year_count'] >= 3]
df_te = df_te.drop(columns=['FantPos', 'Team', '2PP', 'TD_pass'])

# Shifting the data for each player by 1 year, changing all columns to have a _prev suffix
df_te = df_te.sort_values(by=['Player', 'Year'])
df_te = df_te.reset_index(drop=True)
df_te = df_te.rename(columns={'Player': 'Player'})

for col in df_te.columns:
    if col not in ['Player', 'Year', 'Tm', 'Pos', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Pass_QB_def_norm', 'Rush_def_norm']:
        df_te[col + '_prev'] = df_te.groupby('Player')[col].shift(1)
        df_te[col + '_2yr'] = df_te.groupby('Player')[col].shift(1)
        df_te[col + '_2yr_prev'] = df_te.groupby('Player')[col].shift(2)

df_te = df_te.fillna(0)

# Recursively looking at each players history in Yds_rec and creating a historical average for each player excluding the current and prior year data
df_te['PosRank_historical_avg'] = df_te.groupby('Player')['PosRank'].transform(lambda x: x.expanding().mean().shift(2))
df_te['PPR_historical_avg'] = df_te.groupby('Player')['PPR'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Yds_rec_historical_avg'] = df_te.groupby('Player')['Yds_rec'].transform(lambda x: x.expanding().mean().shift(2))
df_te['GS_historical_avg'] = df_te.groupby('Player')['GS'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Tgt_historical_avg'] = df_te.groupby('Player')['Tgt'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Tgt_share_historical_avg'] = df_te.groupby('Player')['Tgt_share'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Yds_10_historical_avg'] = df_te.groupby('Player')['Yds_10'].transform(lambda x: x.expanding().mean().shift(2))
df_te['1D_historical_avg'] = df_te.groupby('Player')['1D'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Team_Yds_rush_historical_avg'] = df_te.groupby('Player')['Team_Yds_rush'].transform(lambda x: x.expanding().mean().shift(2))
df_te['Rec/Br_historical_avg'] = df_te.groupby('Player')['Rec/Br'].transform(lambda x: x.expanding().mean().shift(2))

#Adding a column that is the max PPR for each player
df_te['PPR_max'] = df_te.groupby('Player')['PPR'].transform('max')

#Creating a table that has all columns without the prev suffix exluding QB_top10, RB_top10, WR_top10, TE_top10
df_te_current = df_te.loc[:, ~df_te.columns.str.contains('QB_top10') | ~df_te.columns.str.contains('RB_top10') | ~df_te.columns.str.contains('WR_top10') | ~df_te.columns.str.contains('TE_top10')]

#Adding _prev to all columns in df_te_current except for all columns with a _historical_avg suffix and Player, Team, Year and POS
for col in df_te_current.columns:
    if '_prev' in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Tier_1_Sign', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg']: #, 'Yds_10_historical_avg']: #, '1D_historical_avg', 'Team_Yds_rush_historical_avg', 'Rec/Br_historical_avg']:
        df_te_current = df_te_current.drop(columns=col)

#Adding _prev suffix to all columns in df_te_current except for all columns with a _historical_avg suffix and Player, Team, Year and POS
for col in df_te_current.columns:
    if '_prev' not in col and '_historical_average' not in col and col not in ['Player', 'Tm', 'Year', 'qb_PPR', 'Year_count', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg']: #, 'Yds_10_historical_avg']: #, '1D_historical_avg', 'Team_Yds_rush_historical_avg', 'Rec/Br_historical_avg']:
        df_te_current = df_te_current.rename(columns={col: col + '_prev'})

# Dropping all columns that dont have a _prev suffix except for Player, Year and PPR
# Use a loop
for col in df_te.columns:
    if '_prev' not in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg']:
        df_te = df_te.drop(columns=col)

# Normalizing all columns that have a _prev suffix except any column that has rank in the name
#FIND A WAY TO FIX NORMALIZATION
for col in df_te.columns:
    if col not in ['Player', 'Year', 'PPR', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Year_count']:
        df_te[col] = (df_te[col] - df_te[col].mean()) / df_te[col].std()

for col in df_te_current.columns:
    if col not in ['Tm' ,'Player', 'Year', 'PPR', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']:
        df_te_current[col] = (df_te_current[col] - df_te_current[col].mean()) / df_te_current[col].std()        

# Dropping the smallest 2 years for each player
df_te = df_te.sort_values(by=['Player', 'Year'])
df_te = df_te.reset_index(drop=True)
df_te = df_te.groupby('Player').apply(lambda x: x.iloc[2:]).reset_index(drop=True)

df_te_current = df_te_current[df_te_current['Year'] == 2022]

df_te = df_te[df_te['Year_count'] >= 3]

# Testing to see if the model is better with or without certain columns
df_te = df_te.drop(columns=['FDPt_2yr_prev'])
df_te = df_te.drop(columns=['FDPt_prev'])
df_te = df_te.drop(columns=['DKPt_2yr_prev'])
df_te = df_te.drop(columns=['qb_PPR'])
df_te = df_te.drop(columns=['Year_count'])
df_te = df_te.drop(columns=['FantPt_prev'])
df_te = df_te.drop(columns=['DKPt_prev'])
df_te = df_te.drop(columns=['VBD_prev'])
df_te = df_te.drop(columns=['Y/R_prev'])
df_te = df_te.drop(columns=['OvRank_prev'])
df_te = df_te.drop(columns=['qb_PPR_prev'])
df_te = df_te.drop(columns=['Yds_10_prev'])
df_te = df_te.drop(columns=['Tgt_share_prev'])
df_te = df_te.drop(columns=['Year_count_prev'])
df_te = df_te.drop(columns=['Year_count_2yr_prev'])
df_te = df_te.drop(columns=['QB_top10'])
df_te = df_te.drop(columns=['RB_top10'])
df_te = df_te.drop(columns=['WR_top10'])
df_te = df_te.drop(columns=['TE_top10'])

df_te = df_te.drop(columns=['Att_pass_prev'])
df_te = df_te.drop(columns=['Att_pass_2yr_prev'])
#df_te = df_te.drop(columns=['Att_rush_prev'])
#df_te = df_te.drop(columns=['Att_rush_2yr_prev'])
#df_te = df_te.drop(columns=['Yds_pass_prev'])
#df_te = df_te.drop(columns=['Yds_pass_2yr_prev'])
#df_te = df_te.drop(columns=['Yds_rush_prev'])
#df_te = df_te.drop(columns=['Yds_rush_2yr_prev'])
#df_te = df_te.drop(columns=['TD_rush_prev'])
#df_te = df_te.drop(columns=['TD_rush_2yr_prev'])
#df_te = df_te.drop(columns=['Cmp_prev'])
#df_te = df_te.drop(columns=['Cmp_2yr_prev'])
#df_te = df_te.drop(columns=['Int_prev'])
#df_te = df_te.drop(columns=['Int_2yr_prev'])
#df_te = df_te.drop(columns=['Int_pass_prev'])
#df_te = df_te.drop(columns=['Int_pass_2yr_prev'])
#df_te = df_te.drop(columns=['Age_prev'])
#df_te = df_te.drop(columns=['Age_2yr_prev'])
#df_te = df_te.drop(columns=['Cmp_prev'])
#df_te = df_te.drop(columns=['Cmp_2yr_prev'])
#df_te = df_te.drop(columns=['GS_prev'])
#df_te = df_te.drop(columns=['GS_2yr_prev'])
#df_te = df_te.drop(columns=['G_prev'])
#df_te = df_te.drop(columns=['G_2yr_prev'])
#df_te = df_te.drop(columns=['Fmb_prev'])
#df_te = df_te.drop(columns=['Fmb_2yr_prev'])
#df_te = df_te.drop(columns=['2PM_prev'])
#df_te = df_te.drop(columns=['2PM_2yr_prev'])
#df_te = df_te.drop(columns=['Pass_QB_def_norm_prev'])
#df_te = df_te.drop(columns=['Rush_def_norm_prev'])
#df_te = df_te.drop(columns=['Pass_QB_def_norm_2yr_prev'])
#df_te = df_te.drop(columns=['Rush_def_norm_2yr_prev'])

# Run model evaluation function 10 times and average the and r2 values and add average PPR score to result dataframe
result, r2, mae, te_coef, best, model = ff.evaluate_model(df_te, 2022, 'PPR', 'RandomForestRegressor')
#x_result, x_r2, x_mae, x_te_coef, x_best, x_model = ff.evaluate_model(df_te, 2022, 'PPR', 'XGBRegressor')
#s_result, s_r2, s_mae, s_te_coef, s_best, s_model = ff.evaluate_model(df_te, 2022, 'PPR', 'SVR')

print('Average R2: ', r2)
print('Average MSE: ', mae)
print('Best R2: ', best)

te_coef = te_coef.sort_values(by='Coefficient', ascending=False)
#te_coef = te_coef[te_coef['Coefficient'] >= 0.006]
te_coef = te_coef[te_coef['Coefficient'] >= 0.005]
#te_coef = te_coef[te_coef['Coefficient'] >= 0.004]
te_coef = te_coef.reset_index(drop=True)

#adding the player column to the list
te_coef = te_coef.append({'Feature': 'Player'}, ignore_index=True)
te_coef = te_coef.append({'Feature': 'Year'}, ignore_index=True)
te_coef = te_coef.append({'Feature': 'PPR'}, ignore_index=True)

#Reruning the model with only the top 20 columns
df_te = df_te[te_coef['Feature']]

result_15, r2_15, mae, te_coef_15, best, model_15 = ff.evaluate_model(df_te, 2022, 'PPR', 'RandomForestRegressor')
#x_result_15, x_r2_15, x_mae, x_te_coef_15, x_best, model_x = ff.evaluate_model(df_te, 2022, 'PPR', 'XGBRegressor')
s_result_15, s_r2_15, s_mae, s_te_coef_15, s_best, model_s = ff.evaluate_model(df_te, 2022, 'PPR', 'SVR')

#Creating a PPR column and setting them all to 0
df_te_current['PPR'] = 0

#Taking _prev off of the column names for QB_def_norm and Rush_def_norm
df_te_current = df_te_current.rename(columns={'Pass_QB_def_norm_prev': 'Pass_QB_def_norm', 'Rush_def_norm_prev': 'Rush_def_norm'})

#Putting columns in current dataframe in same order as training dataframe
df_te_current = df_te_current[df_te.columns]

#Drop the following columns from the wr_df_current dataframe: 'Tm', 'Year', 'PPR'
df_te_current = df_te_current.drop(columns=['Year', 'PPR'])
df_te_current = df_te_current.set_index('Player')

#Filling NA values with the median for each column
column_medians = df_te_current.median()

# Iterate over each column in the DataFrame
for column in df_te_current.columns:
    # Check if any NA values exist in the column
    if df_te_current[column].isna().any():
        # Fill NA values with the corresponding column median
        df_te_current[column].fillna(column_medians[column], inplace=True)

#Making CURRENT PREDICTION
ppr_pred_2023 = model_s.predict(df_te_current)
df_te_current['PPR_prediction_2023'] = ppr_pred_2023

#Ordering the dataframe by the PPR_prediction_2023 column descending
df_te_current = df_te_current.sort_values(by='PPR_prediction_2023', ascending=False)

#Putting PPR_prediction_2023 column in front of the dataframe
cols = df_te_current.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_te_current = df_te_current[cols]

#Sending the dataframe to a csv file
df_te_current.to_csv('te_predictions_2023.csv')        

m = 1