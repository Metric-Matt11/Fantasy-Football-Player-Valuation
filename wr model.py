# Description: Value fantasy football players based on previous 3 seasons' performance.

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import ff_functions as ff

# Load data
df_wr = pd.read_csv('wr.csv')
df_qb = pd.read_csv('qb.csv')
df_rz_rec = pd.read_csv('rz_rec.csv')
df_signings = pd.read_excel('FA_Signings.xlsx')
df_def = pd.read_csv('team_def.csv')
df_draft = pd.read_csv('draft.csv')
df_adv_rec = pd.read_csv('adv_rec.csv')

# Draft processing and flags
draft_positions = ['QB', 'RB', 'WR', 'TE']
df_draft = (
    df_draft[df_draft['Pick'] <= 10]
    .loc[~df_draft['Position'].isin(['CB', 'DB', 'DE', 'DT', 'G', 'ILB', 'LB', 'OL', 'S', 'T'])]
    .groupby(['Team', 'Year', 'Position'])['Player'].count()
    .unstack(fill_value=0)
    .reset_index()
)
for pos in draft_positions:
    df_draft[f'{pos}_top10'] = (df_draft.get(pos, 0) > 0).astype(int)
df_draft = df_draft[['Team', 'Year'] + [f'{pos}_top10' for pos in draft_positions]]

# Merge dataframes
df_wr = df_wr.merge(df_draft, how='left', left_on=['Tm', 'Year'], right_on=['Team', 'Year'])
df_wr = df_wr.merge(df_rz_rec[['Player', 'Year', 'Yds_20', 'Yds_10']], on=['Player', 'Year'], how='left')
df_wr = df_wr.merge(df_signings[['Tm', 'Year', 'Tier_1_Sign']], on=['Tm', 'Year'], how='left')
df_wr = df_wr.merge(df_def[['Tm', 'Year', 'Pass_QB_def_norm', 'Rush_def_norm']], on=['Tm', 'Year'], how='left')
df_wr['Tgt_share'] = df_wr['Tgt'] / df_wr['Team_Tgt']

# QB data: Only keep QB with highest PPR per team/year
qb_cols = ['Tm', 'Year', 'PPR']
df_qb = df_qb.loc[df_qb.groupby(['Tm', 'Year'])['PPR'].idxmax(), qb_cols]
df_qb = df_qb.rename(columns={'Tm': 'qb_Tm', 'Year': 'qb_Year', 'PPR': 'qb_PPR'})
df_wr = df_wr.merge(df_qb, how='left', left_on=['Tm', 'Year'], right_on=['qb_Tm', 'qb_Year']).drop(['qb_Tm', 'qb_Year'], axis=1)

# Merge advanced receiving stats
df_wr = df_wr.merge(df_adv_rec.drop('Tm', axis=1), on=['Player', 'Year'], how='left')

# Calculate years played (excluding current year)
df_wr['Year_count'] = df_wr.groupby('Player')['Year'].transform('count') - 1

# Drop unnecessary columns
drop_cols = ['FantPos', 'Team', '2PP', 'TD_pass']
df_wr.drop(columns=[col for col in drop_cols if col in df_wr.columns], inplace=True)

# Shifted features for previous and 2 years ago
id_cols = ['Player', 'Year', 'Tm', 'Pos', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']
shift_cols = [col for col in df_wr.columns if col not in id_cols]
for col in shift_cols:
    df_wr[f'{col}_prev'] = df_wr.groupby('Player')[col].shift(1)
    df_wr[f'{col}_2yr_prev'] = df_wr.groupby('Player')[col].shift(2)

df_wr.fillna(0, inplace=True)

# Historical averages (excluding last two years)
hist_cols = [
    'Yds_rec', 'GS', 'PosRank', 'PPR', 'Tgt', 'Tgt_share', 
    'Yds_10', '1D', 'Team_Yds_rush', 'Rec/Br'
]
for col in hist_cols:
    df_wr[f'{col}_historical_avg'] = (
        df_wr.groupby('Player')[col].transform(lambda x: x.expanding().mean().shift(2))
    )

# Remove first two years for each player
df_wr = (df_wr.sort_values(['Player', 'Year'])
         .groupby('Player', group_keys=False)
         .apply(lambda x: x.iloc[2:]).reset_index(drop=True)
)

# Filter current year data for predictions
df_wr_current = df_wr[df_wr['Year'] == 2022].copy()
df_wr_current['PPR'] = 0

# Align columns with training
train_cols = [col for col in df_wr.columns if col != 'Year']
df_wr_current = df_wr_current[train_cols].set_index('Player')

# Fill NA with medians
df_wr_current = df_wr_current.fillna(df_wr_current.median())

# Model training and prediction
s_result, s_r2, s_mae, s_wr_coef, s_best, test = ff.evaluate_model(df_wr, 2022, 'PPR', 'SVR')
df_wr_current['PPR_prediction_2023'] = test.predict(df_wr_current.drop(columns=['PPR'], errors='ignore'))

# Sort and save predictions
df_wr_current = df_wr_current.sort_values('PPR_prediction_2023', ascending=False)
cols = ['PPR_prediction_2023'] + [c for c in df_wr_current.columns if c != 'PPR_prediction_2023']
df_wr_current[cols].to_csv('wr_predictions_2023.csv')

