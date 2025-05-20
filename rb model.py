# RB Model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import ff_functions as ff

# Load data
df_rb = pd.read_csv('rb.csv')
df_qb = pd.read_csv('qb.csv')
df_rz_rec = pd.read_csv('rz_rec.csv')
df_rz_rush = pd.read_csv('rz_rush.csv')
df_def = pd.read_csv('team_def.csv')
df_adv_rush = pd.read_csv('adv_rush.csv')
df_adv_rec = pd.read_csv('adv_rec.csv')

# Standardize team names
df_def['Tm'] = np.where(df_def['Tm'] == 'WSH', 'WAS', df_def['Tm'])
df_rb['Tm'] = np.where((df_rb['Tm'] == 'LV') & (df_rb['Year'] < 2020), 'OAK', df_rb['Tm'])

# Read and process OL_Rank.txt
with open('OL_Rank.txt') as f:
    content = [x.strip() for x in f.readlines()]
df_ol = pd.DataFrame(content)
df_ol = df_ol[0].str.split(',', expand=True)
df_ol.columns = df_ol.iloc[0]
df_ol = df_ol[1:]
df_ol['Year'] = df_ol['Year'].astype(int)
df_ol = df_ol.rename(columns={'Rank': 'OL_Rank', 'top_10': 'OL_top_10'})

# Merge additional data
df_rb = df_rb.merge(df_rz_rec[['Player', 'Year', 'Yds_20', 'Yds_10']], on=['Player', 'Year'], how='left')
df_rb = df_rb.merge(df_rz_rush[['Player', 'Year', 'Yds_20', 'Yds_10']], on=['Player', 'Year'], how='left', suffixes=('', '_rush'))
df_rb = df_rb.merge(df_def[['Tm', 'Year', 'Pass_QB_def_norm', 'Rush_def_norm']], on=['Tm', 'Year'], how='left')
df_rb = df_rb.merge(df_ol[['Team', 'Year', 'OL_Rank', 'OL_top_10']], left_on=['Tm', 'Year'], right_on=['Team', 'Year'], how='left')

# Calculate share columns
df_rb['Tgt_share'] = df_rb['Tgt'] / df_rb['Team_Tgt']
df_rb['Rush_share'] = df_rb['Att_rush'] / df_rb['Team_Att_rush']

# Use only recent QB data and join max PPR to RBs
df_qb = df_qb[df_qb['Year'] >= 2018]
df_qb = df_qb.groupby(['Tm', 'Year']).max().reset_index().add_prefix('qb_')
df_rb = df_rb.merge(df_qb[['qb_Tm', 'qb_Year', 'qb_PPR']], left_on=['Tm', 'Year'], right_on=['qb_Tm', 'qb_Year'], how='left')
df_rb = df_rb.drop(columns=['qb_Tm', 'qb_Year'])

# Merge advanced stats
df_rb = df_rb.merge(df_adv_rush.drop(columns=['Tm']), on=['Player', 'Year'], how='left', suffixes=('', '_rush'))
df_rb = df_rb.merge(df_adv_rec.drop(columns=['Tm']), on=['Player', 'Year'], how='left', suffixes=('', '_rec'))

# Count years in NFL (excluding 2022) and filter for 3+ years
df_rb['Year_count'] = df_rb.groupby('Player')['Year'].transform('count') - 1
df_rb = df_rb[df_rb['Year_count'] >= 3]

# Sort and reset
df_rb = df_rb.sort_values(['Player', 'Year']).reset_index(drop=True)

# Shift columns for previous/2yr data
base_columns = set([
    'Player', 'Year', 'Tm', 'Pos', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10',
    'Pass_QB_def_norm', 'Rush_def_norm', 'OL_Rank', 'OL_top_10'
])
for col in df_rb.columns:
    if col not in base_columns:
        df_rb[col + '_prev'] = df_rb.groupby('Player')[col].shift(1)
        df_rb[col + '_2yr_prev'] = df_rb.groupby('Player')[col].shift(2)

# Drop unnecessary columns
drop_base = [
    'FantPos', 'Tm', 'FantPos_prev', 'FantPos_2yr', 'FantPos_2yr_prev',
    'Team_prev', 'Team_2yr_prev', 'Team_2yr'
]
df_rb = df_rb.drop(columns=[col for col in drop_base if col in df_rb.columns])

# Fill OL columns
df_rb['OL_Rank'] = df_rb['OL_Rank'].fillna(15).astype(int)
df_rb['OL_top_10'] = df_rb['OL_top_10'].fillna(0).astype(int)
df_rb = df_rb.fillna(0)
df_rb['OL_top_5'] = np.where(df_rb['OL_Rank'] <= 5, 1, 0)

# Drop first 2 years for each player
df_rb = df_rb.groupby('Player').apply(lambda x: x.iloc[2:]).reset_index(drop=True)

# Historical averages
historical_cols = [
    'PosRank', 'PPR', 'Yds_rec', 'GS', 'Tgt', 'Tgt_share', 'Att_rush',
    'Yds_rush', 'Yds_10', '1D', 'Team_Yds_rush', 'Rec/Br'
]
for col in historical_cols:
    df_rb[f'{col}_historical_avg'] = df_rb.groupby('Player')[col].transform(lambda x: x.expanding().mean().shift(2))

df_rb = df_rb[df_rb['Year'] >= 2018]

# Prepare current year dataframe (2022)
df_rb_current = df_rb[df_rb['Year'] == 2022].copy()

# Drop or rename unnecessary columns for modeling
exclude_cols = ['Year', 'PPR']
df_rb_current = df_rb_current.drop(columns=[col for col in exclude_cols if col in df_rb_current.columns])
df_rb_current = df_rb_current.set_index('Player')

# Fill NA with column medians
column_medians = df_rb_current.median()
df_rb_current = df_rb_current.fillna(column_medians)

# Run model evaluation
result_all, r2_all, mae, rb_coef, best, test = ff.evaluate_model(df_rb, 2022, 'PPR', 'RandomForestRegressor')

# Predict for 2023
df_rb_current['PPR_prediction_2023'] = test.predict(df_rb_current)
df_rb_current = df_rb_current.sort_values(by='PPR_prediction_2023', ascending=False)

# Move prediction column to front
cols = df_rb_current.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df_rb_current = df_rb_current[cols]

# Output
df_rb_current.to_csv('rb_predictions_2023.csv')
