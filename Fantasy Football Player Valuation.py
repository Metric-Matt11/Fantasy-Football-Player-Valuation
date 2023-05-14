# Description: This program will set a value for each player 
#              based on their performance in the previous 3 season.
#              The idea for this model is to create a ranking for a fantasy football draft.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import ff_functions as ff

# Creating dataframes for each position, defense, and schedule
df_qb, df_rb, df_wr, df_te = ff.player_scrape(2019, 2022)
df_def = ff.team_def_scrape(2019, 2022)
df_schedule = ff.nfl_schedule(2019, 2022)
df_qb_adv, df_qb_adv_team = ff.qb_adv_stats(2019, 2022)

#Dropping players whose team is not in the nfl_schedule dataframe
#This should drop players with no team or multiple teams
df_qb = df_qb[df_qb['Tm'].isin(df_schedule['TEAM'])]
df_rb = df_rb[df_rb['Tm'].isin(df_schedule['TEAM'])]
df_wr = df_wr[df_wr['Tm'].isin(df_schedule['TEAM'])]
df_te = df_te[df_te['Tm'].isin(df_schedule['TEAM'])]

# Creating WR features
df_wr['Tgt_share'] = df_wr['Tgt'] / df_wr['Team_Tgt']
df_wr['TD_share'] = df_wr['TD_rec'] / df_wr['Team_TD']
df_wr['Team_TD_norm'] = df_wr.groupby('Year')['Team_TD'].transform(lambda x: (x - x.mean()) / x.std())
df_wr = df_wr.reset_index(drop=True)

# Joining df_def_sched by the opponent that week and year to df_schedule
# Using a loop to go through each week and year
# Curretnly only using 17 weeks since 2020 does not have a week 18
for x in range(1, 18):
    df_schedule = df_schedule.merge(df_def, how='left', left_on=['week_' + str(x), 'Year'], right_on=['Tm', 'Year'])
    df_schedule = df_schedule.rename(columns={'Pass_QB_def_norm': 'Pass_QB_def_norm_' + str(x), 'Rush_def_norm': 'Rush_def_norm_' + str(x)})
    df_schedule = df_schedule.drop(columns=['Tm'])
df_schedule = df_schedule.fillna(0)

# Creating a column that is the sum of the normalized defensive stats using a loop
# This will be used to determine the strength of the defense for each team
df_schedule['Pass_QB_def_norm_sum'] = 0
df_schedule['Rush_def_norm_sum'] = 0
for x in range(1, 18):
    df_schedule['Pass_QB_def_norm_sum'] = df_schedule['Pass_QB_def_norm_sum'] + df_schedule['Pass_QB_def_norm_' + str(x)]
    df_schedule['Rush_def_norm_sum'] = df_schedule['Rush_def_norm_sum'] + df_schedule['Rush_def_norm_' + str(x)]
df_schedule = df_schedule[['TEAM', 'Year', 'Pass_QB_def_norm_sum', 'Rush_def_norm_sum']]

# Joining df_schedule to each position dataframe
df_qb = df_qb.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])
df_rb = df_rb.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])
df_wr = df_wr.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])
df_te = df_te.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])

# Joining df_qb_adv_team to df_wr
df_wr = df_wr.merge(df_qb_adv_team, how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_wr['Yds_rec/Att'] = df_wr['Yds_rec'] / df_wr['Team_Att']

# Creating a column with the count of how many years of data we have for each player
df_wr['Years'] = df_wr.groupby('Player')['Year'].transform('count')
df_wr = df_wr[df_wr['Years'] >= 4]

# Creating a column that has the average of the previous 2 years of data exluding the current year
df_wr['TD_rec_2yr_avg'] = df_wr.groupby('Player')['TD_rec'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['TD_share_2yr_avg'] = df_wr.groupby('Player')['TD_share'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['Tgt_share_2yr_avg'] = df_wr.groupby('Player')['Tgt_share'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['Team_TD_norm_2yr_avg'] = df_wr.groupby('Player')['Team_TD_norm'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['Pass_QB_def_norm_sum_2yr_avg'] = df_wr.groupby('Player')['Pass_QB_def_norm_sum'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['Team_OnTgt_norm_2yr_avg'] = df_wr.groupby('Player')['Team_OnTgt_norm'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['Yds_rec/Att_2yr_avg'] = df_wr.groupby('Player')['Yds_rec/Att'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
df_wr['PPR_2yr_avg'] = df_wr.groupby('Player')['PPR'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)

# Figure out how to use 2yr ppr avg
wr_model = df_wr[['Player', 'Year', 'Age', 'Pass_QB_def_norm_sum', 'TD_rec_2yr_avg', 'Tgt_share_2yr_avg', 'Team_TD_norm_2yr_avg', 'Team_OnTgt_norm_2yr_avg', 'Yds_rec/Att_2yr_avg', 'PPR_2yr_avg', 'PPR']]

result, r2, mse, wr_coef = ff.evaluate_model(wr_model, 2022, 2021, 'PPR', RandomForestRegressor())

print(r2, mse)
print(result)
