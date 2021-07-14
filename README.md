- 👋 Hi, I’m @Metric-Matt11
- 👀 I’m interested in Sport Analysis and Data Science
- 🌱 I’m currently learning Python
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me mj133@evansville.edu

<!---
Metric-Matt11/Metric-Matt11 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

mvp = pd.read_csv('datasets/Historical MVPs.csv')
new_header = mvp.iloc[0]
mvp = mvp[1:]
mvp.columns = new_header
mvp_cleaned = mvp.dropna(subset= ['Year'])
mvp_cleaned['Year'] = mvp_cleaned['Year'].astype(int)
float_cols = ['WAR', 'BA', 'OBP', 'SLG', 'HR', 'RBI', 'SB', 'W', 'L', 'SV', 'ERA', 'IP', 'SO']

for cols in float_cols:
    mvp_cleaned[cols] = mvp_cleaned[cols].astype(float)

hitters = mvp_cleaned[pd.isnull(mvp_cleaned['L'])]
print(hitters.columns)
hitters_recent = hitters[(hitters['Year'] >= 2000)]
plt.scatter('Year', 'BA', data=hitters_recent )
plt.colorbar()
plt.xticks(rotation=90)
#plt.show()

y = hitters.WAR
x = hitters[['BA', 'OBP', 'SLG', 'HR', 'RBI']]

train_X, val_X, train_y, val_y = train_test_split(x,y, random_state= 1)
WAR_model = DecisionTreeRegressor(random_state= 1)
WAR_model.fit(train_X, train_y)
val_prediction = WAR_model.predict(val_X)
print(mean_absolute_error(val_y, val_prediction))
