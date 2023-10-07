from oauth2client.service_account import ServiceAccountCredentials
import scraper as sc
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import gspread
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

def my_task():
    # Get the current date
    current_date = datetime.date.today()

    # Calculate the difference in days between the current day of the week (0 = Monday, 6 = Sunday)
    days_until_saturday = (5 - current_date.weekday() + 7) % 7
    upcoming_saturday_1 = current_date + datetime.timedelta(days=days_until_saturday)
    upcoming_saturday = upcoming_saturday_1.strftime('%m/%d')

    # Define the start date of the college football season (adjust as needed)
    college_football_start_date = datetime.date(2023, 8, 26)  # Example start date for 2023 season
    current_date = datetime.date.today()
    if current_date == upcoming_saturday_1:
        current_week = (current_date - college_football_start_date).days // 7
    else:
        current_week = (current_date - college_football_start_date).days // 7 + 1

    #Importing scores
    scores_og = sc.get_past_scores([current_week], [2023])
    #scores_og = sc.get_past_scores([5], [2023])

    #Importing picks
    picks, current_scores_status = sc.get_current_picks(str(current_week), upcoming_saturday)
    #picks, current_scores_status = sc.get_current_picks('5', '9/30')

    #Making the picks list into a dataframe
    picks = pd.DataFrame(picks)
    current_scores_status = pd.DataFrame(current_scores_status)

    #Setting the first row of the picks database to be the column names
    picks.columns = picks.iloc[0]
    picks = picks.drop(0)

    #current_scores_status.columns = current_scores_status.iloc[0]
    #current_scores_status = current_scores_status.drop(0)

    #changing the data in the away and home columns to be the team name only
    #current_scores_status["Away"] = current_scores_status["Away"].apply(sc.convert_to_abbreviation)

    #Addding new column to scores_og database called Ranked_game that is True if both teams are ranked and False if one or both teams are not ranked
    scores_og['Ranked_game'] = False
    for i in range(len(scores_og)):
        try :
            if int(scores_og.iloc[i,7]) > 0 and int(scores_og.iloc[i,8]) > 0:
                scores_og.iloc[i,9] = 1
            else:
                scores_og.iloc[i,9] = 0
        except:
            scores_og.iloc[i,9] = 0

    #Creating a database called home_data that contains the headers of the picks database starting at the 6th column (first column with a pick)
    pick_data = pd.DataFrame()
    column_headers = picks.columns[5:len(picks.columns)-1]
    pick_data['games'] = column_headers

    #Splitting the games column into two columns, one for the away team and one for the home team
    away_team = []
    home_team = []
    for i in range(len(pick_data)):
        away_team.append(pick_data.iloc[i,0].split(' @ ')[0])
        home_team.append(pick_data.iloc[i,0].split(' @ ')[1])
    pick_data['away_team'] = away_team
    pick_data['home_team'] = home_team

    home_data = scores_og[['Date', 'Home Team', 'Home Final', 'Home Rank Prior', 'Home Record After', 'Ranked_game']]
    away_data = scores_og[['Date', 'Away Team', 'Away Final', 'Away Rank Prior', 'Away Record After', 'Ranked_game']]

    #Taking the word 'Home' out of the home_data database columns
    home_data.columns = ['Date', 'Team', 'Final', 'Rank Prior', 'Record After', 'Ranked_game']
    away_data.columns = ['Date', 'Team', 'Final', 'Rank Prior', 'Record After', 'Ranked_game']

    #Combining the home_data and away_data databases into one database called scores
    scores = pd.concat([home_data, away_data], ignore_index = True)

    #Creating a database called winners that contains the winners of each game
    winners = pd.DataFrame()
    losers = pd.DataFrame()
    for i in range(len(scores)):
        if scores.iloc[i,2] > 0:
            winners = winners.append({'Date' : scores.iloc[i,0], 'Winner' : scores.iloc[i,1], 'Ranked_game': scores.iloc[i,5]}, ignore_index = True)
        elif scores.iloc[i,2] < 0:
            losers = losers.append({'Date' : scores.iloc[i,0], 'Loser' : scores.iloc[i,1], 'Ranked_game': scores.iloc[i,5]}, ignore_index = True)

    #Create a new database called game_stats that For each unique game in the picks database, starting at the 6th column (first column with a pick), check to see if the pick is in the home_data database. If it is, add 1 to the home_pick column and check to see if Home Final is positive. If it is add 1 to the correct column. If it is not add 1 to the incorrect column. If it is not, check the away_data database and add 1 to the away_pick column. If Away Final is positive, add 1 to the correct column. If it is not, add 1 to the incorrect column.
    game_stats = pd.DataFrame()
    for i in range(5, len(picks.columns)-3):
        away_team, home_team = picks.columns[i].split(' @ ')
        game_stats = game_stats.append({'Game' : picks.columns[i], 'Home Team' : home_team, 'Away Team' : away_team, 'Home Picks' : 0, 'Away Picks' : 0}, ignore_index = True)
        for j in range(len(picks)):
            if picks.iloc[j,i] in pick_data['home_team'].values:
                game_stats.iloc[i-5,3] += 1
            elif picks.iloc[j,i] in pick_data['away_team'].values:
                game_stats.iloc[i-5,4] += 1

    #Adding a true false column to the game_stats database that is True if the game is ranked and False if the game is not ranked
    scores_og['Game'] = scores_og['Away Team'] + ' @ ' + scores_og['Home Team']
    game_stats['Ranked_game'] = 0
    for i in range(len(game_stats)):
        if game_stats.iloc[i,0] in scores_og['Game'].values:
            game_stats.iloc[i,5] = scores_og.iloc[scores_og.index[game_stats.iloc[i,0] == scores_og['Game'].values][0], 9] 
        else:
            game_stats.iloc[i,5] = 0

    #Create a new database called player_stats that contains a concatenated column of First Name and Last Name from the picks database, the number of correct picks, and the number of incorrect picks.
    player_stats = pd.DataFrame()
    player_stats['Player'] = picks['First Name'] + ' ' + picks['Last Name']
    player_stats['Correct'] = 0
    player_stats['Incorrect'] = 0
    player_stats['Ranked_game_correct'] = 0

    #For each player in the picks database, check to see if their pick is in the winners database. If it is, add 1 to the correct column. If it is not, add 1 to the incorrect column.
    for i in range(5, len(picks.columns)-3):
        for j in range(len(picks)):
            if picks.iloc[j,i] in winners['Winner'].values:
                player_stats.iloc[j,1] += 1
                player_stats.iloc[j,3] += winners.iloc[winners.index[picks.iloc[j,i] == winners['Winner'].values][0],2]
            elif picks.iloc[j,i] in losers['Loser'].values:
                player_stats.iloc[j,2] += 1

    #Creating against the grain column that is a count of picks a player made that was against the most popular pick when the most popular pick takes up at least 75% of the picks
    player_stats['Against_the_grain'] = 0
    player_stats['Against_the_grain_picks'] = ''
    for i in range(5, len(picks.columns)-3):
        for j in range(len(picks)):
            if picks.iloc[j,i] in game_stats['Home Team'].values:
                if game_stats.iloc[game_stats.index[picks.iloc[j,i] == game_stats['Home Team'].values][0],3] <= .30*len(picks):
                    player_stats.iloc[j,4] += 1
                    player_stats.iloc[j,5] += picks.iloc[j,i] + ', '
            elif picks.iloc[j,i] in game_stats['Away Team'].values:
                if game_stats.iloc[game_stats.index[picks.iloc[j,i] == game_stats['Away Team'].values][0],4] <= .30*len(picks):
                    player_stats.iloc[j,4] += 1
                    player_stats.iloc[j,5] += picks.iloc[j,i] + ', '

    #################################################################################################################
    #Writing the player_stats and game_stats to new google sheets
    ################################################################################################################
    credentials_file = 'cfb-pickem-analysis-e52608a8acd7.json'

    # Authenticate with Google Sheets API using the credentials file
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    gc = gspread.authorize(credentials)

    # Open the Google Sheets spreadsheet by its title
    spreadsheet_title = "CFB Pickem Player Data"
    spreadsheet_title2 = "CFB Pickem Game Data"
    #spreadsheet_title3 = "Public Weekly 2023-24 NCAA Football Games"

    worksheet = gc.open(spreadsheet_title).sheet1
    worksheet2 = gc.open(spreadsheet_title2).sheet1
    #worksheet3 = gc.open(spreadsheet_title3).sheet1

    # Convert the DataFrame to a list of lists for Google Sheets
    data_to_update = player_stats.values.tolist()
    data_to_update2 = game_stats.values.tolist()
    #data_to_update3 = picks.values.tolist()

    #Adding column headers to the data_to_update list
    data_to_update.insert(0, ['Player', 'Correct', 'Incorrect', 'Ranked_game_correct', 'Against_the_grain', 'Against_the_grain_picks'])
    data_to_update2.insert(0, ['Game', 'Home Team', 'Away Team', 'Home Picks', 'Away Picks', 'Ranked_game'])

    # Clear existing data in the worksheet (optional)
    worksheet.clear()
    worksheet2.clear()

    # Update the Google Sheets worksheet with the new data
    worksheet.update("A1", data_to_update)  # Start writing data from cell A1
    worksheet2.update("A1", data_to_update2)  # Start writing data from cell A1

    #Write a print statement that says the data has been updated and the time it was updated
    print('Data updated at ' + str(datetime.datetime.now()))

scheduler = BlockingScheduler()
scheduler.add_job(my_task, 'interval', seconds=5)
scheduler.start()

#################################################################################################################
#Dash App
#################################################################################################################

#Setting up the Dash app
#app = dash.Dash(__name__)

#Setting up the layout of the Dash app
#app.layout = html.Div([
#    html.H1("Player Leaderboard"),
    
    # Table to display the leaderboard
#    html.Table([
#        html.Tr([html.Th("Player"), html.Th("Correct")]),
#        html.Tbody(id='leaderboard-table')
#    ])
#])

# Callback to update the leaderboard table
#@app.callback(
#    Output('leaderboard-table', 'children'),
#    [Input('leaderboard-table', 'id')]
#)

#def update_leaderboard(_):
    # Sort the player_stats DataFrame by the "Correct" column in descending order
#    leaderboard_df = player_stats.sort_values(by='Correct', ascending=False)
    
    # Create table rows dynamically
#    rows = []
#    for _, row in leaderboard_df.iterrows():
#        rows.append(html.Tr([html.Td(row['Player']), html.Td(row['Correct'])]))
    
#    return rows

# Run the Dash app
#if __name__ == '__main__':
#    app.run_server(debug=True)
