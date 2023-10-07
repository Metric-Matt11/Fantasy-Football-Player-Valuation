import pandas as pd
import re
import csv
import os
from numpy.linalg import inv
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Create a dictionary mapping school names to abbreviations
school_name_to_abbreviation = {
    'ALA': 'University of Alabama',
    'UGA': 'University of Georgia',
    'OSU': 'Ohio State University',
    'LSU': 'Louisiana State University',
    'CLEM': 'Clemson University',
    'OU': 'Oklahoma',
    'TEX': 'University of Texas',
    'ND': 'University of Notre Dame',
    'MICH': 'University of Michigan',
    'FLA': 'University of Florida',
    'AUB': 'Auburn University',
    'TENN': 'University of Tennessee',
    'ORE': 'University of Oregon',
    'PENNST': 'Pennsylvania State University',
    'WASH': 'University of Washington',
    'FSU': 'Florida State University',
    'NEB': 'University of Nebraska',
    'WIS': 'University of Wisconsin',
    'USC': 'University of Southern California',
    'MIA': 'University of Miami',
    'OKST': 'Oklahoma State University',
    'VT': 'Virginia Tech',
    'UGA': 'University of Georgia',
    'TCU': 'Texas Christian University',
    'IOWA': 'University of Iowa',
    'MSST': 'Mississippi State University',
    'KSU': 'Kansas State University',
    'LOU': 'University of Louisville',
    'ARIZ': 'University of Arizona',
    'UCLA': 'University of California, Los Angeles',
    'ARK': 'University of Arkansas',
    'MINN': 'University of Minnesota',
    'STAN': 'Stanford University',
    'UCF': 'University of Central Florida',
    'USF': 'University of South Florida',
    'WVU': 'West Virginia University',
    'ISU': 'Iowa State University',
    'UK': 'University of Kentucky',
    'UNC': 'University of North Carolina at Chapel Hill',
    'MSU': 'Michigan State University',
    'TAMU': 'Texas A&M University',
    'OREST': 'Oregon State University',
    'NW': 'Northwestern University',
    'UVA': 'University of Virginia',
    'ILL': 'University of Illinois at Urbana-Champaign',
    'TTU': 'Texas Tech University',
    'UAB': 'University of Alabama at Birmingham',
    'ASU': 'Arizona State University',
    'CU': 'Clemson University',
    'UConn': 'University of Connecticut',
    'DUKE': 'Duke University',
    'FAU': 'Florida Atlantic University',
    'FIU': 'Florida International University',
    'FSU': 'Florida State University',
    'GT': 'Georgia Institute of Technology',
    'HAW': 'University of Hawaiʻi at Mānoa',
    'HOUST': 'University of Houston',
    'IDAHO': 'University of Idaho',
    'IU': 'Indiana University Bloomington',
    'ISU': 'Indiana State University',
    'KSU': 'Kansas State University',
    'KU': 'University of Kansas',
    'KENT': 'Kent State University',
    'LOU': 'University of Louisville',
    'MD': 'University of Maryland',
    'UMASS': 'University of Massachusetts Amherst',
    'MEM': 'University of Memphis',
    'UMICH': 'University of Michigan',
    'MSU': 'Michigan State University',
    'MIZZ': 'Missouri',
    'MTSU': 'Middle Tennessee State University',
    'UH': 'University of Houston',
    'UO': 'University of Oregon',
    'UNO': 'University of Nebraska Omaha',
    'NCST': 'North Carolina State University',
    'UNCA': 'University of North Carolina at Asheville',
    'UNCC': 'University of North Carolina at Charlotte',
    'UNCG': 'University of North Carolina at Greensboro',
    'UNCP': 'University of North Carolina at Pembroke',
    'UND': 'University of North Dakota',
    'UNF': 'University of North Florida',
    'UNH': 'University of New Hampshire',
    'UNI': 'University of Northern Iowa',
    'UNK': 'University of Nebraska at Kearney',
    'UNL': 'University of Nebraska–Lincoln',
    'UNLV': 'University of Nevada, Las Vegas',
    'UNM': 'University of New Mexico',
    'UOF': 'University of Findlay',
    'UP': 'University of Portland',
    'USC': 'University of Southern California',
    'USF': 'University of South Florida',
    'USM': 'University of Southern Mississippi',
    'USNA': 'United States Naval Academy',
    'USU': 'Utah State University',
    'USUAA': 'University of South Alabama',
    'UT': 'University of Toledo',
    'UTA': 'University of Texas at Arlington',
    'UTEP': 'University of Texas at El Paso',
    'UTSA': 'University of Texas at San Antonio',
    'UVM': 'University of Vermont',
    'UW': 'University of Wyoming',
    'UWF': 'University of West Florida',
    'UWG': 'University of West Georgia',
    'UWP': 'University of Wisconsin–Platteville',
    'UWSP': 'University of Wisconsin–Stevens Point',
    'UWW': 'University of Wisconsin–Whitewater',
    'UWS': 'University of Wisconsin–Superior',
    'UWSD': 'University of Wisconsin System',
    'UWV': 'University of Wisconsin–Eau Claire',
    'UWRF': 'University of Wisconsin–River Falls',
    'UWGB': 'University of Wisconsin–Green Bay',
    'UWC': 'University of Wisconsin–Colleges',
    'UWEX': 'University of Wisconsin–Extension',
    'UWO': 'University of Wisconsin–Oshkosh',
    'UWL': 'University of Wisconsin–La Crosse',
    'UWM': 'University of Wisconsin–Milwaukee',
    'UWR': 'University of Wisconsin–Richland',
    'UWSA': 'University of Wisconsin–Stout',
    'UWSP': 'University of Wisconsin–Stevens Point',
    'UWV': 'University of Wisconsin–Eau Claire',
    'UWW': 'University of Wisconsin–Whitewater',
    'UW': 'University of Washington',
    'UWO': 'University of Wisconsin–Oshkosh',
    'UWSP': 'University of Wisconsin–Stevens Point',
    'UWV': 'University of Wisconsin–Eau Claire',
    'UWW': 'University of Wisconsin–Whitewater',
    'UWS': 'University of Wisconsin–Superior',
    'UWSD': 'University of Wisconsin System',
    'UWRF': 'University of Wisconsin–River Falls',
    'UWGB': 'University of Wisconsin–Green Bay',
    'UWC': 'University of Wisconsin–Colleges',
    'UWEX': 'University of Wisconsin–Extension',
    'UWO': 'University of Wisconsin–Oshkosh',
    'UWL': 'University of Wisconsin–La Crosse',
    'UWM': 'University of Wisconsin–Milwaukee',
    'UWR': 'University of Wisconsin–Richland',
    'UWSA': 'University of Wisconsin–Stout',
    'UWSP': 'University of Wisconsin–Stevens Point',
    'UWV': 'University of Wisconsin–Eau Claire',
    'UWW': 'University of Wisconsin–Whitewater',
    'UWS': 'University of Wisconsin–Superior',
    'UWSD': 'University of Wisconsin System',
    'UWRF': 'University of Wisconsin–River Falls',
    'UWGB': 'University of Wisconsin–Green Bay',
    'UWC': 'University of Wisconsin–Colleges',
    'UWEX': 'University of Wisconsin–Extension',
    'UWO': 'University of Wisconsin–Oshkosh',
    'UWL': 'University of Wisconsin–La Crosse',
    'UWM': 'University of Wisconsin–Milwaukee',
    'UWR': 'University of Wisconsin–Richland',
    'UWSA': 'University of Wisconsin–Stout',
    'VANDY': 'Vanderbilt University',
    'WF': 'Wake Forest University',
    'WCU': 'Western Carolina University',
    'WSU': 'Washington State University',
    'WVU': 'West Virginia University',
    'WYO': 'University of Wyoming',
    'YALE': 'Yale University',
    'YSU': 'Youngstown State University',
    'SYR': 'Syracuse University',
    'TAMU': 'Texas A&M University',
    'FRES': 'California State University, Fresno',
    'ORST': 'Oregon State University'
    # Add more mappings as needed
}

inpath = os.path.join(os.getcwd(), 'teams.csv')
fbs = []
with open(inpath, 'r') as f:
    for row in f:
        fbs.append(str(row).strip())

def team_map(name):
    lower = name.lower()
    
    if lower in ['akr', 'akron']:
        lower = 'akron'
    elif lower in ['arizona st.', 'arizona state']:
        lower = 'arizona state'
    elif lower in ['arkansas st.', 'arkansas state']:
        lower = 'arkansas state'
    elif lower in ['aub', 'auburn']:
        lower = 'auburn'
    elif lower in ['ball st.', 'ball state', 'ball']:
        lower = 'ball state'
    elif lower in ['bgsu', 'bowling green']:
        lower = 'bowling green'
    elif lower in ['brigham young', 'byu']:
        lower = 'brigham young'
    elif lower in ['buff', 'buffalo']:
        lower = 'buffalo'
    elif lower in ['char', 'charlotte']:
        lower = 'charlotte'
    elif lower in ['cmu', 'c. michigan', 'central michigan']:
        lower = 'central michigan'
    elif lower in ['clem', 'clemson']:
        lower = 'clemson'
    elif lower in ['colorado state', 'csu']:
        lower = 'colorado state'
    elif lower in ['e. michigan', 'emu', 'eastern michigan']:
        lower = 'eastern michigan'
    elif lower in ['fresno st.', 'fresno state']:
        lower = 'fresno state'
    elif lower in ['ga. southern', 'gaso']:
        lower = 'georgia southern'
    elif lower in ['georgia state', 'gast']:
        lower = 'georgia state'
    elif lower in ['idho', 'idaho']:
        lower = 'idaho'
    elif lower in ['ill', 'illinois']:
        lower = 'illinois'
    elif lower in ['kansas state', 'ksu', 'kansas st.']:
        lower = 'kansas state'
    elif lower in ['kentucky', 'ky']:
        lower = 'kentucky'
    elif lower in ['m. tenn. st.', 'mtsu']:
        lower = 'middle tennessee state'
    elif lower in ['mass', 'massachusetts']:
        lower = 'massachusetts'
    elif lower in ['umoh', 'miami (ohio)']:
        lower = 'miami (ohio)'
    elif lower in ['michigan', 'um']:
        lower = 'michigan'
    elif lower in ['mississippi state', 'msst']:
        lower = 'mississippi state'
    elif lower in ['new mexico st.', 'nmsu', 'nmst']:
        lower = 'new mexico state'
    elif lower in ['san diego state', 'san diego st.', 'sdsu']:
        lower = 'san diego state'
    elif lower in ['san jose state', 'san jose st.', 'sjsu']:
        lower = 'san jose state'
    elif lower in ['south alabama', 'usm']:
        lower = 'south alabama'
    elif lower in ['ul lafayette', 'ull', 'laf']:
        lower = 'ul lafayette'
    elif lower in ['ul monroe', 'ulm', 'la.-monroe']:
        lower = 'ul monroe'
    elif lower in ['unt', 'north texas']:
        lower = 'north texas'
    elif lower in ['stan', 'stanford']:
        lower = 'stanford'
    elif lower in ['tulsa', 'tlsa']:
        lower = 'tulsa'
    elif lower in ['tul', 'tulane']:
        lower = 'tulane'
    elif lower in ['texas-el paso', 'utep']:
        lower = 'utep'
    elif lower in ['w. kentucky', 'wky']:
        lower = 'western kentucky'
    elif lower in ['washington st.', 'wsu']:
        lower = 'washington state'
    elif lower in ['w. michigan', 'wmu']:
        lower = 'western michigan'
    elif lower not in fbs:
        # function to return all fcs teams (to ensure that they're in fact fcs)
        lower = "fcs_team"

    return lower

#Creating a function so it is callable in other scripts
def get_past_scores(week_range, year_range):

# Collect URLs to scrape

    urls = []

    for year in year_range:
        for week in week_range:
            urls.append('http://www.cbssports.com/college-football/scoreboard/FBS/' + str(year) + '/regular/' + str(week))

# Scrape and compile scores

    weeks = []

    for url in urls:
        frame = pd.read_html(url)
        fb_week = url.split('regular/')[1]
    
        this_year_arr = re.split('\D', url)
        this_year = list(filter(lambda v : len(v) > 0, this_year_arr))
        year = this_year[0]
    
        game_frame = pd.DataFrame({'Date' : [], 'Away Team' : [], 'Home Team' : [], 
        'Away Final' : [], 'Home Final' : [], 'Away Record After' : [], 'Home Record After' : [], 
        'Away Rank Prior' : [], 'Home Rank Prior' : []})

        games1 = map(lambda x : x, filter(lambda y : len(y) > 1, frame))

        try:
            games = map(lambda x : x, filter(lambda y : len(y.columns) >= 6, games1))
        except:
            games = games1

        for game in games:
            def collect_info(game, num):
                game_string = game.iloc[num,0]
                try:
                    rank = int(game_string[:2])
                except ValueError:
                    try:
                        rank = int(game_string[:1])
                    except ValueError:
                        rank = "Unranked"
                record = game_string[-3:]
                if len(str(rank)) == 2:
                    team = game_string[3:-3]
                elif len(str(rank)) == 1:
                    team = game_string[2:-3]
                else:
                    team = game_string[:-3]
                wins = record.split('-')[0]; losses = record.split('-')[1]

                final_margin = game.iloc[num,5] - game.iloc[(num+1)%2,5]
                #team = team_map(team)
                return str(team), str(rank), str(wins), str(losses), int(final_margin)

            try:
                away_team, away_rank_prior, away_wins, away_losses, away_margin = collect_info(game, 0)
                home_team, home_rank_prior, home_wins, home_losses, home_margin = collect_info(game, 1)

                date = "Week " + fb_week + ", " + year

                game_frame = game_frame.append({'Date' : date, 'Away Team' : away_team, 'Home Team' : home_team, 
                'Away Final' : away_margin, 'Home Final' : home_margin, 
                'Away Record After' : "(" + away_wins + "-" + away_losses + ")", 'Home Record After' : "(" + home_wins + "-" + home_losses + ")",
                'Away Rank Prior' : away_rank_prior, 'Home Rank Prior' : home_rank_prior}, ignore_index=True)
        
            except:
                print("Error")
                print(game)
                print()
                pass

    
        game_frame = game_frame[['Date', 'Away Team' , 'Home Team' , 
        'Away Final' , 'Home Final' ,
        'Away Record After', 'Home Record After',
        'Away Rank Prior', 'Home Rank Prior']]

        weeks.append(game_frame)
    
    final = pd.DataFrame()
    for week in weeks:
        final = pd.concat([final, week], ignore_index = True)
    
    return final

def get_current_picks(week_number, week_date):
    # Set the path to your credentials file (JSON)
    credentials_file = 'cfb-pickem-analysis-e52608a8acd7.json'

    # Set the name of the Google Sheets document and the sheet you want to scrape
    spreadsheet_name = 'CFB Pickem Week '+ week_number + ' (' + week_date +') (Responses)'
    spreadsheet_title3 = "Public Weekly 2023-24 NCAA Football Games"
    sheet_name = 'Form Responses 1'
    sheet_name3 = 'Live Scoring'

    # Authenticate with Google Sheets API using the credentials file
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    gc = gspread.authorize(credentials)

    # Open the spreadsheet by name
    spreadsheet = gc.open(spreadsheet_name)
    spreadsheet3 = gc.open(spreadsheet_title3)

    # Open the specific sheet by name
    worksheet = spreadsheet.worksheet(sheet_name)
    worksheet3 = spreadsheet3.worksheet(sheet_name3)

    # Get all values from the worksheet
    data = worksheet.get_all_values()
    data_games = worksheet3.get_all_values()

    return data, data_games

def convert_to_abbreviation(school_name):
    return school_name_to_abbreviation.get(school_name, school_name)

############################################################################################################################################

# Write scores to 

#outpath = os.path.join(os.getcwd(), 'all_weeks.csv')
#final.to_csv(outpath, index = False, header = True)

############################################################################################################################################

# Read back in the scores

#outpath = os.path.join(os.getcwd(), 'all_weeks.csv')
#final = pd.read_csv(outpath)

############################################################################################################################################

