# helper.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

matches = matches[matches['result'] == 'normal']
matches = matches[matches['dl_applied'] == 0]
matches.dropna(inplace=True)

combined = pd.merge(deliveries, matches[['id', 'city', 'date', 'venue', 'winner']], left_on='match_id', right_on='id')

innings1 = combined[combined['inning'] == 1]
innings1_grouped = innings1.groupby('match_id').agg({
    'total_runs': 'sum'
}).reset_index().rename(columns={'total_runs': 'target'})

innings2 = combined[combined['inning'] == 2]
innings2 = pd.merge(innings2, innings1_grouped, on='match_id')

innings2['current_score'] = innings2.groupby('match_id')['total_runs'].cumsum()
innings2['wickets'] = innings2['player_dismissed'].notnull().astype(int)
innings2['wickets'] = innings2.groupby('match_id')['wickets'].cumsum()
innings2['ball_number'] = ((innings2['over'] - 1) * 6 + innings2['ball'])

df_model = innings2[innings2['ball_number'] == 30].copy()

df_model['balls_left'] = 120 - df_model['ball_number']
df_model['runs_left'] = df_model['target'] - df_model['current_score']
df_model['wickets_left'] = 10 - df_model['wickets']
df_model['crr'] = df_model['current_score'] / (df_model['ball_number'] / 6)
df_model['rrr'] = (df_model['runs_left'] * 6) / df_model['balls_left']

df_model = df_model[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left',
                     'target', 'crr', 'rrr', 'winner']].dropna()

le_team = LabelEncoder()
le_city = LabelEncoder()
le_winner = LabelEncoder()

df_model['batting_team'] = le_team.fit_transform(df_model['batting_team'])
df_model['bowling_team'] = le_team.transform(df_model['bowling_team'])
df_model['city'] = le_city.fit_transform(df_model['city'])
df_model['winner'] = le_winner.fit_transform(df_model['winner'])

X = df_model.drop('winner', axis=1)
y = df_model['winner']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

teams = list(le_team.classes_)
cities = list(le_city.classes_)

def get_teams():
    return teams

def get_cities():
    return cities

def predict_win_probability(batting_team, bowling_team, city, target, score, overs, wickets_out):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets = 10 - wickets_out
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame([{
        'batting_team': le_team.transform([batting_team])[0],
        'bowling_team': le_team.transform([bowling_team])[0],
        'city': le_city.transform([city])[0] if city in cities else 0,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets,
        'target': target,
        'crr': crr,
        'rrr': rrr
    }])

    probs = model.predict_proba(input_df)[0]
    batting_idx = le_team.transform([batting_team])[0]
    win_prob = probs[batting_idx]
    return {
        "batting_team_win": round(win_prob * 100, 2),
        "bowling_team_win": round(100 - win_prob * 100, 2)
    }

def plot_toss_decision():
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=matches, x='toss_decision', palette='pastel', ax=ax)
    ax.set_title('Toss Decision (Field or Bat)', fontsize=12)
    return fig

def plot_top_teams():
    fig, ax = plt.subplots(figsize=(6, 4))
    matches['winner'].value_counts().head(10).plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title('Top 10 Winning Teams', fontsize=12)
    return fig

def plot_win_by_batting_first():
    df_result = matches[matches['result'] == 'normal'].copy()
    df_result['win_by'] = df_result.apply(lambda row: 'Batting First' if row['win_by_runs'] > 0 else 'Chasing', axis=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    df_result['win_by'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                                colors=['#ff9999','#66b3ff'], ax=ax)
    ax.set_ylabel('')
    ax.set_title('Bat First vs Chase Win %', fontsize=12)
    return fig

def plot_matches_per_city():
    fig, ax = plt.subplots(figsize=(7, 4))
    matches['city'].value_counts().plot(kind='bar', color='lightgreen', ax=ax)
    ax.set_title('Matches Played per City', fontsize=12)
    ax.set_ylabel('Match Count')
    return fig

def plot_toss_winner_vs_match_winner():
    toss_match = matches[matches['toss_winner'] == matches['winner']]
    fig, ax = plt.subplots(figsize=(5, 4))
    values = [len(toss_match), len(matches) - len(toss_match)]
    labels = ['Toss Winner = Match Winner', 'Toss Loser = Match Winner']
    colors = ['#90ee90', '#ffcccb']
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Toss Winning Impact', fontsize=12)
    return fig

def plot_result_type_distribution():
    fig, ax = plt.subplots(figsize=(5, 4))
    matches['result'].value_counts().plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Match Result Types', fontsize=12)
    return fig

def plot_win_margin_distribution():
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(matches[matches['win_by_runs'] > 0]['win_by_runs'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Win Margin by Runs (when Batting First)', fontsize=12)
    ax.set_xlabel('Win by Runs')
    return fig

def plot_win_wickets_distribution():
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(matches[matches['win_by_wickets'] > 0]['win_by_wickets'], bins=10, kde=True, color='coral', ax=ax)
    ax.set_title('Win Margin by Wickets (when Chasing)', fontsize=12)
    ax.set_xlabel('Win by Wickets')
    return fig
