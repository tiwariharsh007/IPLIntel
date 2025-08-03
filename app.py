# app.py
import streamlit as st
from helper import (
    get_teams, get_cities, predict_win_probability,
    plot_toss_decision, plot_top_teams, plot_win_by_batting_first,
    plot_matches_per_city, plot_toss_winner_vs_match_winner,
    plot_result_type_distribution, plot_win_margin_distribution,
    plot_win_wickets_distribution
)

st.set_page_config(page_title="IPL Win Predictor & Analysis", layout="wide")
st.title('🏏 IPL Win Predictor & Match Analysis')

tab1, tab2 = st.tabs(["📈 Win Predictor", "📊 Match Analysis"])

with tab1:
    st.header("Enter Match Situation")

    teams = get_teams()
    cities = get_cities()

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select Batting Team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select Bowling Team', sorted([team for team in teams if team != batting_team]))

    selected_city = st.selectbox('Select Host City', sorted(cities))
    target = st.number_input('Target Score', min_value=1)

    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Current Score', min_value=0)
    with col4:
        overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
    with col5:
        wickets_out = st.number_input('Wickets Fallen', min_value=0, max_value=10)

    if st.button('Predict Probability'):
        prediction = predict_win_probability(
            batting_team, bowling_team, selected_city,
            target, score, overs, wickets_out
        )

        st.subheader("🏁 Winning Probabilities")
        st.success(f"**{batting_team}** Winning Probability: **{prediction['batting_team_win']}%**")
        st.error(f"**{bowling_team}** Winning Probability: **{prediction['bowling_team_win']}%**")

with tab2:
    st.header("📊 IPL Match Insights (2008–2020)")

    st.subheader("1. Toss Decision Distribution")
    st.pyplot(plot_toss_decision())

    st.subheader("2. Top 10 Winning Teams")
    st.pyplot(plot_top_teams())

    st.subheader("3. Win by Batting First vs Chasing")
    st.pyplot(plot_win_by_batting_first())

    st.subheader("4. Number of Matches per City")
    st.pyplot(plot_matches_per_city())

    st.subheader("5. Toss Winner vs Match Winner")
    st.pyplot(plot_toss_winner_vs_match_winner())

    st.subheader("6. Match Result Types")
    st.pyplot(plot_result_type_distribution())

    st.subheader("7. Distribution of Win by Runs (Batting First)")
    st.pyplot(plot_win_margin_distribution())

    st.subheader("8. Distribution of Win by Wickets (Chasing)")
    st.pyplot(plot_win_wickets_distribution())
