# Quant_Models
football-score-simulator
"""# Football Score Simulator using Poisson Model

This project simulates football match outcomes using the Poisson distribution based on historical data.

## Features
- Predict goals scored by each team using a statistical model
- Generate score matrices
- Visualize probabilities for match outcomes

## How It Works
- Each team's average goal scoring and conceding rate is calculated.
- The Poisson distribution is used to model goal scoring probabilities.
- Match outcomes are simulated using these probabilities.
## Setup
```bash
pip install -r requirements.txt
```

## Run the Model
```bash
python poisson_model.py
```

##  Run Streamlit App
```bash
streamlit run streamlit_app.py
```

## Data
structured football match dataset (from football-data.co.uk ).

---

Developed by Pranjnay Bhardwaj
""",

    "requirements.txt": 
"""pandas
numpy
scipy
matplotlib
seaborn
streamlit
""",

    "poisson_model.py": 
"""import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset 
data = pd.read_csv('data/epl_matches.csv')

# Preprocessing example
data = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
data.columns = ['home_team', 'away_team', 'home_goals', 'away_goals']

# Calculate average goals
avg_home_goals = data.groupby('home_team')['home_goals'].mean()
avg_away_goals = data.groupby('away_team')['away_goals'].mean()

# Create a function to predict scores
def predict_match(home_team, away_team):
    lambda_home = avg_home_goals[home_team]
    lambda_away = avg_away_goals[away_team]
    
    prob_matrix = np.outer(
        poisson.pmf(range(6), lambda_home),
        poisson.pmf(range(6), lambda_away)
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.xlabel(f'{away_team} Goals')
    plt.ylabel(f'{home_team} Goals')
    plt.title('Score Probability Matrix')
    plt.show()

# Example usage
predict_match('Manchester United', 'Chelsea')
""",

    "streamlit_app.py":
"""import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/epl_matches.csv')
data = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
data.columns = ['home_team', 'away_team', 'home_goals', 'away_goals']

avg_home_goals = data.groupby('home_team')['home_goals'].mean()
avg_away_goals = data.groupby('away_team')['away_goals'].mean()

teams = sorted(data['home_team'].unique())
home = st.selectbox("Select Home Team", teams)
away = st.selectbox("Select Away Team", teams)

if home != away:
    lambda_home = avg_home_goals[home]
    lambda_away = avg_away_goals[away]

    prob_matrix = np.outer(
        poisson.pmf(range(6), lambda_home),
        poisson.pmf(range(6), lambda_away)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    ax.set_xlabel(f"{away} Goals")
    ax.set_ylabel(f"{home} Goals")
    ax.set_title('Score Probability Matrix')
    st.pyplot(fig)
else:
    st.warning("Select two different teams.")
""",




