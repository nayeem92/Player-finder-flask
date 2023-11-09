from flask import Flask, request, render_template, jsonify 
import pandas as pd
import plotly.graph_objects as go
import os
import joblib

app = Flask(__name__)

# Load your filtered data
new_df = pd.read_csv("data/filtered_data.csv")

# Set OMP_NUM_THREADS to 1 to avoid memory leak with K-Means and MKL
os.environ['OMP_NUM_THREADS'] = '1'

# Load the saved K-Means model
kmeans = joblib.load('kmeans_model.pkl')

def get_cluster_info(player_name, data_frame):
    try:
        player_data = data_frame[data_frame['Player'] == player_name]
        if player_data.empty:
            return "Player not found"

        cluster_number = player_data['Cluster'].values[0]

        # Create a scatter plot for the player's cluster
        cluster_data = data_frame[data_frame['Cluster'] == cluster_number]

        fig = go.Figure()

        # Create a scatter plot for players in the cluster, excluding the player of interest
        other_players = cluster_data[cluster_data['Player'] != player_name]
        fig.add_trace(go.Scatter(
            x=other_players['SCA'],
            y=other_players['GCA'],
            mode='markers',
            text=other_players['Player'],  # Display player names on hover
            marker=dict(
                color='light blue',
                size=12,
                line=dict(
                    color='black',
                    width=1
                ))
        ))

        # Create a scatter plot for the player of interest and color it differently
        fig.add_trace(go.Scatter(
            x=player_data['SCA'],
            y=player_data['GCA'],
            mode='markers',
            text=player_data['Player'],  # Display player name on hover
            marker=dict(
                color='red',  # Highlight the player with a different color
                size=12,
                line=dict(
                    color='black',
                    width=1
                ))
        ))

        # Customize the layout of the scatter plot
        fig.update_layout(
            title=f"GCA vs SCA for Players in Cluster {cluster_number}",
            xaxis_title="SCA (Shot-Creating Actions)",
            yaxis_title="GCA (Goal-Creating Actions)",
            showlegend=False  # Hide the legend
        )

        # Get the list of other player names in the same cluster
        other_player_names = other_players['Player'].tolist()

        # Show the scatter plot
        return f"{player_name} is in cluster {cluster_number}", fig.to_html(), other_player_names
    except IndexError:
        return "Player not found", None

@app.route('/')
def home():
    # Pass the player names from the dataset to the template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    player_name = request.form['player_name']
    result, plot_html, other_players = get_cluster_info(player_name, new_df)
    return render_template('result.html', result=result, plot_html=plot_html, other_players=other_players)

@app.route('/get_player_names', methods=['GET'])
def get_player_names():
    player_names = new_df['Player'].tolist()
    return jsonify(player_names)

if __name__ == '__main__':
    app.run(debug=True)
