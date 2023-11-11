from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.graph_objects as go
import os
import joblib

app = Flask(__name__)

# CSV file path
csv_path = "data/filtered_data.csv"

# Load your filtered data
new_df = pd.read_csv(csv_path)

# Set OMP_NUM_THREADS to 1 to avoid memory leak with K-Means and MKL
os.environ['OMP_NUM_THREADS'] = '1'

# Load the saved K-Means model
kmeans = joblib.load('kmeans_model.pkl')

def update_clusters(data_frame, kmeans_model):
    # Extract relevant features for clustering
    sca_features = ['Pass SCA Ratio', 'Deadball SCA Ratio', 'Dribble SCA Ratio', 'Shot SCA Ratio', 'Fouls Drawn SCA Ratio', 'Defense SCA Ratio']
    features = data_frame[sca_features]

    # Predict clusters using the pre-trained K-means model
    clusters = kmeans_model.predict(features)

    # Update the 'Cluster' column in the DataFrame
    data_frame['Cluster'] = clusters

    # Save the updated DataFrame back to the CSV file
    data_frame.to_csv(csv_path, index=False)

def get_cluster_info(player_name, data_frame, kmeans_model):
    # Update the clusters before getting cluster info
    update_clusters(data_frame, kmeans_model)

    try:
        player_data = data_frame[data_frame['Player'] == player_name]
        if player_data.empty:
            return "Player not found", None, None

        # Extract relevant features for clustering (using the specified features)
        sca_features = ['Pass SCA Ratio', 'Deadball SCA Ratio', 'Dribble SCA Ratio', 'Shot SCA Ratio', 'Fouls Drawn SCA Ratio', 'Defense SCA Ratio']
        player_features = player_data[sca_features]

        # Predict the cluster for the player using the pre-trained model
        cluster_number = kmeans_model.predict(player_features)[0]

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
        return "Player not found", None, None

@app.route('/')
def home():
    # Pass the player names from the dataset to the template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    player_name = request.form['player_name']

    # Get cluster information for the specified player
    result, plot_html, other_players = get_cluster_info(player_name, new_df, kmeans)

    return render_template('result.html', result=result, plot_html=plot_html, other_players=other_players)

@app.route('/get_player_names', methods=['GET'])
def get_player_names():
    player_names = new_df['Player'].tolist()
    return jsonify(player_names)

if __name__ == '__main__':
    app.run(debug=True)
