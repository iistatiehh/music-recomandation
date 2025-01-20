from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import joblib

# Load models and data
nn_combined = joblib.load(r"C:\Users\asus\Desktop\models\nn_combined_model.pkl")
dataset = pd.read_csv(r'C:\Users\asus\Desktop\models\Spotify_Song_Attributes_with_clusters.csv', encoding='ISO-8859-1')
combined_features = np.load(r"C:\Users\asus\Desktop\models\combined_features.npy")

app = FastAPI()

def get_recommendations(song_name, dataset, combined_features, nn_model, n_recommendations=10):
    """
    Generate recommendations based on combined weighted genre + audio features.

    Parameters:
    - song_name: Name of the song to base recommendations on.
    - dataset: The dataset containing song information.
    - combined_features: Combined weighted features used for clustering.
    - nn_model: Pretrained NearestNeighbors model.
    - n_recommendations: Number of recommendations.
    """
    # Find the song index based on the song name
    song_index = dataset[dataset['trackName'] == song_name].index

    # Ensure the song exists in the dataset
    if len(song_index) == 0:
        return {"error": f"Song '{song_name}' not found in the dataset."}

    song_index = song_index[0]

    # Get Nearest Neighbors for combined features
    distances, indices = nn_model.kneighbors(combined_features[song_index].reshape(1, -1), n_neighbors=n_recommendations + 1)
    
    # Exclude the input song itself
    recommendations_indices = indices[0][1:]
    recommendations = dataset.iloc[recommendations_indices]

    # Remove duplicates
    recommendations = recommendations.drop_duplicates(subset=['trackName', 'artistName'])

    return {
        "recommendations": recommendations[["trackName", "artistName", "combined_cluster"]].to_dict(orient="records")
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Music Recommendation System</title>
    </head>
    <body>
        <h2>Get Song Recommendations</h2>
        <form action="/recommend" method="post">
            <label>Enter Song Name:</label><br>
            <input type="text" name="song_name" placeholder="e.g., Shape of You" required>
            <br><br>
            <button type="submit">Get Recommendations</button>
        </form>
    </body>
    </html>
    """

@app.post("/recommend", response_class=HTMLResponse)
def recommend(song_name: str = Form(...)):
    recommendations = get_recommendations(song_name, dataset, combined_features, nn_combined)

    if "error" in recommendations:
        return f"<h3>{recommendations['error']}</h3>"

    html = "<h4>Recommendations:</h4><ul>"
    for rec in recommendations["recommendations"]:
        html += f"<li>{rec['trackName']} by {rec['artistName']} (Combined Cluster: {rec['combined_cluster']})</li>"
    html += "</ul>"

    return f"""
    <html>
    <body>
        <h2>Recommendations for '{song_name}':</h2>
        {html}
        <a href="/">Go Back</a>
    </body>
    </html>
    """
