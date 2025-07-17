
# Hybrid Music Recommendation System
Project Documentaion: 
[Music recommendation system documentation.pdf](https://github.com/user-attachments/files/21298753/Music.recommendation.system.documentation.pdf)

A machine learning-based music recommendation system that combines audio feature analysis with genre clustering to provide personalized and diverse music suggestions.

## Overview

This project implements a hybrid approach to music recommendation using Spotify song data. By combining audio content analysis with genre-based categorization, the system delivers balanced recommendations that help users discover both familiar and new music.

## Features

- **Hybrid Clustering**: Combines K-Means++ (audio features) and Hierarchical Clustering (genres)
- **Weighted Similarity**: Uses 80% audio/genre similarity and 20% artist preference
- **Duplicate Removal**: Ensures diverse recommendations by removing duplicate tracks
- **Similarity Scoring**: Provides transparency with similarity metrics for each recommendation
- **Cluster Visualization**: t-SNE visualization for combined clusters

## Dataset

The system uses the Spotify Song Attributes dataset (~11,000 tracks) with features including:
- Audio features: danceability, energy, valence, tempo, loudness, etc.
- Track metadata: song name, artist, genre
- Musical attributes: key, mode, speechiness, acousticness

## Methodology

### 1. Audio Features Clustering
- Uses Elbow Method to find optimal cluster count
- Applies K-Means++ algorithm for better centroid initialization
- Groups songs by musical characteristics

### 2. Genre-Based Clustering
- Hierarchical Clustering for genre and subgenre grouping
- Captures nested structure of music categories
- Provides detailed music categorization

### 3. Hybrid Recommendation
- Combines both clustering results using weighted similarity
- Filters candidates by combined cluster membership
- Ranks recommendations using cosine similarity and artist matching

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-recommendation-system.git
```

## Usage

```python
# Load your processed dataset and combined features
from recommendation_system import get_recommendations

# Get recommendations for a song
get_recommendations(
    song_name="Last Friday Night (T.G.I.F.)", 
    dataset=dataset, 
    combined_features=combined_features, 
    n_recommendations=15
)
```

### Sample Output
```
Finding recommendations for 'Last Friday Night (T.G.I.F.)' by Katy Perry in cluster 2

Top 15 recommendations:
1. Teenage Dream by Katy Perry (similarity: 0.892) (Same Artist)
2. Good 4 U by Olivia Rodrigo (similarity: 0.847)
3. Shake It Off by Taylor Swift (similarity: 0.823)
...
```

## Key Functions

### `get_recommendations(song_name, dataset, combined_features, n_recommendations=15)`
Main recommendation function that:
- Finds the input song in the dataset
- Filters songs by combined cluster
- Calculates weighted similarity scores
- Returns ranked recommendations with similarity metrics

## Evaluation Metrics

The system uses multiple metrics to evaluate clustering quality:
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Calinski-Harabasz Index**: Evaluates cluster definition (higher is better)
- **Silhouette Score**: Measures clustering performance (-1 to 1, higher is better)

## Results

The hybrid approach demonstrates:
- Better balance between similarity and diversity
- Improved accuracy over single-method systems
- Reduced redundancy through intelligent filtering
- Transparent recommendation reasoning

## Future Enhancements

- Integration with Spotify API for real-time data
- User-specific weight customization
- Mood-based playlist generation
- Streaming history incorporation
- Dynamic feature weighting

## Acknowledgments

- Spotify for the song attributes dataset
- Kaggle community for dataset availability

## Contact

For questions or collaboration opportunities, please open an issue or contact istatiehahmad0@gmail.com.

---

*Note: This system is for educational and research purposes. For commercial use, please ensure compliance with Spotify's terms of service and data usage policies.*
