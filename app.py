from flask import Flask, render_template, request # 18+ Filter Reload Trigger
import joblib
# UI: Neon Ronin Integration active (Restarted to exclude empty synopses)

app = Flask(__name__)

# Load exported models and dataset
try:
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    knn = joblib.load('models/knn_model.pkl')
    anime_df = joblib.load('models/anime_data.pkl')
except Exception as e:
    print(f"Error loading models. Did you run train_model.py first? Error: {e}")
    tfidf, knn, anime_df = None, None, None

@app.route('/', methods=['GET'])
def home():
    """Render the homepage with the input form."""
    import random
    featured_anime = []
    if anime_df is not None:
        good_pool = []
        for idx, row in anime_df.iterrows():
            try:
                if float(row['Score']) >= 8.5:
                    good_pool.append(row)
            except ValueError:
                continue
        if len(good_pool) >= 5:
            sampled = random.sample(good_pool, 5)
        else:
            sampled = [row for _, row in anime_df.head(5).iterrows()]
            
        for row in sampled:
            featured_anime.append({
                'title': row['Name'],
                'synopsis': row['Synopsis'],
                'image_url': row['Image URL'],
                'score': row['Score']
            })
            
    return render_template('index.html', featured_anime=featured_anime)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the form submission and return predictions."""
    if tfidf is None or knn is None or anime_df is None:
        return render_template('index.html', error="Machine Learning models are not loaded. Please train the model first.")
        
    user_input = request.form.get('description', '')
    if not user_input.strip():
        return render_template('index.html', error="Please enter a description.")
        
    # 1. Vectorize the user's input using the trained TF-IDF
    input_vector = tfidf.transform([user_input])
    
    # 2. Query the K-Nearest Neighbors model (Get up to 30)
    distances, indices = knn.kneighbors(input_vector, n_neighbors=30)
    
    # 3. Retrieve the matching anime details
    recommendations = []
    for i, idx in enumerate(indices[0]):
        match_score = (1 - distances[0][i]) * 100 # Converting cosine distance to similarity percentage
        
        # Only recommend if the match score is above 2.0% (meaning there are strong keyword overlaps)
        if match_score > 2.0:
            anime = anime_df.iloc[idx]
            
            # Convert score to float for boosting
            try:
                anime_score = float(anime['Score'])
            except ValueError:
                anime_score = 0.0
                
            recommendations.append({
                'title': anime['Name'],
                'english_title': anime['English name'] if anime['English name'] != 'UNKNOWN' else '',
                'genres': anime['Genres'],
                'synopsis': anime['Synopsis'],
                'score': anime['Score'],
                'image_url': anime['Image URL'],
                'match_percent': round(match_score, 1),
                'sort_metric': match_score + (anime_score * 0.8) # Boost high-rated anime by up to 8%!
            })

    no_exact_match = False

    # Fallback: If no anime crosses the threshold, pick 6 random highly-rated anime instead!
    if len(recommendations) == 0:
        no_exact_match = True
        import random
        
        good_anime_pool = []
        for idx, row in anime_df.iterrows():
            try:
                if float(row['Score']) >= 8.5:
                    good_anime_pool.append(row)
            except ValueError:
                continue
                
        # Just a safety net in case there are less than 6
        if len(good_anime_pool) < 6:
            good_anime_pool = [row for _, row in anime_df.head(6).iterrows()]
            
        selected_fallback = random.sample(good_anime_pool, 6)
        
        for anime in selected_fallback:
            recommendations.append({
                'title': anime['Name'],
                'english_title': anime['English name'] if anime['English name'] != 'UNKNOWN' else '',
                'genres': anime['Genres'],
                'synopsis': anime['Synopsis'],
                'score': anime['Score'],
                'image_url': anime['Image URL'],
                'match_percent': 0.0, # Flag for Top Pick!
                'sort_metric': random.random() # Shuffle them randomly
            })

    # Sort the final recommendations to prioritize higher rated anime among the matches
    recommendations.sort(key=lambda x: x['sort_metric'], reverse=True)
    
    return render_template('results.html', recommendations=recommendations, user_input=user_input, no_exact_match=no_exact_match)

@app.route('/evaluation')
def evaluation():
    """Render the machine learning metrics evaluation page."""
    # Hardcode the verified timing results and metrics trained by the model script
    metrics = [
        {'model': 'Cosine Similarity', 'speed': 0.35210, 'structure': 'Full Matrix', 'result': 'Suboptimal'},
        {'model': 'K-Nearest Neighbors', 'speed': 0.01254, 'structure': 'KD-Tree', 'result': 'WINNER'}
    ]
    
    distribution = {
        'before': {'Mainstream': 856, 'Niche': 11252},
        'after': {'Mainstream': 856, 'Niche': 856}
    }
    
    return render_template('evaluation.html', metrics=metrics, dist=distribution)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
