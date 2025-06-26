from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load and prepare data
df2 = pd.read_csv('tmdb_5000_movies.csv')
df1 = pd.read_csv('tmdb_5000_credits.csv')
df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

# Fill NaNs in overview
df2['overview'] = df2['overview'].fillna('')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create reverse mapping of titles to indices
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return []  # Return empty if movie not found
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations (skip the first because it's the movie itself)
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices].tolist()


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    query = ''
    if request.method == 'POST':
        query = request.form.get('movie_title')
        recommendations = get_recommendations(query)
    all_movies = df2['title'].tolist()
    return render_template('index.html', movies=all_movies, recommendations=recommendations, query=query)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

