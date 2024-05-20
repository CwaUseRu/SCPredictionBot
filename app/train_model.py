import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from gensim.models import Word2Vec
from joblib import dump


df = pd.read_csv('data/scdb.csv')


artists = ['Billie Eilish', 'Eminem', 'Skrillex', 'КИНО', 'Егор Крид']



df_filtered = df[df['artist'].isin(artists)].copy()
le_artist = LabelEncoder()
df_filtered['artist'] = le_artist.fit_transform(df_filtered['artist'])
le_genre = LabelEncoder()
df_filtered['genre'] = le_genre.fit_transform(df_filtered['genre'])
df_filtered['date'] = pd.to_datetime(df_filtered['date']).dt.year

word2vec_model = Word2Vec(sentences=df_filtered['tags'].fillna('').apply(str).apply(str.split), vector_size=10, window=5, min_count=1, workers=4)
tag_vectors = []
for tags in df_filtered['tags'].fillna(''):
    tags_list = tags.split()
    vectors = [word2vec_model.wv[tag] for tag in tags_list if tag in word2vec_model.wv]
    if vectors:
        tag_vectors.append(pd.DataFrame(vectors).mean(axis=0))
    else:
        tag_vectors.append(pd.Series([0] * 10))


X = pd.DataFrame({
    'genre': df_filtered['genre'],
    'date': df_filtered['date'],
    'likes': df_filtered['likes'],
    'stream': df_filtered['stream']
})


for i in range(len(tag_vectors[0])):
    X[f'tag_{i}'] = [tag[i] for tag in tag_vectors]

print(X.head())
X.columns = X.columns.astype(str)
y = df_filtered['artist']

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# Обучение модели логистической регрессии
model = LogisticRegression(solver='sag', random_state=0, max_iter=10000)
model.fit(X_scaled, y)
dump(model, 'data/model/logistic_regression_model.joblib')

# Обучение модели дерева решений
clf = DecisionTreeClassifier()
clf.fit(X, y)
dump(clf, 'data/model/decision_tree_model.joblib')

# Обучение модели K-ближайших соседей
neighbors = KNeighborsClassifier(n_neighbors=5)
neighbors.fit(X, y)
dump(neighbors, 'data/model/knn_model.joblib')

# Сохранение label encoder для artist
dump(le_artist, 'data/model/label_encoder_artist.joblib')
dump(scaler, 'data/model/scaler.joblib')
dump(imputer, 'data/model/imputer.joblib')

# Оценка модели логистической регрессии
lr_scores = cross_val_score(model, X_scaled, y, cv=5)
print("Средняя точность модели логистической регрессии:", lr_scores.mean())

# Оценка модели дерева решений
dt_scores = cross_val_score(clf, X, y, cv=5)
print("Средняя точность модели дерева решений:", dt_scores.mean())

# Оценка модели K-ближайших соседей
knn_scores = cross_val_score(neighbors, X, y, cv=5)
print("Средняя точность модели K-ближайших соседей:", knn_scores.mean())