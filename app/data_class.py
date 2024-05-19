import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from joblib import load
import asyncio


model = load('data/model/logistic_regression_model.joblib')
clf = load('data/model/decision_tree_model.joblib')
neighbors = load('data/model/knn_model.joblib')
le_artist = load('data/model/label_encoder_artist.joblib')
scaler = load('data/model/scaler.joblib')
imputer = load('data/model/imputer.joblib')

async def mus_class(playlist):
    
    df = pd.DataFrame.from_records(playlist)

    # Преобразование данных
    le = LabelEncoder()
    df['genre'] = df['genre'].fillna('unknown')
    df['genre'] = le.fit_transform(df['genre'])
    df['date'] = pd.to_datetime(df['date']).dt.year

    word2vec_model = Word2Vec(sentences=df['tags'].fillna('').apply(str).apply(str.split), vector_size=10, window=5, min_count=1, workers=4)
    tag_vectors = []
    for tags in df['tags'].fillna(''):
        tags_list = tags.split()
        vectors = [word2vec_model.wv[tag] for tag in tags_list if tag in word2vec_model.wv]
        if vectors:
            tag_vectors.append(pd.DataFrame(vectors).mean(axis=0))
        else:
            tag_vectors.append(pd.Series([0] * 10))



    X_playlist = pd.DataFrame({
        'genre': df['genre'],
        'date': df['date'],
        'likes': df['likes'],
        'stream': df['stream']
    })


    for i in range(len(tag_vectors[0])):
        X_playlist[f'tag_{i}'] = [tag[i] for tag in tag_vectors]

    # Обработка пропущенных значений и масштабирование
    X_playlist_imputed = imputer.transform(X_playlist)
    X_playlist_scaled = scaler.transform(X_playlist_imputed)

    # Предсказание вероятностей
    probabilities_lr = model.predict_proba(X_playlist_scaled)
    probabilities_tree = clf.predict_proba(X_playlist)
    probabilities_knn = neighbors.predict_proba(X_playlist)

    # Получение средних значений вероятностей для каждого артиста
    mean_probabilities_lr = probabilities_lr.mean(axis=0)
    mean_probabilities_dt = probabilities_tree.mean(axis=0)
    mean_probabilities_knn = probabilities_knn.mean(axis=0)

    # Выбор артиста с наивысшим средним значением вероятности
    best_artist_lr = le_artist.inverse_transform([np.argmax(mean_probabilities_lr)])[0]
    best_artist_dt = le_artist.inverse_transform([np.argmax(mean_probabilities_dt)])[0]
    best_artist_knn = le_artist.inverse_transform([np.argmax(mean_probabilities_knn)])[0]


    result = ("Предсказанные авторы методами: \n"
            f"Логистической регрессией: {best_artist_lr} - с вероятностью: {mean_probabilities_lr[np.argmax(mean_probabilities_lr)]:.2f}\n" +
            f"Деревом принятия решений: {best_artist_dt} - с вероятностью: {mean_probabilities_dt[np.argmax(mean_probabilities_dt)]:.2f}\n" +
            f"k-ближайших соседей: {best_artist_knn} - с вероятностью: {mean_probabilities_knn[np.argmax(mean_probabilities_knn)]:.2f}")

    
    return result

if __name__ == "__main__":
    asyncio.run(mus_class(None))
