import asyncio, json, io
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



async def mus_analys(input):
    input = eval(input)
    data = pd.DataFrame(input)

    df = pd.DataFrame(data)

    # Разделение данных на обучающий и тестовый наборы
    X = df[['duration_ms']]  # Признаки (продолжительность песни)
    y = df['popularity']      # Целевая переменная (популярность)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказание на тестовом наборе
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Создание и сохранение графика в буфер памяти
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Duration (ms)')
    plt.ylabel('Popularity')
    plt.title('Linear Regression')

    plt.savefig('data/pics/plot1.png')

    return

    # Очистка графика из памяти после использования
    plt.close()