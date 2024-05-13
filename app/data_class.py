import asyncio
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data\scdb.csv')

le = LabelEncoder()

async def mus_class(input):

    data = pd.DataFrame.from_records(input)


    X = df.iloc[:, 8:16]
    y = df['artist']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    model = LogisticRegression(solver='sag', random_state=0)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    s = clf.predict(X_train)
    score = np.array([s, y_train])
    score = score.T
    print(pd.DataFrame(score).head(10))

    clf.score(X_train, y_train)

    print(X_train)

    neighbors = KNeighborsClassifier(n_neighbors=1)
    neighbors.fit(X_train, y_train)

    print(neighbors.score(X_train, y_train))

    
    data.fit(X_train, y_train)

if __name__ == "__main__":
    asyncio.run(mus_class('https://soundcloud.com/user-818400639-218240550/sets/zzgcthqyvuhe?si=bfbda8427ad8474ebad3fad2379462b1&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing'))