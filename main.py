import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


df = pd.read_csv('dataset.csv')


print(f"Датасет: {len(df)} строк")
print("Пример:")
print(df.head(3))


X = df['text']
y = df[['category', 'style']]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['category']
)


model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 2),
        stop_words=None
    )),
    ('clf', MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            max_iter=500,
            random_state=42
        ),
        n_jobs=-1
    ))
])


print("Обучение")
model.fit(X_train, y_train)


print("оценка на тестовой выборке:")
y_pred = model.predict(X_test)


for i, target in enumerate(['category', 'style']):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"→ {target}: accuracy = {acc:.2%}")


print("\n" + "="*50)
print("ОТЧЁТ ПО КАТЕГОРИИ:")
print(classification_report(y_test['category'], y_pred[:, 0], zero_division=0))
print("\nОТЧЁТ ПО СТИЛЮ:")
print(classification_report(y_test['style'], y_pred[:, 1], zero_division=0))





joblib.dump(model, 'model_multi.pkl')
print("Модель готова'")