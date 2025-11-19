# run.py
import joblib


model = joblib.load('model_multi.pkl')
print("Модель загружена")



while True:
    text = input("➤ ").strip()
    
    if text.lower() in ('exit', 'quit', 'выход'):
        
        break
    
    if not text:
        print("Надо чтот о написать")
        continue

    try:
        pred = model.predict([text])[0]
        category_pred, style_pred = pred[0], pred[1]


        probas = model.predict_proba([text])
        prob_cat = dict(zip(model.named_steps['clf'].estimators_[0].classes_, probas[0][0]))
        prob_style = dict(zip(model.named_steps['clf'].estimators_[1].classes_, probas[1][0]))
        conf_cat = max(prob_cat.values())
        conf_style = max(prob_style.values())

        print(f"Категория:   {category_pred} уверенность: {conf_cat:.0%}")
        print(f"Стиль: {style_pred} уверенность: {conf_style:.0%}")
        



    except Exception as e:
        print(f"Ошибка: {e}")