import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Загрузка моделей
genmodel = load_model('/Users/diassss/Desktop/amlfinal/gender_model.keras')  # Модель для пола
agemodel = load_model('/Users/diassss/Desktop/amlfinal/age_model.keras')    # Модель для возраста

# Заголовок приложения
st.title("Определение пола и возраста")
st.write("Загрузите изображение лица, чтобы узнать пол и возраст.")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем изображение
    img = Image.open(uploaded_file).resize((200, 200))
    st.image(img, caption="Загруженное изображение", use_column_width=True)

    # Преобразуем изображение в массив NumPy
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Предсказания
    gender_prediction = (genmodel.predict(img_array) > 0.5).astype(int)[0][0]
    age_prediction = agemodel.predict(img_array)[0][0]

    # Вывод результата
    gender = "Мужской" if gender_prediction == 0 else "Женский"
    st.write(f"**Пол:** {gender}")
    st.write(f"**Возраст:** {age_prediction:.1f} лет")
