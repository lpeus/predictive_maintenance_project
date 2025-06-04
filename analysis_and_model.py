import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,
                           confusion_matrix,
                           classification_report,
                           roc_auc_score,
                           roc_curve)
from ucimlrepo import fetch_ucirepo

def clean_column_name(col):
    """Приводит названия столбцов к стандартному виду для сравнения"""
    return str(col).lower().replace(' ', '').replace('[', '').replace(']', '').replace('_', '')

def find_matching_column(target_col, available_cols):
    """Находит наиболее подходящее название столбца"""
    target_clean = clean_column_name(target_col)
    for col in available_cols:
        if target_clean == clean_column_name(col):
            return col
    return None
    #старт
def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    if st.button("Загрузить стандартный датасет"):
        try:
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            st.session_state.data = data
            st.success("Данные успешно загружены!")
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {str(e)}")

    uploaded_file = st.file_uploader("Или загрузите CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success("CSV файл успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка чтения CSV: {str(e)}")

    if 'data' not in st.session_state:
        return

    data = st.session_state.data.copy()

    try:
        # Проверяем и переименовываем столбцы
        column_mapping = {}
        
        # Создаем гибкие варианты названий
        for col in data.columns:
            col_lower = str(col).lower().replace(' ', '')
            if 'type' in col_lower:
                column_mapping[col] = 'Type'
            elif 'air' in col_lower and 'temp' in col_lower:
                column_mapping[col] = 'Air_temperature'
            elif 'process' in col_lower and 'temp' in col_lower:
                column_mapping[col] = 'Process_temperature'
            elif 'rotation' in col_lower or 'speed' in col_lower:
                column_mapping[col] = 'Rotational_speed'
            elif 'torque' in col_lower:
                column_mapping[col] = 'Torque'
            elif 'tool' in col_lower and 'wear' in col_lower:
                column_mapping[col] = 'Tool_wear'
            elif 'machine' in col_lower and 'fail' in col_lower:
                column_mapping[col] = 'Machine_failure'
        
        # Применяем переименование
        data = data.rename(columns=column_mapping)
        
        # Проверяем наличие обязательных столбцов
        required_columns = ['Type', 'Air_temperature', 'Process_temperature',
                          'Rotational_speed', 'Torque', 'Tool_wear', 'Machine_failure']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_columns}")

        # Удаляем лишние столбцы
        data = data[required_columns]
        
        # Преобразование категориальных переменных
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
        
        # Масштабирование числовых признаков
        numeric_cols = ['Air_temperature', 'Process_temperature',
                       'Rotational_speed', 'Torque', 'Tool_wear']
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        # Разделение данных
        X = data.drop(columns=['Machine_failure'])
        y = data['Machine_failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        st.error(f"Ошибка предобработки данных: {str(e)}")
        st.write("Обнаруженные столбцы:", list(st.session_state.data.columns))
        return

    # ... (остальной код обучения моделей и интерфейса остается без изменений)
    # Обучение моделей
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            results[name] = auc
            
            if auc > best_auc:
                best_auc = auc
                best_model = model
        except Exception as e:
            st.warning(f"Ошибка в модели {name}: {str(e)}")
    
    # Визуализация результатов
    st.subheader("Сравнение моделей")
    if results:
        st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['ROC-AUC']))
    
    if best_model is not None:
        st.subheader(f"Лучшая модель: {best_model.__class__.__name__}")
        y_pred = best_model.predict(X_test)
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        
        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred),
                   annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    
    # Интерфейс для предсказаний
    st.header("Предсказание отказа оборудования")
    with st.form("prediction_form"):
        st.write("Введите параметры оборудования:")
        
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.number_input("Температура воздуха [K]", value=300.0)
            process_temp = st.number_input("Температура процесса [K]", value=310.0)
            rot_speed = st.number_input("Скорость вращения [rpm]", value=1500)
        
        with col2:
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", value=100)
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
        
        submitted = st.form_submit_button("Предсказать")

        if submitted and best_model is not None:
            try:
                # Подготовка входных данных
                input_data = pd.DataFrame({
                    'Type': [product_type],
                    'Air_temperature': [air_temp],
                    'Process_temperature': [process_temp],
                    'Rotational_speed': [rot_speed],
                    'Torque': [torque],
                    'Tool_wear': [tool_wear]
                })

                # Преобразование категориальных признаков
                input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])

                # Масштабирование
                input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

                # Предсказание
                prediction = best_model.predict(input_data)
                prediction_proba = best_model.predict_proba(input_data)[0][1]

                # Вывод результатов
                st.subheader("Результат предсказания")
                if prediction[0] == 1:
                    st.error(f"Вероятность отказа: {prediction_proba:.2%} ❌")
                    st.write("Рекомендуется провести техническое обслуживание!")
                else:
                    st.success(f"Вероятность отказа: {prediction_proba:.2%} ✅")
                    st.write("Статус: Нормальная работа")

            except Exception as e:
                st.error(f"Ошибка предсказания: {str(e)}")
                st.write("Проверьте введенные данные")
if __name__ == "__main__":
    analysis_and_model_page()
