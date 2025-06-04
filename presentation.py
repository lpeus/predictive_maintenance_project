import streamlit as st

def presentation_page():
    st.title("Презентация проекта")
    
    # Создаем слайды с помощью st.markdown
    slides = [
        """
        <h1>Прогнозирование отказов оборудования</h1>
        <hr>
        <h2>Введение</h2>
        <ul>
            <li>Задача: Предсказание отказов промышленного оборудования</li>
            <li>Датасет: AI4I 2020 Predictive Maintenance</li>
            <li>Целевая переменная: Machine failure (0/1)</li>
        </ul>
        """,
        
        """
        <h2>Этапы работы</h2>
        <ol>
            <li>Загрузка и предобработка данных</li>
            <li>Обучение моделей классификации</li>
            <li>Оценка метрик (Accuracy, ROC-AUC)</li>
            <li>Разработка Streamlit-приложения</li>
        </ol>
        """,
        
        """
        <h2>Результаты</h2>
        <ul>
            <li>Лучшая модель: Random Forest (AUC = 0.98)</li>
            <li>Точность предсказаний: >95%</li>
            <li>Веб-интерфейс для прогнозирования</li>
        </ul>
        """
    ]
    
    # Навигация по слайдам
    if 'slide' not in st.session_state:
        st.session_state.slide = 0
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Предыдущий") and st.session_state.slide > 0:
            st.session_state.slide -= 1
    with col2:
        if st.button("Следующий →") and st.session_state.slide < len(slides) - 1:
            st.session_state.slide += 1
    
    # Отображение текущего слайда
    st.markdown(slides[st.session_state.slide], unsafe_allow_html=True)
    
    # Индикатор прогресса
    st.progress((st.session_state.slide + 1) / len(slides))
    st.caption(f"Слайд {st.session_state.slide + 1} из {len(slides)}")
