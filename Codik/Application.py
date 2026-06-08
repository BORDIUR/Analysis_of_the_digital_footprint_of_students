"""
Приложение для анализа цифрового образовательного следа студентов
Дисциплина: Численные методы
Автор: Боков В.Е.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.express as px
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist
from sklearn.preprocessing import OneHotEncoder
import base64
import warnings
warnings.filterwarnings('ignore')
import os

# Определяем путь к папке, где находится текущий скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "Княгининский университет (логотип PNG)-13.png")

# Затем везде, где использовался логотип, замените имя файла на переменную logo_path
# Например, при загрузке изображения:
if os.path.exists(logo_path):
    st.image(logo_path, width=80)
else:
    st.markdown("**НГИЭУ**")

# Функция для преобразования изображения в base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Настройка страницы
st.set_page_config(
    page_title="Анализ успеваемости студентов",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Бело-бордовые стили
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .stApp header { background-color: #8B0000; }
    .main-header { color: #8B0000; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    h1, h2, h3 { color: #8B0000; }
    .stButton > button { background-color: #8B0000; color: white; border: none; border-radius: 5px; padding: 8px 20px; font-weight: bold; transition: all 0.3s; }
    .stButton > button:hover { background-color: #6B0000; color: white; }
    .info-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #8B0000; margin-bottom: 15px; }
    .badge-cluster-0 { background-color: #8B0000; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .badge-cluster-1 { background-color: #28a745; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .badge-cluster-2 { background-color: #ffc107; color: #333; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .badge-cluster-3 { background-color: #dc3545; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .streamlit-expanderHeader { background-color: #f8f9fa; color: #8B0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Верхняя панель с логотипом
logo_base64 = get_image_base64("Княгининский университет (логотип PNG)-13.png")
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="80">', unsafe_allow_html=True)
    else:
        st.markdown("**НГИЭУ**")
with col_title:
    st.markdown("<div class='main-header'>Анализ успеваемости студентов</div>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# ФУНКЦИИ
# =====================================================

def format_minutes_to_time(minutes):
    if pd.isna(minutes) or minutes <= 0:
        return "-"
    hours = minutes // 60
    mins = minutes % 60
    if hours > 0 and mins > 0:
        return f"{int(hours)} ч {int(mins)} мин"
    elif hours > 0:
        return f"{int(hours)} ч"
    else:
        return f"{int(mins)} мин"

def extract_practice_number(filename):
    patterns = [
        r'П\.З\._(\d+)', r'ПЗ[_\s\-]?(\d+)', r'практика[_\s\-]?(\d+)',
        r'practice[_\s\-]?(\d+)', r'(\d+)\s*практика', r'(\d+)\.xlsx?',
        r'[_-](\d+)[_-]', r'(\d+)$'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def parse_russian_date(date_str):
    if pd.isna(date_str) or date_str == 0:
        return None
    months_ru = {
        'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
        'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
        'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
    }
    try:
        parts = str(date_str).strip().split()
        if len(parts) >= 4:
            day = parts[0].zfill(2)
            month_ru = parts[1]
            year = parts[2]
            month = months_ru.get(month_ru, '01')
            return datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
    except:
        return None
    return None

def time_to_minutes(time_str):
    if pd.isna(time_str):
        return None
    time_str = str(time_str).strip().lower()
    total_minutes = 0
    hours_match = re.search(r'(\d+)\s*ч', time_str)
    if hours_match:
        total_minutes += int(hours_match.group(1)) * 60
    minutes_match = re.search(r'(\d+)\s*мин', time_str)
    if minutes_match:
        total_minutes += int(minutes_match.group(1))
    days_match = re.search(r'(\d+)\s*дн', time_str)
    if days_match:
        total_minutes += int(days_match.group(1)) * 24 * 60
    seconds_match = re.search(r'(\d+)\s*сек', time_str)
    if seconds_match:
        total_minutes += round(int(seconds_match.group(1)) / 60)
    return round(total_minutes)

def clean_practice_file(df, practice_num):
    cleaned_data = []
    for idx, row in df.iterrows():
        fio_raw = f"{row.get('Фамилия', '')} {row.get('Имя', '')}".strip()
        fio = re.sub(r'[★*]', '', fio_raw).strip()
        if not fio or fio == 'nan nan':
            continue
        test_start = parse_russian_date(row.get('Тест начат'))
        test_end = parse_russian_date(row.get('Завершено'))
        time_spent = time_to_minutes(row.get('Затраченное время'))
        score_raw = row.get('Оценка/5,00', row.get('Оценка', 0))
        if pd.isna(score_raw) or score_raw == '-' or score_raw == '—' or str(score_raw).strip() == '':
            score = 0
        else:
            try:
                score_str = str(score_raw).replace(',', '.').strip()
                score = float(score_str)
            except:
                score = 0
        cleaned_data.append({
            'ФИО': fio,
            f'Оценка_{practice_num}': score,
            f'Тест_начат_{practice_num}': test_start,
            f'Завершено_{practice_num}': test_end,
            f'Затраченное_время_{practice_num}': time_spent
        })
    return pd.DataFrame(cleaned_data)

def calculate_survival(student_row, practice_nums):
    survival = []
    for k_idx in range(1, len(practice_nums) + 1):
        all_done = True
        for n in practice_nums[:k_idx]:
            score = student_row.get(f'Оценка_{n}', 0)
            if score <= 0:
                all_done = False
                break
        survival.append(1 if all_done else 0)
    return survival

def grade_category(x):
    if pd.isna(x) or x == 0:
        return 'Н'
    elif x < 2.0:
        return '2'
    elif x < 3.1:
        return '3'
    elif x < 4.1:
        return '4'
    else:
        return '5'

def grade_cat_for_stats(score):
    if pd.isna(score) or score <= 0:
        return None
    elif score < 2.5:
        return 2
    elif score < 3.5:
        return 3
    elif score < 4.5:
        return 4
    else:
        return 5

# =====================================================
# ЗАГРУЗКА ДАННЫХ
# =====================================================

if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'practice_nums' not in st.session_state:
    st.session_state.practice_nums = []

st.markdown("### Загрузка данных")
uploaded_files = st.file_uploader(
    "Выберите файлы практик",
    type=['xlsx', 'xls'],
    accept_multiple_files=True,
    help="Можно выбрать несколько файлов одновременно (Ctrl+клик)"
)

if uploaded_files:
    practice_files = {}
    unknown_files = []
    for file in uploaded_files:
        num = extract_practice_number(file.name)
        if num is not None:
            practice_files[num] = file
        else:
            unknown_files.append(file.name)
    
    if unknown_files:
        st.warning(f"Не удалось определить номер практики для файлов: {unknown_files[:3]}...")
    
    if practice_files:
        st.success(f"Загружено файлов: {len(uploaded_files)}, определено практик: {len(practice_files)}")
        st.info(f"Определены практики: {sorted(practice_files.keys())}")
        
        st.markdown("### Укажите даты открытия практик")
        st.caption("Даты используются для расчёта запаздывания начала выполнения")
        
        open_dates = {}
        cols = st.columns(4)
        for i, n in enumerate(sorted(practice_files.keys())):
            col_idx = i % 4
            with cols[col_idx]:
                default_date = datetime(2025, 9, 1)
                open_dates[n] = st.date_input(f"Практика {n}", value=default_date, key=f"open_{n}")
                open_dates[n] = open_dates[n].strftime("%Y-%m-%d")
        
        if st.button("Выполнить анализ"):
            with st.spinner("Обработка данных..."):
                all_data = {}
                for n, file in practice_files.items():
                    df = pd.read_excel(file)
                    cleaned = clean_practice_file(df, n)
                    all_data[n] = cleaned
                
                students = {}
                for n, df in all_data.items():
                    for _, row in df.iterrows():
                        fio = row['ФИО']
                        if fio not in students:
                            students[fio] = {}
                        students[fio][f'Оценка_{n}'] = row.get(f'Оценка_{n}', 0)
                        students[fio][f'Затраченное_время_{n}'] = row.get(f'Затраченное_время_{n}', 0)
                        students[fio][f'Тест_начат_{n}'] = row.get(f'Тест_начат_{n}')
                
                final_df = pd.DataFrame([{
                    'ID': i + 1,
                    'ФИО': fio,
                    **{f'Оценка_{n}': data.get(f'Оценка_{n}', 0) for n in practice_files.keys()},
                    **{f'Затраченное_время_{n}': data.get(f'Затраченное_время_{n}', 0) for n in practice_files.keys()},
                    **{f'Тест_начат_{n}': data.get(f'Тест_начат_{n}') for n in practice_files.keys()},
                } for i, (fio, data) in enumerate(students.items())])
                
                # Расчёт запаздываний
                for n in practice_files.keys():
                    if n in open_dates:
                        open_date = datetime.strptime(open_dates[n], "%Y-%m-%d")
                        start_col = f'Тест_начат_{n}'
                        if start_col in final_df.columns:
                            final_df[f'Запаздывание_{n}'] = final_df[start_col].apply(
                                lambda x: (x - open_date).days if pd.notna(x) else None
                            )
                        else:
                            final_df[f'Запаздывание_{n}'] = 0
                
                final_df = final_df.fillna(0)
                st.session_state.df_processed = final_df
                st.session_state.practice_nums = list(practice_files.keys())
                st.session_state.open_dates = open_dates
                st.success("Данные успешно обработаны!")
                with st.expander("Предпросмотр обработанных данных"):
                    st.dataframe(final_df.head(10), use_container_width=True)

# =====================================================
# АНАЛИЗ ДАННЫХ
# =====================================================

if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    practice_nums = st.session_state.practice_nums
    
    st.info(f"Анализируются практики: {practice_nums}")
    
    # Кластеризация
    sequences = []
    for idx, row in df.iterrows():
        seq = ''
        for n in practice_nums:
            score = row.get(f'Оценка_{n}', 0)
            seq += grade_category(score)
        sequences.append(seq)
    
    all_symbols = ['2', '3', '4', '5', 'Н']
    encoder = OneHotEncoder(categories=[all_symbols], sparse_output=False)
    encoded_vectors = []
    for seq in sequences:
        symbols = list(seq)
        symbols_array = np.array(symbols).reshape(-1, 1)
        encoded = encoder.fit_transform(symbols_array)
        encoded_vectors.append(encoded.flatten())
    X = np.array(encoded_vectors)
    
    if len(X) >= 4:
        distance_matrix = pdist(X, metric='euclidean')
        linkage_matrix = linkage(distance_matrix, method='ward')
        clusters = cut_tree(linkage_matrix, n_clusters=min(4, len(X))).flatten()
    else:
        clusters = [0] * len(X)
    
    # Расчёт показателей студентов
    student_stats = []
    for idx, row in df.iterrows():
        student_id = row['ID']
        total_score = 0
        count_done = 0
        for n in practice_nums:
            score = row.get(f'Оценка_{n}', 0)
            if score > 0:
                total_score += score
                count_done += 1
        max_possible = len(practice_nums) * 5
        percent_of_max = total_score / max_possible * 100 if max_possible > 0 else 0
        seq = ''.join([grade_category(row.get(f'Оценка_{n}', 0)) for n in practice_nums])
        etalon = '5' * len(practice_nums)
        hamming = sum(1 for a, b in zip(seq, etalon) if a != b)
        survival = calculate_survival(row, practice_nums)
        student_stats.append({
            'ID': student_id, 'ФИО': row['ФИО'], 'Сумма баллов': round(total_score, 2),
            'Процент от максимума': round(percent_of_max, 2), 'Выполнено практик': count_done,
            'Всего практик': len(practice_nums), 'Кластер': clusters[idx] if len(clusters) > idx else 0,
            'Расстояние_Хэмминга': hamming, 'Выживаемость': survival
        })
    stats_df = pd.DataFrame(student_stats)
    
    if len(stats_df) > 0:
        risk_threshold = stats_df['Процент от максимума'].quantile(0.33)
        stats_df['Группа риска'] = stats_df['Процент от максимума'] <= risk_threshold
    else:
        stats_df['Группа риска'] = False
    stats_df = stats_df.sort_values(['Группа риска', 'Процент от максимума'], ascending=[False, False])
    
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего студентов", len(stats_df))
    with col2:
        st.metric("Средний балл", f"{stats_df['Процент от максимума'].mean():.1f}%" if len(stats_df) > 0 else "0%")
    with col3:
        st.metric("В группе риска", stats_df['Группа риска'].sum())
    with col4:
        st.metric("Успеваемость (медиана)", f"{stats_df['Процент от максимума'].median():.1f}%" if len(stats_df) > 0 else "0%")
    
    st.markdown("---")
    
    # ========== ОСНОВНЫЕ ВКЛАДКИ ==========
    tab_students, tab_practices, tab_correlations, tab_stats = st.tabs([
        "Студенты", "Анализ практик", "Корреляции", "Статистика"
    ])
    
    # ========== ВКЛАДКА 1: СТУДЕНТЫ ==========
    with tab_students:
        st.markdown("<h3 style='color:#8B0000;'>Все студенты</h3>", unsafe_allow_html=True)
        st.markdown("<p>Нажмите на ID студента, чтобы увидеть подробную информацию</p>", unsafe_allow_html=True)
        
        if 'selected_student' not in st.session_state:
            st.session_state.selected_student = None
        
        for _, student in stats_df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1.5, 1.5, 1.5, 1.5])
            with col1:
                if st.button(f"ID {int(student['ID'])}", key=f"btn_{student['ID']}"):
                    st.session_state.selected_student = student['ID']
            with col2:
                st.write(f"{student['ФИО']}")
            with col3:
                st.write(f"Успеваемость: {student['Процент от максимума']:.1f}%")
            with col4:
                st.write(f"Выполнено: {student['Выполнено практик']} из {student['Всего практик']}")
            with col5:
                cluster_id = int(student['Кластер'])
                if cluster_id == 0:
                    st.markdown("<span class='badge-cluster-0'>Кластер 0</span>", unsafe_allow_html=True)
                elif cluster_id == 1:
                    st.markdown("<span class='badge-cluster-1'>Кластер 1</span>", unsafe_allow_html=True)
                elif cluster_id == 2:
                    st.markdown("<span class='badge-cluster-2'>Кластер 2</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-cluster-3'>Кластер 3</span>", unsafe_allow_html=True)
            with col6:
                if student['Группа риска']:
                    st.markdown("<span class='badge-cluster-3'>Группа риска</span>", unsafe_allow_html=True)
                else:
                    if student['Процент от максимума'] >= 70:
                        st.markdown("<span class='badge-cluster-0'>Высокая успеваемость</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-cluster-2'>Средняя успеваемость</span>", unsafe_allow_html=True)
        
        # Детальная информация о студенте
        if st.session_state.selected_student is not None:
            st.markdown("---")
            st.markdown("<h3 style='color:#8B0000;'>Детальная информация о студенте</h3>", unsafe_allow_html=True)
            
            student_data = stats_df[stats_df['ID'] == st.session_state.selected_student].iloc[0]
            student_row = df[df['ID'] == st.session_state.selected_student].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**ID студента:** {int(student_data['ID'])}")
            with col2:
                st.info(f"**ФИО:** {student_data['ФИО']}")
            with col3:
                st.info(f"**Общая успеваемость:** {student_data['Процент от максимума']:.1f}%")
            
            tab_res, tab_surv = st.tabs(["Результаты по практикам", "Расстояние до эталона и выживаемость"])
            
            with tab_res:
                practice_results = []
                for n in practice_nums:
                    score = student_row.get(f'Оценка_{n}', 0)
                    time_minutes = student_row.get(f'Затраченное_время_{n}', 0)
                    if score > 0:
                        оценка = f"{score} (хорошо)" if score >= 3.5 else f"{score} (низкий балл)"
                        статус = "выполнено"
                    else:
                        оценка = "не выполнялась"
                        статус = "не выполнено"
                    practice_results.append({
                        'Практика': n, 'Результат': оценка,
                        'Затраченное время': format_minutes_to_time(time_minutes), 'Статус': статус
                    })
                practice_df = pd.DataFrame(practice_results)
                def highlight_status(val):
                    return 'background-color: #ffcccc' if val == 'не выполнено' else ''
                st.dataframe(practice_df.style.applymap(highlight_status, subset=['Статус']), 
                            use_container_width=True, hide_index=True)
                
                failed = [n for n in practice_nums if student_row.get(f'Оценка_{n}', 0) <= 0]
                low = [n for n in practice_nums if 0 < student_row.get(f'Оценка_{n}', 0) < 3.5]
                if failed:
                    st.warning(f"Не приступал к {len(failed)} практикам: {failed}")
                if low:
                    st.warning(f"Низкие оценки по практикам: {low}")
            
            with tab_surv:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Расстояние Хэмминга")
                    etalon_str = '5' * len(practice_nums)
                    student_seq = ''.join([grade_category(student_row.get(f'Оценка_{n}', 0)) for n in practice_nums])
                    st.metric("Эталон", etalon_str)
                    st.metric("Студент", student_seq)
                    st.metric("Расстояние", student_data['Расстояние_Хэмминга'])
                with col2:
                    st.subheader("Выживаемость")
                    survival_data = student_data['Выживаемость']
                    survival_df = pd.DataFrame({
                        'Практика': [f"ПЗ {n}" for n in practice_nums],
                        'Последовательно': ['Да' if x == 1 else 'Нет' for x in survival_data]
                    })
                    st.dataframe(survival_df, use_container_width=True, hide_index=True)
                    survival_rate = sum(survival_data) / len(survival_data) * 100
                    st.metric("Доля последовательно выполненных", f"{survival_rate:.0f}%")
            
            st.markdown("<h4>Рекомендации</h4>", unsafe_allow_html=True)
            if student_data['Процент от максимума'] < 30:
                st.error("Срочное вмешательство: индивидуальные консультации")
            elif student_data['Процент от максимума'] < 50:
                st.warning("Рекомендуется дополнительная поддержка")
            else:
                st.success("Поддерживать текущий темп")
            
            if st.button("Закрыть"):
                st.session_state.selected_student = None
                st.rerun()
    
    # ========== ВКЛАДКА 2: АНАЛИЗ ПРАКТИК ==========
    with tab_practices:
        st.markdown("<h3 style='color:#8B0000;'>Анализ практик</h3>", unsafe_allow_html=True)
        
        # Статистика по каждой практике
        practice_stats = []
        for n in practice_nums:
            scores = df[f'Оценка_{n}'].values
            times = df[f'Затраченное_время_{n}'].values
            delays = df.get(f'Запаздывание_{n}', pd.Series([0]*len(df))).values
            
            completed_mask = scores > 0
            completed_count = completed_mask.sum()
            completion_rate = completed_count / len(df) * 100 if len(df) > 0 else 0
            
            if completed_count > 0:
                grades = [grade_cat_for_stats(s) for s in scores[completed_mask]]
                pct_2 = grades.count(2) / completed_count * 100
                pct_3 = grades.count(3) / completed_count * 100
                pct_4 = grades.count(4) / completed_count * 100
                pct_5 = grades.count(5) / completed_count * 100
                median_score = np.median(scores[completed_mask])
                
                times_completed = times[completed_mask]
                times_positive = times_completed[times_completed > 0]
                median_time = np.median(times_positive) if len(times_positive) > 0 else 0
                
                delays_completed = delays[completed_mask]
                delays_positive = delays_completed[delays_completed > 0]
                median_delay = np.median(delays_positive) if len(delays_positive) > 0 else 0
                
                no_delay_count = (delays_completed <= 0).sum() if len(delays_completed) > 0 else 0
                pct_no_delay = no_delay_count / completed_count * 100
            else:
                pct_2 = pct_3 = pct_4 = pct_5 = 0
                median_score = 0
                median_time = 0
                median_delay = 0
                pct_no_delay = 0
            
            practice_stats.append({
                'Практика': n,
                'Выполнили, %': round(completion_rate, 1),
                '% оценок 2': round(pct_2, 1),
                '% оценок 3': round(pct_3, 1),
                '% оценок 4': round(pct_4, 1),
                '% оценок 5': round(pct_5, 1),
                'Медианная оценка': round(median_score, 2),
                'Медианное время': format_minutes_to_time(median_time),
                'Медианное опоздание (дни)': round(median_delay, 1) if median_delay > 0 else 0,
                '% без опозданий': round(pct_no_delay, 1)
            })
        
        practice_df = pd.DataFrame(practice_stats)
        st.dataframe(practice_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Доля выполнивших")
            fig = px.bar(practice_df, x='Практика', y='Выполнили, %', 
                         title="Доля студентов, выполнивших практики",
                         color='Выполнили, %', color_continuous_scale='Reds')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Медианная оценка")
            fig2 = px.bar(practice_df, x='Практика', y='Медианная оценка',
                          title="Медианная оценка по практикам",
                          color='Медианная оценка', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Распределение оценок")
        grade_df = practice_df[['Практика', '% оценок 2', '% оценок 3', '% оценок 4', '% оценок 5']].melt(
            id_vars=['Практика'], var_name='Оценка', value_name='Процент'
        )
        fig3 = px.bar(grade_df, x='Практика', y='Процент', color='Оценка',
                      title="Распределение оценок по практикам",
                      color_discrete_map={'% оценок 2': '#dc3545', '% оценок 3': '#ffc107',
                                          '% оценок 4': '#28a745', '% оценок 5': '#8B0000'},
                      barmode='group')
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Узкие места курса")
        problem = practice_df.nsmallest(5, 'Выполнили, %')
        st.dataframe(problem[['Практика', 'Выполнили, %', 'Медианная оценка', '% без опозданий']], 
                     use_container_width=True, hide_index=True)
        
        st.caption("""
        **Пояснение:**
        - Выполнили, % — доля студентов, получивших положительную оценку
        - Оценки 2-5 — распределение по пятибалльной шкале
        - Медианная оценка — типичный результат по практике
        - Медианное время — типичное время выполнения (часы/минуты)
        - % без опозданий — доля выполнивших без задержки (начали в срок или раньше)
        """)
    
    # ========== ВКЛАДКА 3: КОРРЕЛЯЦИИ ==========
    with tab_correlations:
        st.markdown("<h3 style='color:#8B0000;'>Корреляционный анализ</h3>", unsafe_allow_html=True)
        
        # Сбор данных для корреляций
        corr_data = []
        for n in practice_nums:
            scores = df[f'Оценка_{n}'].values
            times = df[f'Затраченное_время_{n}'].values
            delays = df.get(f'Запаздывание_{n}', pd.Series([0]*len(df))).values
            
            mask = scores > 0
            if mask.sum() > 1:
                corr_data.append({
                    'Практика': n,
                    'Оценка-Время': np.corrcoef(scores[mask], times[mask])[0, 1] if len(times[mask]) > 1 else 0,
                    'Оценка-Опоздание': np.corrcoef(scores[mask], delays[mask])[0, 1] if len(delays[mask]) > 1 else 0,
                    'Время-Опоздание': np.corrcoef(times[mask], delays[mask])[0, 1] if len(delays[mask]) > 1 else 0
                })
            else:
                corr_data.append({'Практика': n, 'Оценка-Время': 0, 'Оценка-Опоздание': 0, 'Время-Опоздание': 0})
        
        corr_df = pd.DataFrame(corr_data)
        
        st.subheader("Средние корреляции по всем практикам")
        avg_corr = pd.DataFrame({
            'Пара': ['Оценка-Время', 'Оценка-Опоздание', 'Время-Опоздание'],
            'Корреляция': [
                corr_df['Оценка-Время'].mean(),
                corr_df['Оценка-Опоздание'].mean(),
                corr_df['Время-Опоздание'].mean()
            ]
        })
        fig3 = px.bar(avg_corr, x='Пара', y='Корреляция', 
                      title="Средние корреляции между показателями",
                      color='Корреляция', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Корреляции по отдельным практикам")
        st.dataframe(corr_df.round(3), use_container_width=True, hide_index=True)
        
        st.subheader("Общая корреляционная матрица")
        all_scores = []
        all_times = []
        all_delays = []
        for n in practice_nums:
            mask = df[f'Оценка_{n}'] > 0
            all_scores.extend(df[f'Оценка_{n}'][mask].values)
            all_times.extend(df[f'Затраченное_время_{n}'][mask].values)
            delays = df.get(f'Запаздывание_{n}', pd.Series([0]*len(df)))[mask].values
            all_delays.extend(delays)
        
        if len(all_scores) > 2:
            overall_corr = pd.DataFrame({
                'Оценка': all_scores,
                'Время (мин)': all_times,
                'Опоздание (дни)': all_delays
            }).corr()
            st.dataframe(overall_corr.round(3), use_container_width=True)
            
            st.caption("""
            **Интерпретация корреляций:**
            - Близко к 1 → сильная положительная связь
            - Близко к -1 → сильная отрицательная связь
            - Близко к 0 → связь отсутствует
            """)
    
    # ========== ВКЛАДКА 4: СТАТИСТИКА ==========
    with tab_stats:
        st.markdown("<h3 style='color:#8B0000;'>Типы студентов (кластеры)</h3>", unsafe_allow_html=True)
        
        cluster_info = {
            0: {"название": "Отличники", "цвет": "⭐", "описание": "Высокая успеваемость, выполняют почти все практики."},
            1: {"название": "Хорошая успеваемость", "цвет": "🟢", "описание": "Хорошие результаты, требуется поддержка."},
            2: {"название": "Средняя успеваемость", "цвет": "🟡", "описание": "Средние результаты, нуждаются в мотивации."},
            3: {"название": "Группа риска", "цвет": "🔴", "описание": "Низкая успеваемость, мало выполненных практик."}
        }
        for cluster_id in [0, 1, 2, 3]:
            info = cluster_info[cluster_id]
            st.markdown(f"**{info['цвет']} Кластер {cluster_id} — {info['название']}:** {info['описание']}")
        
        st.markdown("---")
        st.markdown("<h3 style='color:#8B0000;'>Общая статистика по группам студентов</h3>", unsafe_allow_html=True)
        
        if len(stats_df) > 0:
            cluster_summary = stats_df.groupby('Кластер').agg(
                Количество=('ID', 'count'),
                Средняя_успеваемость=('Процент от максимума', 'mean'),
                Среднее_количество_практик=('Выполнено практик', 'mean')
            ).round(1)
            cluster_summary = cluster_summary.rename(columns={
                'Количество': 'Количество студентов',
                'Средняя_успеваемость': 'Средняя успеваемость, %',
                'Среднее_количество_практик': 'Среднее кол-во выполненных практик'
            })
            st.dataframe(cluster_summary, use_container_width=True)
        
        st.markdown("---")
        st.markdown("<h3 style='color:#8B0000;'>Рекомендации по курсу</h3>", unsafe_allow_html=True)
        
        problem_practices = []
        for n in practice_nums:
            fail_rate = (df[f'Оценка_{n}'] <= 0).sum() / len(df) * 100
            if fail_rate > 30:
                problem_practices.append((n, fail_rate))
        
        if problem_practices:
            st.warning("**Практики, вызывающие наибольшие трудности:**")
            for n, rate in sorted(problem_practices, key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"- Практика {n}: не выполнили {rate:.0f}% студентов")
            st.info("Рекомендуется добавить поясняющие материалы и провести дополнительные консультации.")
        else:
            st.success("Курс хорошо структурирован, критических проблем не выявлено")

else:
    st.markdown('<div class="info-box">📁 Загрузите файлы практик для начала анализа</div>', unsafe_allow_html=True)
    st.markdown("""
    **Инструкция:**
    1. Нажмите кнопку "Upload" и выберите файлы практик (Ctrl+клик)
    2. Программа автоматически определит номер практики из названия файла
    3. Укажите дату открытия каждой практики
    4. Нажмите "Выполнить анализ"
    """)
