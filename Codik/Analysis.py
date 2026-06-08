import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
import warnings
warnings.filterwarnings('ignore')

# Создаём папку для результатов
os.makedirs('results', exist_ok=True)

# Загрузка данных
file_path = r"C:\Users\BORDIUR\Documents\Diplom\Obshi_Excel.xlsx"
df = pd.read_excel(file_path)

# Замена NaN на 0
df = df.fillna(0)

# Настройки и функции
practice_nums = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # исключаем практику 2

def col_score(n):
    return f"Оценка {n}"

def grade_category(x):
    if pd.isna(x) or x == 0:
        return 'Н'
    elif x < 2.0:
        return '2'
    elif 2.0 <= x < 3.1:
        return '3'
    elif 3.1 <= x < 4.1:
        return '4'
    else:
        return '5'

# M1: Кодирование статусных последовательностей
print("\n" + "="*60)
print("КОДИРОВАНИЕ СТАТУСНЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
print("="*60)

sequences = []
for idx, row in df.iterrows():
    seq = ''
    for n in practice_nums:
        score = row[col_score(n)]
        seq += grade_category(score)
    sequences.append(seq)

seq_df = pd.DataFrame({
    'ID_Студента': df['ID_Студента'].values,
    'Последовательность': sequences
})

print(f"Сформировано {len(seq_df)} статусных последовательностей")
print("\nПримеры последовательностей:")
print(seq_df.head(10).to_string(index=False))
seq_df.to_csv('results/status_sequences.csv', index=False)

# Преобразоввание в числовой вектор
print("\n" + "="*60)
print("ПРЕОБРАЗОВАНИЕ В ЧИСЛОВЫЕ ВЕКТОРЫ")
print("="*60)

all_symbols = ['2', '3', '4', '5', 'Н']
encoder = OneHotEncoder(categories=[all_symbols], sparse_output=False)

encoded_vectors = []
for seq in sequences:
    symbols = list(seq)
    symbols_array = np.array(symbols).reshape(-1, 1)
    encoded = encoder.fit_transform(symbols_array)
    encoded_flat = encoded.flatten()
    encoded_vectors.append(encoded_flat)

X = np.array(encoded_vectors)
print(f"Размерность признакового пространства: {X.shape}")

# M2: Иерархическая кластеризация с отсечением для k = 4
print("\n" + "="*60)
print("ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ (k = 4)")
print("="*60)

# Установка числа кластеров
SELECTED_K = 4
print(f"Выбранное число кластеров: k = {SELECTED_K}")

# Построение матрицы расстояний и иерархической кластеризации
distance_matrix = pdist(X, metric='euclidean')
linkage_matrix = linkage(distance_matrix, method='ward')

# Дендрограмма с отсечением для k=4
plt.figure(figsize=(14, 10))
dendrogram(linkage_matrix, 
           labels=df['ID_Студента'].astype(str).values, 
           leaf_rotation=90, 
           leaf_font_size=9,
           color_threshold=linkage_matrix[-SELECTED_K+1, 2])
plt.xlabel('ID студента', fontsize=12)
plt.ylabel('Евклидово расстояние', fontsize=12)
plt.title('Дендрограмма иерархической кластеризации (метод Уорда)', fontsize=14)

# Добавляем линию отсечения для k=4
threshold = linkage_matrix[-SELECTED_K+1, 2]
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
            label=f'Отсечение для k={SELECTED_K} (расстояние = {threshold:.2f})')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('results/dendrogram_k4.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Дендрограмма сохранена в 'results/dendrogram_k4.png'")
print(f"Порог отсечения для k={SELECTED_K}: {threshold:.2f}")

# Выполняем иерархическую кластеризацию
hierarchical = AgglomerativeClustering(n_clusters=SELECTED_K, linkage='ward')
clusters = hierarchical.fit_predict(X)
seq_df['Кластер'] = clusters

print(f"\nРаспределение студентов по кластерам:")
cluster_counts = seq_df['Кластер'].value_counts().sort_index()
for cluster_id in sorted(cluster_counts.index):
    print(f"  Кластер {cluster_id}: {cluster_counts[cluster_id]} студентов")

# Вывод студентов по кластерам
print("\nСостав кластеров (ID студентов):")
for cluster_id in sorted(seq_df['Кластер'].unique()):
    students = seq_df[seq_df['Кластер'] == cluster_id]['ID_Студента'].tolist()
    print(f"  Кластер {cluster_id}: {students}")

# M3: Расстояние Хэмминга до эталона
print("\n" + "="*60)
print("РАССТОЯНИЕ ХЭММИНГА ДО ЭТАЛОНА")
print("="*60)

etalon = '5' * len(practice_nums)
print(f"Эталонная последовательность (все '5'): {etalon}")

hamming_distances = []
for seq in sequences:
    distance = sum(1 for a, b in zip(seq, etalon) if a != b)
    hamming_distances.append(distance)

seq_df['Расстояние_Хэмминга'] = hamming_distances

print(f"\nСреднее расстояние до эталона по всем студентам: {np.mean(hamming_distances):.2f}")
print("\nРасстояния Хэмминга по кластерам:")
for cluster_id in sorted(seq_df['Кластер'].unique()):
    cluster_distances = seq_df[seq_df['Кластер'] == cluster_id]['Расстояние_Хэмминга']
    print(f"  Кластер {cluster_id}: среднее = {cluster_distances.mean():.2f}, "
          f"мин = {cluster_distances.min():.0f}, макс = {cluster_distances.max():.0f}")

# M4: Анализ выживаемости
print("\n" + "="*60)
print("АНАЛИЗ ВЫЖИВАЕМОСТИ")
print("="*60)

def survival_curve(df_students, practice_list):
    """Расчёт кривой выживаемости для заданного набора студентов"""
    survival = []
    n_students = len(df_students)
    if n_students == 0:
        return [0] * len(practice_list)
    for k_idx in range(1, len(practice_list) + 1):
        alive = 0
        for idx, row in df_students.iterrows():
            all_done = True
            for n in practice_list[:k_idx]:
                score = row[col_score(n)]
                if score <= 0:
                    all_done = False
                    break
            if all_done:
                alive += 1
        survival.append(alive / n_students * 100)
    return survival

# Общая кривая выживаемости
survival_overall = survival_curve(df, practice_nums)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(practice_nums) + 1), survival_overall, 'o-', 
         color='steelblue', linewidth=2, markersize=8)
plt.fill_between(range(1, len(practice_nums) + 1), survival_overall, alpha=0.3, color='steelblue')
plt.xlabel('Номер практики (по порядку)', fontsize=12)
plt.ylabel('Доля "выживших" студентов, %', fontsize=12)
plt.title('Общая кривая выживаемости (последовательное выполнение)', fontsize=14)
plt.xticks(range(1, len(practice_nums) + 1), practice_nums, rotation=45)
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
for i, s in enumerate(survival_overall):
    plt.text(i + 1, s + 2, f'{s:.0f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/survival_curve_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Доля студентов, выполнивших все 14 практик: {survival_overall[-1]:.1f}%")

# Кривые выживаемости по кластерам
plt.figure(figsize=(12, 7))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
for cluster_id in sorted(seq_df['Кластер'].unique()):
    cluster_students = df[df['ID_Студента'].isin(
        seq_df[seq_df['Кластер'] == cluster_id]['ID_Студента'].values
    )]
    survival_cluster = survival_curve(cluster_students, practice_nums)
    plt.plot(range(1, len(practice_nums) + 1), survival_cluster, 'o-',
             color=colors[cluster_id % len(colors)], linewidth=2, markersize=6,
             label=f'Кластер {cluster_id} (n={len(cluster_students)})')

plt.xlabel('Номер практики (по порядку)', fontsize=12)
plt.ylabel('Доля "выживших" студентов, %', fontsize=12)
plt.title('Кривые выживаемости по кластерам (k=4)', fontsize=14)
plt.xticks(range(1, len(practice_nums) + 1), practice_nums, rotation=45)
plt.ylim(0, 105)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/survival_curve_by_cluster.png', dpi=150, bbox_inches='tight')
plt.close()

# вывод таблицы кривых ввыживаемости по кластерам
print("\n" + "="*60)
print("ТАБЛИЦА ВЫЖИВАЕМОСТИ ПО КЛАСТЕРАМ")
print("="*60)

# Расчёт выживаемости для каждого кластера
survival_by_cluster = {}
for cluster_id in sorted(seq_df['Кластер'].unique()):
    cluster_students = df[df['ID_Студента'].isin(
        seq_df[seq_df['Кластер'] == cluster_id]['ID_Студента'].values
    )]
    survival_by_cluster[cluster_id] = survival_curve(cluster_students, practice_nums)

# Создание DataFrame для таблицы
survival_table = pd.DataFrame()
survival_table['Практика'] = practice_nums

for cluster_id in sorted(seq_df['Кластер'].unique()):
    survival_table[f'Кластер {cluster_id} (n={len(seq_df[seq_df["Кластер"] == cluster_id])})'] = survival_by_cluster[cluster_id]

# Вывод таблицы в консоль
print("\nТаблица выживаемости по кластерам (%):")
print(survival_table.to_string(index=False))

# Сохранение таблицы в CSV
survival_table.to_csv('results/survival_table_by_cluster.csv', index=False)
print("\nТаблица сохранена в 'results/survival_table_by_cluster.csv'")

# Позиционный анализ (узкие места)
print("\n" + "="*60)
print("ПОЗИЦИОННЫЙ АНАЛИЗ (УЗКИЕ МЕСТА КУРСА)")
print("="*60)

position_stats = []
for n in practice_nums:
    scores = df[col_score(n)].values
    n_students = len(scores)
    pct_skip = 100 * (scores == 0).sum() / n_students
    completed = scores[scores > 0]
    if len(completed) > 0:
        pct_low = 100 * ((completed < 3.5).sum()) / len(completed)
        pct_high = 100 * ((completed >= 3.5).sum()) / len(completed)
    else:
        pct_low = pct_high = 0
    position_stats.append({
        'Практика': n,
        'Доля не приступавших (%)': pct_skip,
        'Доля низких оценок (2-3) (%)': pct_low,
        'Доля высоких оценок (4-5) (%)': pct_high
    })

pos_df = pd.DataFrame(position_stats)
print(pos_df.to_string(index=False))

pos_df['Проблемный_индекс'] = pos_df['Доля не приступавших (%)'] + pos_df['Доля низких оценок (2-3) (%)']
pos_df_sorted = pos_df.sort_values('Проблемный_индекс', ascending=False)

print("\nУзкие места курса (наиболее проблемные практики):")
for i, row in pos_df_sorted.head(5).iterrows():
    print(f"  Практика {int(row['Практика'])}: не приступали {row['Доля не приступавших (%)']:.1f}%, "
          f"низкие оценки {row['Доля низких оценок (2-3) (%)']:.1f}%")

# M5: Статистическая провверка различий (ANOVA)
print("\n" + "="*60)
print("СТАТИСТИЧЕСКАЯ ПРОВЕРКА РАЗЛИЧИЙ (ANOVA)")
print("="*60)

# Расчёт итоговых показателей для каждого студента
student_stats = []
for idx, row in df.iterrows():
    student_id = row['ID_Студента']
    total_score = 0
    total_time = 0
    total_delay = 0
    count_done = 0
    for n in practice_nums:
        score = row[col_score(n)]
        if score > 0:
            total_score += score
            count_done += 1
            time_col = f"Затраченное время {n} (мин)"
            delay_col = f"Запаздывание {n}"
            if time_col in df.columns and row[time_col] > 0:
                total_time += row[time_col]
            if delay_col in df.columns and row[delay_col] > 0:
                total_delay += row[delay_col]
    percent_of_max = total_score / (len(practice_nums) * 5) * 100
    student_stats.append({
        'ID': student_id,
        'total_score': total_score,
        'total_time': total_time,
        'total_delay': total_delay,
        'practices_done': count_done,
        'percent_of_max': percent_of_max
    })

stats_df = pd.DataFrame(student_stats)
stats_df = stats_df.merge(seq_df[['ID_Студента', 'Кластер']], left_on='ID', right_on='ID_Студента', how='left')
stats_df.drop('ID_Студента', axis=1, inplace=True)

print("\nРезультаты дисперсионного анализа (ANOVA) для k=4:")
print("-" * 55)
for metric in ['total_time', 'total_delay', 'practices_done', 'percent_of_max']:
    groups = []
    for cluster_id in sorted(stats_df['Кластер'].unique()):
        group_data = stats_df[stats_df['Кластер'] == cluster_id][metric].values
        if len(group_data) > 0:
            groups.append(group_data)
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        significance = "ЗНАЧИМО" if p_val < 0.05 else "не значимо"
        print(f"{metric:25s}: F = {f_stat:.4f}, p = {p_val:.6f} ({significance})")

# Тест Тьюки
print("\n" + "="*60)
print("АНАЛИЗ (ТЕСТ ТЬЮКИ)")
print("="*60)

metrics_for_tukey = ['percent_of_max', 'total_time', 'total_delay', 'practices_done']
metric_names_ru = {
    'percent_of_max': 'Процент от максимума',
    'total_time': 'Суммарное время',
    'total_delay': 'Суммарное опоздание',
    'practices_done': 'Количество выполненных практик'
}

for metric in metrics_for_tukey:
    values = []
    group_labels = []
    for idx, row in stats_df.iterrows():
        values.append(row[metric])
        group_labels.append(f"Кл{int(row['Кластер'])}")
    
    tukey = pairwise_tukeyhsd(values, group_labels, alpha=0.05)
    
    print(f"\n{'='*50}")
    print(f"Тест Тьюки для показателя: {metric_names_ru[metric]}")
    print(f"{'='*50}")
    
    print(tukey)
    
    print(f"\nИнтерпретация для {metric_names_ru[metric]}:")
    
    tukey_str = str(tukey)
    lines = tukey_str.strip().split('\n')
    
    data_lines = []
    header_found = False
    separator_found = False
    
    for line in lines:
        if 'group1' in line and 'group2' in line:
            header_found = True
            continue
        if header_found and not separator_found:
            if '-' in line or '=' in line:
                separator_found = True
            continue
        if separator_found and line.strip():
            data_lines.append(line.strip())
    
    significant_found = False
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 6:
            group1 = parts[0]
            group2 = parts[1]
            meandiff = float(parts[2])
            p_adj = float(parts[3])
            reject = parts[5] == 'True'
            
            if reject:
                print(f"  {group1} vs {group2}: разница = {meandiff:.2f}, p = {p_adj:.6f} (ЗНАЧИМО)")
                significant_found = True
    
    if not significant_found:
        print("  Статистически значимых различий между кластерами не обнаружено")
    
    try:
        if hasattr(tukey, '_results_table'):
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                    columns=tukey._results_table.data[0])
            tukey_df.to_csv(f'results/tukey_{metric}.csv', index=False)
            print(f"  Результаты сохранены в 'results/tukey_{metric}.csv'")
        else:
            with open(f'results/tukey_{metric}.txt', 'w', encoding='utf-8') as f:
                f.write(tukey_str)
            print(f"  Результаты сохранены в 'results/tukey_{metric}.txt'")
    except Exception as e:
        print(f"  При сохранении файла возникла ошибка: {e}")

# Сводка по кластерам
print("\n" + "="*60)
print("СВОДКА ПО КЛАСТЕРАМ (k=4)")
print("="*60)

cluster_summary = stats_df.groupby('Кластер').agg(
    Количество_студентов=('ID', 'count'),
    Средний_процент_от_максимума=('percent_of_max', 'mean'),
    Среднее_суммарное_время=('total_time', 'mean'),
    Среднее_суммарное_опоздание=('total_delay', 'mean'),
    Среднее_количество_выполненных_практик=('practices_done', 'mean')
).round(2)

print(cluster_summary.to_string())

# Модальные последовательности для кластеров
print("\n" + "="*60)
print("МОДАЛЬНЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ КЛАСТЕРОВ")
print("="*60)

for cluster_id in sorted(seq_df['Кластер'].unique()):
    cluster_seqs = seq_df[seq_df['Кластер'] == cluster_id]['Последовательность']
    modal_seq = cluster_seqs.mode()[0] if len(cluster_seqs) > 0 else "Нет данных"
    freq = (cluster_seqs == modal_seq).sum()
    pct = 100 * freq / len(cluster_seqs)
    print(f"Кластер {cluster_id} (n={len(cluster_seqs)}):")
    print(f"  Наиболее частая последовательность: {modal_seq}")
    print(f"  Встречается: {freq} раз ({pct:.1f}% студентов кластера)")

# Визуализация сраввнения кластеровв
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Процент от максимума
data_to_plot = [stats_df[stats_df['Кластер'] == c]['percent_of_max'].values 
                for c in sorted(stats_df['Кластер'].unique())]
axes[0, 0].boxplot(data_to_plot, labels=[f'Кластер {c}' for c in sorted(stats_df['Кластер'].unique())])
axes[0, 0].set_ylabel('Процент от максимума, %')
axes[0, 0].set_title('Успеваемость по кластерам')
axes[0, 0].grid(True, alpha=0.3)

# Суммарное время
data_to_plot = [stats_df[stats_df['Кластер'] == c]['total_time'].values 
                for c in sorted(stats_df['Кластер'].unique())]
axes[0, 1].boxplot(data_to_plot, labels=[f'Кластер {c}' for c in sorted(stats_df['Кластер'].unique())])
axes[0, 1].set_ylabel('Суммарное время, мин')
axes[0, 1].set_title('Время выполнения по кластерам')
axes[0, 1].grid(True, alpha=0.3)

# Суммарное опоздание
data_to_plot = [stats_df[stats_df['Кластер'] == c]['total_delay'].values 
                for c in sorted(stats_df['Кластер'].unique())]
axes[1, 0].boxplot(data_to_plot, labels=[f'Кластер {c}' for c in sorted(stats_df['Кластер'].unique())])
axes[1, 0].set_ylabel('Суммарное опоздание, дни')
axes[1, 0].set_title('Опоздания по кластерам')
axes[1, 0].grid(True, alpha=0.3)

# Количество выполненных практик
data_to_plot = [stats_df[stats_df['Кластер'] == c]['practices_done'].values 
                for c in sorted(stats_df['Кластер'].unique())]
axes[1, 1].boxplot(data_to_plot, labels=[f'Кластер {c}' for c in sorted(stats_df['Кластер'].unique())])
axes[1, 1].set_ylabel('Количество выполненных практик')
axes[1, 1].set_title('Активность по кластерам')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/cluster_comparison_k4.png', dpi=150, bbox_inches='tight')
plt.close()

# Сохранение результатов
seq_df.to_csv('results/final_results_k4.csv', index=False)
pos_df.to_csv('results/bottlenecks.csv', index=False)
cluster_summary.to_csv('results/cluster_summary_k4.csv')
stats_df.to_csv('results/student_stats_k4.csv', index=False)