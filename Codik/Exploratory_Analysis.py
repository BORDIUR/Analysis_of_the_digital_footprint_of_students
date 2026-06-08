import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, f_oneway

# Создаём папку для графиков
os.makedirs('plots', exist_ok=True)

#### Загрузка Данных ####
file_path = r"C:\Users\BORDIUR\Documents\Diplom\Obshi_Excel.xlsx"
df = pd.read_excel(file_path)

# Заменяем NaN на 0 в числовых колонках для корректной работы
num_cols = df.select_dtypes(include=["number"]).columns
df[num_cols] = df[num_cols].fillna(0)

print("="*60)
print("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ ЦИФРОВОГО ОБРАЗОВАТЕЛЬНОГО СЛЕДА")
print("="*60)

#### Настройка ####
practice_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def col_score(n):
    return f"Оценка {n}"

def col_time(n):
    return f"Затраченное время {n} (мин)"

def col_delay(n):
    return f"Запаздывание {n}"

def grade_category(x):
    if pd.isna(x) or x == 0:
        return None
    if x < 2.0:
        return 2
    elif 2.0 <= x < 3.1:
        return 3
    elif 3.1 <= x < 4.1:
        return 4
    else:
        return 5

all_students = df["ID_Студента"].nunique()

#### Метрики по каждой практике ####
print(f"\n1. Метрики по каждой практике:")

rows = []
for n in practice_nums:
    score_col = col_score(n)
    time_col = col_time(n)
    delay_col = col_delay(n)

    if score_col not in df.columns:
        continue

    sub = df[["ID_Студента", score_col, time_col, delay_col]].copy()

    # выполнена практика: есть оценка > 0
    done_mask = (sub[score_col] > 0)
    students_done = sub.loc[done_mask, "ID_Студента"].nunique()
    pct_students_done = 100 * students_done / all_students

    # категории оценок
    sub["grade_cat"] = sub[score_col].apply(grade_category)
    
    total_done = done_mask.sum()
    if total_done > 0:
        pct_2 = 100 * (sub.loc[done_mask, "grade_cat"] == 2).sum() / total_done
        pct_3 = 100 * (sub.loc[done_mask, "grade_cat"] == 3).sum() / total_done
        pct_4 = 100 * (sub.loc[done_mask, "grade_cat"] == 4).sum() / total_done
        pct_5 = 100 * (sub.loc[done_mask, "grade_cat"] == 5).sum() / total_done
    else:
        pct_2 = pct_3 = pct_4 = pct_5 = np.nan

    # студенты без опозданий среди выполнивших
    no_delay_mask = (sub[delay_col] <= 0) & done_mask
    pct_no_delay = 100 * no_delay_mask.sum() / total_done if total_done > 0 else np.nan

    # медианные величины только по выполненным
    med_score = sub.loc[done_mask, score_col].median()
    med_time = sub.loc[done_mask, time_col].median()
    med_delay = sub.loc[done_mask, delay_col].median()

    print(f"   Практика {n:2d}: выполнено {students_done:2d} студ. ({pct_students_done:5.1f}%), "
          f"мед.оценка={med_score:.2f}, мед.время={med_time:.0f} мин, мед.опоздание={med_delay:.0f} дн")

    rows.append({
        "Практика": n,
        "% студентов, выполнивших": pct_students_done,
        "% с оценкой 2": pct_2,
        "% с оценкой 3": pct_3,
        "% с оценкой 4": pct_4,
        "% с оценкой 5": pct_5,
        "% без опозданий": pct_no_delay,
        "Медианная оценка": med_score,
        "Медианное время (мин)": med_time,
        "Медианное опоздание (дни)": med_delay,
    })

practice_stats = pd.DataFrame(rows).set_index("Практика")
print("\n   Полная таблица метрик:")
print(practice_stats.to_string())

# Сохраняем таблицу в CSV
practice_stats.to_csv("plots/practice_stats.csv")

#### График выполненых практик ####
plt.figure(figsize=(12, 5))
plt.bar(practice_stats.index, practice_stats["% студентов, выполнивших"], 
        color='steelblue', edgecolor='black')
plt.xlabel('Номер практики', fontsize=12)
plt.ylabel('Доля выполнивших, %', fontsize=12)
plt.title('Доля студентов, выполнивших каждую практику', fontsize=14)
plt.xticks(practice_stats.index, rotation=45)
for i, rate in enumerate(practice_stats["% студентов, выполнивших"]):
    plt.text(i + 1, rate + 1, f'{rate:.0f}%', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('plots/completion_rate.png', dpi=150, bbox_inches='tight')
plt.close()

#### Корреляции метрик по практикам ####
print(f"\n2. Корреляции метрик по практикам:")
corr_matrix = practice_stats[[
    "Медианная оценка",
    "Медианное время (мин)",
    "Медианное опоздание (дни)",
    "% без опозданий"
]].corr()

print("Корреляции между агрегированными метриками по практикам:")
print(corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Корреляции метрик по практикам", fontsize=12)
plt.tight_layout()
plt.savefig('plots/metrics_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

#### Корреляции по каждой практике отдельно #### 
print(f"\n3. Корреляции по каждой практике (оценка vs время, оценка vs опоздание):")

corr_by_practice = []

for n in practice_nums:
    score_col = col_score(n)
    time_col = col_time(n)
    delay_col = col_delay(n)
    
    if not all(c in df.columns for c in [score_col, time_col, delay_col]):
        continue
    
    # Берём только выполнивших (оценка > 0)
    mask = (df[score_col] > 0)
    if mask.sum() < 3:
        print(f"   Практика {n}: недостаточно данных для корреляции")
        continue
    
    scores = df.loc[mask, score_col].values
    times = df.loc[mask, time_col].values
    delays = df.loc[mask, delay_col].values
    
    # Корреляция оценка-время
    corr_st, p_st = pearsonr(scores, times)
    # Корреляция оценка-опоздание
    corr_sd, p_sd = pearsonr(scores, delays)
    
    corr_by_practice.append({
        "Практика": n,
        "Кол-во студентов": mask.sum(),
        "Корр оценка-время": corr_st,
        "p-value (оценка-время)": p_st,
        "Корр оценка-опоздание": corr_sd,
        "p-value (оценка-опоздание)": p_sd
    })
    
    print(f"   Практика {n:2d} (n={mask.sum():2d}): "
          f"corr(оценка, время)={corr_st:.3f} (p={p_st:.4f}), "
          f"corr(оценка, опоздание)={corr_sd:.3f} (p={p_sd:.4f})")

corr_df = pd.DataFrame(corr_by_practice)
corr_df.to_csv("plots/correlations_by_practice.csv", index=False)

#### Графики распределения времени выполнения по каждой практике ####
print(f"\n4. Построение графиков распределения времени выполнения по каждой практике:")

for n in practice_nums:
    score_col = col_score(n)
    time_col = col_time(n)
    if score_col not in df.columns or time_col not in df.columns:
        continue
    
    # Берём только выполнивших (оценка > 0)
    mask = (df[score_col] > 0) & (df[time_col] > 0)
    times = df.loc[mask, time_col].dropna()
    if len(times) == 0:
        print(f"   Практика {n}: нет данных для построения графика")
        continue
    
    # Удаляем выбросы для лучшей визуализации 
    times_clean = times[times < times.quantile(0.95)]

    plt.figure(figsize=(8, 5))
    sns.histplot(times_clean, bins=20, kde=True, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.xlabel('Время выполнения (минуты)', fontsize=12)
    plt.ylabel('Количество студентов', fontsize=12)
    plt.title(f'Распределение времени выполнения, практика {n}\n(выполнило {len(times)} студентов)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/time_distribution_practice_{n}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Практика {n}: график сохранён (n={len(times)}, медиана={times.median():.0f} мин)")

#### График распределения опозданий по каждой практике ####
print(f"\n4. Построение графиков распределения опозданий по каждой практике:")

for n in practice_nums:
    score_col = col_score(n)
    delay_col = col_delay(n)
    if score_col not in df.columns or delay_col not in df.columns:
        continue
    
    # Берём только выполнивших (оценка > 0)
    mask = (df[score_col] > 0)
    delays = df.loc[mask, delay_col].dropna()
    if len(delays) == 0:
        print(f"   Практика {n}: нет данных для построения графика")
        continue

    # Удаляем экстремальные выбросы для лучшей визуализации
    delays_clean = delays[delays < delays.quantile(0.95)]

    plt.figure(figsize=(8, 5))
    sns.histplot(delays_clean, bins=20, kde=True, color='salmon', edgecolor='black', alpha=0.7)
    plt.xlabel('Опоздание (дни)', fontsize=12)
    plt.ylabel('Количество студентов', fontsize=12)
    plt.title(f'Распределение опозданий, практика {n}\n(выполнило {len(delays)} студентов)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/delay_distribution_practice_{n}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Практика {n}: график сохранён (n={len(delays)}, медиана={delays.median():.0f} дн)")

#### Общая корреляционная матрица ####
print(f"\n5. Общая корреляционная матрица (по всем студентам и практикам):")

# Собираем все оценки, время и опоздания в один массив
all_scores = []
all_times = []
all_delays = []

for n in practice_nums:
    score_col = col_score(n)
    time_col = col_time(n)
    delay_col = col_delay(n)
    
    if all(c in df.columns for c in [score_col, time_col, delay_col]):
        mask = (df[score_col] > 0)
        all_scores.extend(df.loc[mask, score_col].values)
        all_times.extend(df.loc[mask, time_col].values)
        all_delays.extend(df.loc[mask, delay_col].values)

# Общая корреляция
if len(all_scores) > 2:
    overall_corr_st, p_st = pearsonr(all_scores, all_times)
    overall_corr_sd, p_sd = pearsonr(all_scores, all_delays)
    overall_corr_td, p_td = pearsonr(all_times, all_delays)
    
    print(f"   Корреляция оценка-время (всего {len(all_scores)} наблюдений): {overall_corr_st:.3f} (p={p_st:.4f})")
    print(f"   Корреляция оценка-опоздание: {overall_corr_sd:.3f} (p={p_sd:.4f})")
    print(f"   Корреляция время-опоздание: {overall_corr_td:.3f} (p={p_td:.4f})")
    
    # Общая тепловая карта
    overall_data = pd.DataFrame({
        "Оценка": all_scores,
        "Время (мин)": all_times,
        "Опоздание (дни)": all_delays
    })
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(overall_data.corr(), annot=True, fmt=".3f", cmap="coolwarm", center=0)
    plt.title("Общая корреляционная матрица\n(по всем выполненным работам)", fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/overall_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

#### Суммарные показатели по стиудентам ####
print(f"\n5. Суммарные показатели по студентам:")

# Считаем сумму баллов, времени и опозданий для каждого студента
student_scores = {}
student_times = {}
student_delays = {}
student_count_done = {}

for idx, row in df.iterrows():
    student_id = row["ID_Студента"]
    
    total_score = 0
    total_time = 0
    total_delay = 0
    count_done = 0
    
    for n in practice_nums:
        score_col = col_score(n)
        time_col = col_time(n)
        delay_col = col_delay(n)
        
        if score_col in df.columns:
            score = row[score_col]
            if score > 0:
                total_score += score
                count_done += 1
                
                if time_col in df.columns and row[time_col] > 0:
                    total_time += row[time_col]
                if delay_col in df.columns and row[delay_col] > 0:
                    total_delay += row[delay_col]
    
    student_scores[student_id] = total_score
    student_times[student_id] = total_time
    student_delays[student_id] = total_delay
    student_count_done[student_id] = count_done

students_summary = pd.DataFrame({
    "ID_Студента": list(student_scores.keys()),
    "Сумма баллов": list(student_scores.values()),
    "Суммарное время (мин)": list(student_times.values()),
    "Суммарное опоздание (дни)": list(student_delays.values()),
    "Кол-во выполненных практик": list(student_count_done.values())
})

max_possible = len(practice_nums) * 5
students_summary["Процент от максимума"] = 100 * students_summary["Сумма баллов"] / max_possible
students_summary = students_summary.sort_values("Процент от максимума", ascending=False)

print("\n   Топ-5 студентов по успеваемости:")
print(students_summary.head(5).to_string(index=False))
print("\n   Худшие 5 студентов по успеваемости:")
print(students_summary.tail(5).to_string(index=False))

# Гистограммы
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(students_summary["Процент от максимума"], bins=10, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Процент от максимума, %')
axes[0, 0].set_ylabel('Количество студентов')
axes[0, 0].set_title('Распределение по итоговой успеваемости')

axes[0, 1].hist(students_summary["Суммарное время (мин)"], bins=10, color='lightgreen', edgecolor='black')
axes[0, 1].set_xlabel('Суммарное время (минуты)')
axes[0, 1].set_ylabel('Количество студентов')
axes[0, 1].set_title('Распределение по суммарному времени')

axes[1, 0].hist(students_summary["Суммарное опоздание (дни)"], bins=10, color='salmon', edgecolor='black')
axes[1, 0].set_xlabel('Суммарное опоздание (дни)')
axes[1, 0].set_ylabel('Количество студентов')
axes[1, 0].set_title('Распределение по суммарному опозданию')

axes[1, 1].hist(students_summary["Кол-во выполненных практик"], bins=range(1, 17), color='coral', edgecolor='black')
axes[1, 1].set_xlabel('Количество выполненных практик')
axes[1, 1].set_ylabel('Количество студентов')
axes[1, 1].set_title('Распределение по количеству выполненных практик')

plt.tight_layout()
plt.savefig('plots/student_summary.png', dpi=150, bbox_inches='tight')
plt.close()

#### ANOVA ####
print(f"\n6. Статистические тесты (ANOVA):")

students_summary['group'] = pd.qcut(students_summary["Процент от максимума"], q=3, labels=['Низкая', 'Средняя', 'Высокая'])

time_by_group = [students_summary[students_summary['group'] == g]["Суммарное время (мин)"].values for g in ['Низкая', 'Средняя', 'Высокая']]
f_stat_time, p_val_time = f_oneway(*time_by_group)
print(f"   ANOVA для суммарного времени: F={f_stat_time:.3f}, p={p_val_time:.4f}")

delay_by_group = [students_summary[students_summary['group'] == g]["Суммарное опоздание (дни)"].values for g in ['Низкая', 'Средняя', 'Высокая']]
f_stat_delay, p_val_delay = f_oneway(*delay_by_group)
print(f"   ANOVA для суммарного опоздания: F={f_stat_delay:.3f}, p={p_val_delay:.4f}")

done_by_group = [students_summary[students_summary['group'] == g]["Кол-во выполненных практик"].values for g in ['Низкая', 'Средняя', 'Высокая']]
f_stat_done, p_val_done = f_oneway(*done_by_group)
print(f"   ANOVA для количества выполненных практик: F={f_stat_done:.3f}, p={p_val_done:.4f}")

#### Итоговая сводка ####
print("\n" + "="*60)
print("ИТОГОВАЯ СВОДКА РАЗВЕДОЧНОГО АНАЛИЗА")
print("="*60)
print(f"1. Всего студентов: {all_students}")
print(f"2. Всего практик: {len(practice_nums)}")
print(f"3. Доля выполнивших первую практику: {practice_stats.loc[1, '% студентов, выполнивших']:.1f}%")
print(f"4. Доля выполнивших последнюю практику: {practice_stats.loc[15, '% студентов, выполнивших']:.1f}%")
print(f"5. Падение активности: {practice_stats.loc[1, '% студентов, выполнивших'] - practice_stats.loc[15, '% студентов, выполнивших']:.1f}%")
print(f"6. Средняя медианная оценка по курсу: {practice_stats['Медианная оценка'].mean():.2f}")
print(f"7. Общая корреляция оценка-время: {overall_corr_st:.3f}" if 'overall_corr_st' in dir() else "7. Общая корреляция оценка-время: не рассчитана")
print(f"8. Средняя корреляция оценка-время по практикам: {corr_df['Корр оценка-время'].mean():.3f}")
print(f"9. Средняя корреляция оценка-опоздание по практикам: {corr_df['Корр оценка-опоздание'].mean():.3f}")