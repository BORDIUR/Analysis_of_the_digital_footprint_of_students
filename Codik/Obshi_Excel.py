import pandas as pd
import os
import re
from datetime import datetime, date

# Даты начала практик
practice_start_dates = {
    1: "03.09.2025",
    2: "16.09.2025",
    3: "01.10.2025",
    4: "14.10.2025",
    5: "21.10.2025",
    6: "05.11.2025",
    7: "07.11.2025",
    8: "12.11.2025",
    9: "12.11.2025",
    10: "17.11.2025",
    11: "17.11.2025",
    12: "18.11.2025",
    13: "20.11.2025",
    14: "24.11.2025",
    15: "25.11.2025"
}

# Словарь месяцев для преобразования русских дат
months_ru = {
    'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
    'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
    'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
}

def parse_russian_date(date_str):
    """Преобразует дату вида '6 сентября 2025 17:17' в дату (без времени)"""
    if pd.isna(date_str):
        return None
    try:
        # Разделяем дату и время
        parts = date_str.strip().split()
        if len(parts) >= 4:
            day = parts[0].zfill(2)
            month_ru = parts[1]
            year = parts[2]
            
            month = months_ru.get(month_ru, '01')
            date_str_only = f"{day}.{month}.{year}"
            return datetime.strptime(date_str_only, "%d.%m.%Y").date()
    except Exception as e:
        print(f"Ошибка парсинга даты '{date_str}': {e}")
        return None
    return None

def time_to_minutes(time_str):
    """Преобразует время из форматов '1 ч.', '56 мин. 14 сек.' в минуты и округляет до целого"""
    if pd.isna(time_str):
        return None
    
    time_str = str(time_str).strip().lower()
    total_minutes = 0
    
    # Ищем часы
    hours_match = re.search(r'(\d+)\s*ч', time_str)
    if hours_match:
        total_minutes += int(hours_match.group(1)) * 60
    
    # Ищем минуты
    minutes_match = re.search(r'(\d+)\s*мин', time_str)
    if minutes_match:
        total_minutes += int(minutes_match.group(1))
    
    # Ищем секунды (округляем до целого)
    seconds_match = re.search(r'(\d+\.?\d*)\s*сек', time_str)
    if seconds_match:
        total_minutes += round(float(seconds_match.group(1)) / 60)
    
    return round(total_minutes)

input_folder = r"C:\Users\BORDIUR\Documents\Diplom\Dataset"
output_file = "Obshi_Excel.xlsx"

# Словарь для хранения данных студентов
students_data = {}

# Обрабатываем практики с 1 по 15
for i in range(1, 16):
    file_path = os.path.join(input_folder, f"ПЗ_{i}.xlsx")
    
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден, пропускаем")
        continue
    
    df = pd.read_excel(file_path)
    
    # Проверяем наличие колонок
    required_cols = ['Фамилия', 'Имя', 'Тест начат', 'Завершено', 'Затраченное время', 'Оценка']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Ошибка в файле ПЗ_{i}.xlsx: нет колонок {missing_cols}")
        continue
    
    # Объединяем ФИО (убираем звездочки и лишние пробелы)
    df['ФИО'] = df['Фамилия'].astype(str).str.replace('★', '').str.strip() + " " + df['Имя'].astype(str).str.strip()
    
    # Парсим даты (только дата, без времени)
    df['Тест начат_parsed'] = df['Тест начат'].apply(parse_russian_date)
    df['Завершено_parsed'] = df['Завершено'].apply(parse_russian_date)
    
    # Преобразуем дату начала практики в date
    start_date = datetime.strptime(practice_start_dates[i], "%d.%m.%Y").date()
    
    # Рассчитываем запаздывание в днях (используем list comprehension)
    df['Запаздывание'] = [(date_val - start_date).days if date_val is not None else None for date_val in df['Тест начат_parsed']]
    
    # Переводим затраченное время в минуты (целое число)
    df['Время_мин'] = df['Затраченное время'].apply(time_to_minutes)
    
    # Преобразуем оценку (запятая на точку)
    df['Оценка_clean'] = df['Оценка'].astype(str).str.replace(',', '.').str.replace(' ', '')
    df['Оценка_clean'] = pd.to_numeric(df['Оценка_clean'], errors='coerce')
    
    # Записываем данные для каждого студента
    for _, row in df.iterrows():
        fio = row['ФИО']
        if fio not in students_data:
            students_data[fio] = {}
        
        students_data[fio][i] = {
            'запаздывание': row['Запаздывание'],
            'тест_начат': row['Тест начат_parsed'],
            'завершено': row['Завершено_parsed'],
            'время_мин': row['Время_мин'],
            'оценка': row['Оценка_clean']
        }
    
    print(f"Обработана практика {i}: {len(df)} студентов")

# Если нет данных
if not students_data:
    print("Нет данных для объединения. Проверьте путь к файлам.")
else:
    # Создаём итоговый DataFrame
    all_fios = sorted(students_data.keys())
    
    # Формируем названия колонок
    columns = []
    for i in range(1, 16):
        columns.append(f"Запаздывание {i}")
        columns.append(f"Тест начат {i}")
        columns.append(f"Завершено {i}")
        columns.append(f"Затраченное время {i} (мин)")
        columns.append(f"Оценка {i}")
    
    result_df = pd.DataFrame(index=all_fios, columns=columns)
    
    # Заполняем данные
    for fio in all_fios:
        for practice_num, data in students_data[fio].items():
            result_df.at[fio, f"Запаздывание {practice_num}"] = data['запаздывание']
            result_df.at[fio, f"Тест начат {practice_num}"] = data['тест_начат']
            result_df.at[fio, f"Завершено {practice_num}"] = data['завершено']
            result_df.at[fio, f"Затраченное время {practice_num} (мин)"] = data['время_мин']
            result_df.at[fio, f"Оценка {practice_num}"] = data['оценка']
    
    # Сохраняем в Excel
    result_df.to_excel(output_file)
    print(f"\n Файл сохранён: {output_file}")
    print(f"   Всего уникальных студентов: {len(result_df)}")
    print(f"   Всего колонок: {len(result_df.columns)}")
    
    # Показываем пример результата
    print(f"\nПример первых 3 строк и первых 5 колонок:")
    print(result_df.iloc[:3, :5].to_string())