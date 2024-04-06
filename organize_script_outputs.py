import os
import shutil
import re
import argparse

# Создание парсера аргументов командной строки
parser = argparse.ArgumentParser(description="Script for creating a folder and moving files")
parser.add_argument("--input_folder", type=str, default="assets", help="Путь к папке с входными данными")
parser.add_argument("--output_folder", type=str, default="results", help="Путь к папке для сохранения результатов")
args = parser.parse_args()

# Путь к bat-файлу
bat_file = "run_scripts.bat"

# Регулярные выражения для извлечения параметров
datasets_regex = r"--datasets\s+((?:datasets\\[\w-]+\s+)+)"
evaluation_regex = r"--data_folder\s+((?:datasets\\[\w-]+(?:\s+datasets\\[\w-]+)*\s*))"
emotions_regex = r"--emotions\s+(.*?)(?=\s+(?:--output_folder|--data_folder|--datasets)|\s*$)"

# Чтение содержимого bat-файла
with open(bat_file, "r") as file:
    bat_content = file.read()

# Извлечение параметров с помощью регулярных выражений
dataset_match = re.search(datasets_regex, bat_content)
evaluation_match = re.search(evaluation_regex, bat_content)
emotions_match = re.search(emotions_regex, bat_content)

# Получение значений параметров
datasets = dataset_match.group(1).split() if dataset_match else ["Unknown"]
dataset_names = [os.path.basename(dataset) for dataset in datasets]
dataset_str = "+".join(dataset_names)

evaluation = evaluation_match.group(1).split("\\")[-1] if evaluation_match else "Unknown"
emotions = emotions_match.group(1).replace(" ", "+") if emotions_match else "Unknown"

# Создание названия папки
folder_name = f"{dataset_str}___{evaluation}___{emotions}"

# Создание папки в указанной директории
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)
folder_path = os.path.join(output_folder, folder_name)
os.makedirs(folder_path, exist_ok=True)

# Перемещение файлов из указанной папки в созданную папку
input_folder = args.input_folder
for file_name in os.listdir(input_folder):
    src_path = os.path.join(input_folder, file_name)
    dst_path = os.path.join(folder_path, file_name)
    shutil.copy(src_path, dst_path)

# Создание файла txt с информацией о запуске скрипта
txt_file = os.path.join(folder_path, "script_run_info.txt")
with open(txt_file, "w") as file:
    file.write(f"Bat file: {bat_file}\n")
    file.write(f"Datasets: {', '.join(datasets)}\n")
    file.write(f"Evaluation: {evaluation}\n")
    file.write(f"Emotions: {emotions}\n")
    file.write(f"Input folder: {input_folder}\n")
    file.write(f"Output folder: {folder_path}\n")
    file.write("\nBat file content:\n")
    file.write(bat_content)

print(f"Файлы из папки {input_folder} скопированы в {folder_path}")
print(f"Инфо: {txt_file}")