import os
import random
import argparse
import shutil
from glob import glob

emotions_mapping_crema_d = {
    'ANG': 'Anger',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad',
}

emotions_mapping_ravdess = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

emotions_mapping_savee = {
    'a': 'Angry',
    'd': 'Disgust',
    'f': 'Fear',
    'h': 'Happy',
    'n': 'Neutral',
    'sa': 'Sad',
    'su': 'Surprise'
}

emotions_mapping_tess = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'neutral': 'Neutral',
    'sad': 'Sad',
    'ps': 'Surprise'
}

def crema_d_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    emotion_files = {}

    for file in audios:
        filename = os.path.basename(file)
        part = filename.split('_')
        emotion = emotions_mapping_crema_d.get(part[2], 'Unknown')
        if emotion not in emotion_files:
            emotion_files[emotion] = []
        emotion_files[emotion].append(file)

    return emotion_files

def ravdess_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    emotion_files = {}

    for file in audios:
        filename = os.path.basename(file)
        parts = filename.split('-')
        emotion = emotions_mapping_ravdess.get(parts[2], 'Unknown')
        if emotion not in emotion_files:
            emotion_files[emotion] = []
        emotion_files[emotion].append(file)

    return emotion_files

def savee_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    emotion_files = {}

    for file in audios:
        filename = os.path.basename(file)
        parts = filename.split('.')
        if len(parts) == 2 and len(parts[0]) >= 1:
            if parts[0][:2] in ['sa', 'su']:
                emotion = emotions_mapping_savee.get(parts[0][:2], 'Unknown')
            else:
                emotion = emotions_mapping_savee.get(parts[0][0], 'Unknown')
        else:
            emotion = 'Unknown'
        if emotion not in emotion_files:
            emotion_files[emotion] = []
        emotion_files[emotion].append(file)

    return emotion_files

def tess_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    emotion_files = {}

    for file in audios:
        filename = os.path.basename(file)
        parts = filename.split('_')
        emotion = emotions_mapping_tess.get(parts[-1].split('.')[0], 'Unknown')
        if emotion not in emotion_files:
            emotion_files[emotion] = []
        emotion_files[emotion].append(file)

    return emotion_files

def create_subset(datasets, count_per_class, output_folder):
    os.makedirs(os.path.join(output_folder, "subset1"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "subset2"), exist_ok=True)

    for dataset, load_func in datasets.items():
        emotion_files = load_func(dataset)
        for emotion, files in emotion_files.items():
            random.shuffle(files)
            subset2 = files[:count_per_class]
            subset1 = files[count_per_class:]

            for file in subset1:
                rel_path = os.path.relpath(file, dataset)
                dest_path = os.path.join(output_folder, "subset1", os.path.basename(dataset), rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file, dest_path)

            for file in subset2:
                rel_path = os.path.relpath(file, dataset)
                dest_path = os.path.join(output_folder, "subset2", os.path.basename(dataset), rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file, dest_path)

def main():
    parser = argparse.ArgumentParser(description="Разделение данных на тренировочное подмножество и тестовое.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Путь к датасетам.")
    parser.add_argument("--count_per_class", type=int, required=True, help="Количество эмоций для тестовой выборки для каждого класса.")
    parser.add_argument("--output_folder", type=str, required=True, help="Путь к папке для сохранения результатов.")
    args = parser.parse_args()

    dataset_load_funcs = {
        'datasets/CREMA-D': crema_d_load_and_preprocess_data,
        'datasets/RAVDESS': ravdess_load_and_preprocess_data,
        'datasets/SAVEE': savee_load_and_preprocess_data,
        'datasets/TESS': tess_load_and_preprocess_data
    }

    create_subset(dataset_load_funcs, args.count_per_class, args.output_folder)
    print(f"Выборки созданы с {args.count_per_class} файлами на каждую эмоцию для каждого набора данных в subset2.")
    print(f"Subset1 содержит все файлы, кроме тех, которые находятся в subset2.")
    print(f"Выборки сохранены в {args.output_folder}/subset1 и {args.output_folder}/subset2 с той же структурой папок, что и исходные наборы данных.")
    
if __name__ == "__main__":
    main()