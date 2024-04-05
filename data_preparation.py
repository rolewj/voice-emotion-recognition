import librosa
import numpy as np
import pandas as pd
import os
import re
import argparse
import pickle
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import Parallel, delayed


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

emotion_level_mapping_crema_d = {
    'LO': 'Low',
    'MD': 'Medium',
    'HI': 'High',
    'XX': 'Unspecified'
}

female_id_crema_d = [ '1002', '1003', '1004', '1006', '1007', '1008', '1009',
                      '1010', '1012', '1013', '1018', '1020', '1021', '1024',
                      '1025', '1028', '1029', '1030', '1037', '1043', '1046',
                      '1047', '1049', '1052', '1053', '1054', '1055', '1056',
                      '1058', '1060', '1061', '1063', '1072', '1073', '1074',
                      '1075', '1076', '1078', '1079', '1082', '1084', '1089',
                      '1091', ]

def crema_d_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    path_list, emotion_list, emotion_level_list, sex_list = [], [], [], []

    for file in audios:
        path_list.append(file)
        filename = os.path.basename(file)
        part = filename.split('_')
        emotion = emotions_mapping_crema_d.get(part[2], 'Unknown')
        emotion_level = emotion_level_mapping_crema_d.get(part[-1].replace('.wav', ''), 'Unknown')
        sex = 'Female' if part[0] in female_id_crema_d else 'Male'
        emotion_list.append(emotion)
        emotion_level_list.append(emotion_level)
        sex_list.append(sex)

    # Создание DataFrame
    crema_df = pd.DataFrame({
        'Path': path_list,
        'Emotions': emotion_list,
    #   'Emotion levels': emotion_level_list,
        'Sex': sex_list
    })
    
    target_file = '1040_ITH_SAD_X.wav'
    file_exists = crema_df['Path'].str.contains(target_file).any()
    if file_exists:
        index_to_modify = crema_df[crema_df['Path'].str.contains(target_file)].index[0]
        crema_df.loc[index_to_modify, 'Emotion levels'] = 'Unspecified'
    else:
        print(f"Файл {target_file} не найден в датасете.")
    
    print(crema_df.Emotions.value_counts())
    
    return crema_df

def ravdess_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    path_list, emotion_list, sex_list = [], [], []

    for file in audios:
        path_list.append(file)
        filename = os.path.basename(file)
        parts = filename.split('-')
        emotion = emotions_mapping_ravdess.get(parts[2], 'Unknown')
        actor = int(parts[6].replace(".wav", ""))

        # Определение пола актёра по идентификатору: чётные - женщины, нечётные - мужчины
        sex = 'Female' if actor % 2 == 0 else 'Male'

        emotion_list.append(emotion)
        sex_list.append(sex)

    # Создание DataFrame
    ravdess_df = pd.DataFrame({
        'Path': path_list,
        'Emotions': emotion_list,
        'Sex': sex_list
    })
    
    print(ravdess_df.Emotions.value_counts())
    
    return ravdess_df

def savee_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    path_list, emotion_list, sex_list = [], [], []

    for file in audios:
        path_list.append(file)
        filename = os.path.basename(file)
        parts = filename.split('.')
        if len(parts) == 2 and len(parts[0]) >= 1:
            if parts[0][:2] in ['sa', 'su']:
                emotion = emotions_mapping_savee.get(parts[0][:2], 'Unknown')
            else:
                emotion = emotions_mapping_savee.get(parts[0][0], 'Unknown')
        else:
            emotion = 'Unknown'
        sex = 'Male'

        emotion_list.append(emotion)
        sex_list.append(sex)

    # Создание DataFrame
    savee_df = pd.DataFrame({
        'Path': path_list,
        'Emotions': emotion_list,
        'Sex': sex_list
    })
    
    print(savee_df.Emotions.value_counts())
    
    return savee_df

def tess_load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/**/*.wav", recursive=True)
    path_list, emotion_list, sex_list = [], [], []

    for file in audios:
        path_list.append(file)
        filename = os.path.basename(file)
        parts = filename.split('_')
        emotion = emotions_mapping_tess.get(parts[-1].split('.')[0], 'Unknown')
        sex = 'Female'

        emotion_list.append(emotion)
        sex_list.append(sex)

    # Создание DataFrame
    tess_df = pd.DataFrame({
        'Path': path_list,
        'Emotions': emotion_list,
        'Sex': sex_list
    })
    
    print(tess_df.Emotions.value_counts())
    
    return tess_df

def load_and_preprocess_datasets(dataset_paths, emotions_list):
    combined_df = pd.DataFrame()
    
    for dataset_path in dataset_paths:
        if "CREMA-D" in dataset_path:
            df = crema_d_load_and_preprocess_data(dataset_path)
        elif "RAVDESS" in dataset_path:
            df = ravdess_load_and_preprocess_data(dataset_path)
            df['Emotions'] = df['Emotions'].replace({'Angry': 'Anger', 'Fearful': 'Fear', 'Surprised': 'Surprise'})
        elif "SAVEE" in dataset_path:
            df = savee_load_and_preprocess_data(dataset_path)
            df['Emotions'] = df['Emotions'].replace({'Angry': 'Anger'})
        elif "TESS" in dataset_path:
            df = tess_load_and_preprocess_data(dataset_path)
            df['Emotions'] = df['Emotions'].replace({'Angry': 'Anger'})
        else:
            print(f"Неизвестный датасет: {dataset_path}")
            continue
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    if emotions_list:
        # Создаем словарь для сопоставления эмоций
        emotion_mapping = {
            'Anger': ['Anger', 'Angry'],
            'Happy': ['Happy'],
            'Sad': ['Sad'],
            'Fear': ['Fear', 'Fearful'],
            'Disgust': ['Disgust'],
            'Surprise': ['Surprise', 'Surprised', 'Pleasant Surprise'],
            'Neutral': ['Neutral'],
            'Calm': ['Calm']
        }
        
        # Создаем список выбранных эмоций с учетом различных вариантов названий
        selected_emotions = []
        for emotion in emotions_list:
            selected_emotions.extend(emotion_mapping.get(emotion.capitalize(), []))
        
        combined_df = combined_df[combined_df['Emotions'].isin(selected_emotions)]
    
    return combined_df

def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfccs(data, sr, n_mfcc=13, frame_length=2048, hop_length=512, flatten: bool = True):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=frame_length)
    return np.ravel(mfccs.T) if flatten else np.squeeze(mfccs.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    features = np.array([])

    features = np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfccs(data, sr, frame_length=frame_length, hop_length=hop_length)
    ))
    return features

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def get_features(path, duration=2.5, offset=0.6, sr=22050):
    data, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)

    original_features = extract_features(data, sr)
    noised_features = extract_features(noise(data), sr)
    pitched_features = extract_features(librosa.effects.pitch_shift(data, sr=sr, n_steps=0.7), sr)
    pitched_noised_features = extract_features(noise(librosa.effects.pitch_shift(data, sr=sr, n_steps=0.7)), sr)

    return np.vstack((original_features, noised_features, pitched_features, pitched_noised_features))


def process_feature(path, emotion):
    features = get_features(path)
    X = [element for element in features]
    Y = [emotion] * len(features)
    return X, Y, path

def parallel_feature_extraction(df):
    paths = df['Path'].tolist()
    emotions = df['Emotions'].tolist()

    # Параллельная обработка
    results = Parallel(n_jobs=-1, verbose=2)(delayed(process_feature)(path, emotion) for path, emotion in zip(paths, emotions))

    # Сбор результатов
    X = []
    Y = []
    file_paths_with_emotions = [] # Список для хранения путей к файлам вместе с эмоциями
    for result in results:
        x, y, file_path = result
        X.extend(x)
        Y.extend(y)
        file_paths_with_emotions.extend([f"{file_path},{emotion}" for emotion in y]) # Добавление путей к файлам с эмоциями для каждого фрагмента

    Features = pd.DataFrame(X)
    Features['labels'] = Y
    Features['file_path_with_emotion'] = file_paths_with_emotions # Добавление столбца с путями к файлам и эмоциями

    return Features

# def preprocess_features(Features, output_folder):
#     Features = Features.fillna(0)

#     X = Features.iloc[:, :-2].values
#     Y = Features['labels'].values

#     # Сохранение путей к файлам
#     file_path_with_emotions = Features['file_path_with_emotion'].tolist()

#     encoder = OneHotEncoder()
#     Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    
#     encoder_file_path = os.path.join(output_folder, 'encoder.pickle')
#     with open(encoder_file_path, 'wb') as f:
#         pickle.dump(encoder, f)

#     x_train, x_test, y_train, y_test, train_paths_with_emotions, test_paths_with_emotions = train_test_split(X, Y, file_path_with_emotions, random_state=0, shuffle=True)

#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)

#     scaler_file_path = os.path.join(output_folder, 'scaler.pickle')
#     with open(scaler_file_path, 'wb') as f:
#         pickle.dump(scaler, f)

#     print("Encoder и Scaler успешно загружены.")

#     x_train = np.expand_dims(x_train, axis=2)
#     x_test = np.expand_dims(x_test, axis=2)
    
#     train_paths_file = os.path.join(output_folder, 'train_paths_with_emotions.txt')
#     with open(train_paths_file, 'w') as f:
#         f.write('\n'.join(train_paths_with_emotions))
    
#     test_paths_file = os.path.join(output_folder, 'test_paths_with_emotions.txt')
#     with open(test_paths_file, 'w') as f:
#         f.write('\n'.join(test_paths_with_emotions))

#     return x_train, x_test, y_train, y_test

def preprocess_features_with_stratified_split(Features, output_folder, test_size_per_class):
    Features = Features.fillna(0)
    
    X = Features.iloc[:, :-2].values
    Y = Features['labels'].values
    
    # Сохранение путей к файлам
    file_path_with_emotions = Features['file_path_with_emotion'].tolist()

    encoder = OneHotEncoder()
    Y_encoded = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    encoder_file_path = os.path.join(output_folder, 'encoder.pickle')
    with open(encoder_file_path, 'wb') as f:
        pickle.dump(encoder, f)

    # Разделение данных на обучающую и тестовую выборки с фиксированным количеством записей для каждой эмоции и каждого датасета
    X_train, X_test, Y_train, Y_test, train_paths, test_paths = [], [], [], [], [], []
    for emotion in np.unique(Y):
        for dataset_pattern in ['^\d{4}_\w{3}_\w{3}_\w{2}\.wav$', '^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.wav$', '^[a-z]{1,2}\d{2}\.wav$', '^\w{3}_\w+_\w+\.wav$']:
            emotion_indices = [i for i, path in enumerate(file_path_with_emotions) if Y[i] == emotion and re.match(dataset_pattern, os.path.basename(path.split(',')[0]))]
            X_emotion = X[emotion_indices]
            Y_emotion = Y_encoded[emotion_indices]
            paths_emotion = [file_path_with_emotions[i] for i in emotion_indices]

            if len(X_emotion) >= test_size_per_class:
                test_indices = np.random.choice(len(X_emotion), test_size_per_class, replace=False)
                train_indices = np.setdiff1d(np.arange(len(X_emotion)), test_indices)
                X_train_emotion, X_test_emotion = X_emotion[train_indices], X_emotion[test_indices]
                Y_train_emotion, Y_test_emotion = Y_emotion[train_indices], Y_emotion[test_indices]
                train_paths_emotion = [paths_emotion[i] for i in train_indices]
                test_paths_emotion = [paths_emotion[i] for i in test_indices]
                X_train.append(X_train_emotion)
                X_test.append(X_test_emotion)
                Y_train.append(Y_train_emotion)
                Y_test.append(Y_test_emotion)
                train_paths.extend(train_paths_emotion)
                test_paths.extend(test_paths_emotion)

    x_train = np.concatenate(X_train)
    x_test = np.concatenate(X_test)
    y_train = np.concatenate(Y_train)
    y_test = np.concatenate(Y_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    scaler_file_path = os.path.join(output_folder, 'scaler.pickle')
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("Encoder и Scaler успешно загружены.")

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    train_paths_file = os.path.join(output_folder, 'train_paths_with_emotions.txt')
    with open(train_paths_file, 'w') as f:
        f.write('\n'.join(train_paths))

    test_paths_file = os.path.join(output_folder, 'test_paths_with_emotions.txt')
    with open(test_paths_file, 'w') as f:
        f.write('\n'.join(test_paths))

    return x_train, x_test, y_train, y_test

def save_report(df, report_path):
    with open(os.path.join(report_path, "data_preparation_report.txt"), 'w') as f:
        # Распределение эмоций
        emotion_distrib = df['Emotions'].value_counts()
        f.write("Распределение эмоций:\n")
        f.write(emotion_distrib.to_string())
        f.write("\n\n")

        # Распределение по полу
        sex_distrib = df['Sex'].value_counts()
        f.write("Распределение по полу:\n")
        f.write(sex_distrib.to_string())
        f.write("\n\n")

        # Распределение эмоций по полу
        emotion_sex_distrib = df.groupby(['Emotions', 'Sex']).size().unstack().fillna(0)
        f.write("Распределение эмоций по полу:\n")
        f.write(emotion_sex_distrib.to_string())
        f.write("\n\n")

def save_data(x_train, x_test, y_train, y_test, output_folder):
    file_path_npz = os.path.join(output_folder, 'dataset_splits.npz')
    np.savez(file_path_npz, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    file_path_csv = os.path.join(output_folder, 'features.csv')
    Features.to_csv(file_path_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предобработка данных и извлечение признаков для обучения модели распознавания эмоций по голосу.")
    parser.add_argument("--datasets", nargs='+', required=True, help="Пути к папкам с датасетами (например, --datasets path/to/CREMA-D path/to/RAVDESS).")
    parser.add_argument("--output_folder", type=str, required=True, help="Путь к папке для сохранения результатов.")
    parser.add_argument("--emotions", nargs="+", type=str, help="Список эмоций для включения в анализ.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    df = load_and_preprocess_datasets(args.datasets, args.emotions)
    save_report(df, args.output_folder)
    Features = parallel_feature_extraction(df)
    # x_train, x_test, y_train, y_test = preprocess_features(Features, args.output_folder)
    x_train, x_test, y_train, y_test = preprocess_features_with_stratified_split(Features, args.output_folder, 10)
    save_data(x_train, x_test, y_train, y_test, args.output_folder)
    print(f"Предобработка данных завершена. Данные сохранены в папку '{args.output_folder}'")