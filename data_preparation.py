import librosa
import numpy as np
import pandas as pd
import os
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


emotions_mapping = {
  'ANG': 'Anger',
  'DIS': 'Disgust',
  'FEA': 'Fear',
  'HAP': 'Happy',
  'NEU': 'Neutral',
  'SAD': 'Sad',
}

emotion_level_mapping = {
  'LO': 'Low',
  'MD': 'Medium',
  'HI': 'High',
  'XX': 'Unspecified'
}

female_id = [ '1002', '1003', '1004', '1006', '1007', '1008', '1009',
              '1010', '1012', '1013', '1018', '1020', '1021', '1024',
              '1025', '1028', '1029', '1030', '1037', '1043', '1046',
              '1047', '1049', '1052', '1053', '1054', '1055', '1056',
              '1058', '1060', '1061', '1063', '1072', '1073', '1074',
              '1075', '1076', '1078', '1079', '1082', '1084', '1089',
              '1091', ]

def load_and_preprocess_data(audios_folder):
    audios = glob(f"{audios_folder}/*.wav")
    path_list, emotion_list, emotion_level_list, sex_list = [], [], [], []

    for file in audios:
        path_list.append(file)
        filename = os.path.basename(file)
        part = filename.split('_')
        emotion = emotions_mapping.get(part[2], 'Unknown')
        emotion_level = emotion_level_mapping.get(part[-1].replace('.wav', ''), 'Unknown')
        sex = 'Female' if part[0] in female_id else 'Male'
        emotion_list.append(emotion)
        emotion_level_list.append(emotion_level)
        sex_list.append(sex)

    crema_df = pd.concat([
        pd.DataFrame({'Path': path_list}),
        pd.DataFrame({'Emotions': emotion_list}),
        pd.DataFrame({'Emotion levels': emotion_level_list}),
        pd.DataFrame({'Sex': sex_list}),
    ], axis=1)
    
    target_file = '1040_ITH_SAD_X.wav'
    file_exists = crema_df['Path'].str.contains(target_file).any()
    if file_exists:
        index_to_modify = crema_df[crema_df['Path'].str.contains(target_file)].index[0]
        crema_df.loc[index_to_modify, 'Emotion levels'] = 'Unspecified'
    else:
        print(f"Файл {target_file} не найден в датасете.")
    
    return crema_df

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
    return X, Y

def parallel_feature_extraction(crema_df):
    paths = crema_df['Path'].tolist()
    emotions = crema_df['Emotions'].tolist()

    # Параллельная обработка
    results = Parallel(n_jobs=-1, verbose=2)(delayed(process_feature)(path, emotion) for path, emotion in zip(paths, emotions))

    # Сбор результатов
    X = []
    Y = []
    for result in results:
        x, y = result
        X.extend(x)
        Y.extend(y)

    Features = pd.DataFrame(X)
    Features['labels'] = Y

    return Features

# def preprocess_features(Features):
#     Features = Features.fillna(0)

#     grouped = Features.groupby('labels')
    
#     x_test_list = []
#     y_test_list = []
#     x_train_list = []
#     y_train_list = []

#     for name, group in grouped:
#         X_group = group.iloc[:, :-1].values
#         Y_group = group['labels'].values.reshape(-1, 1)

#         # Использование train_test_split для разделения каждой группы на тестовую и обучающую выборки
#         X_temp_train, X_temp_test, Y_temp_train, Y_temp_test = train_test_split(X_group, Y_group, test_size=10, random_state=42, shuffle=True)
        
#         x_test_list.append(X_temp_test)
#         y_test_list.append(Y_temp_test)
#         x_train_list.append(X_temp_train)
#         y_train_list.append(Y_temp_train)

#     # Объединение данных из списков в один массив
#     x_test = np.vstack(x_test_list)
#     y_test = np.vstack(y_test_list)
#     x_train = np.vstack(x_train_list)
#     y_train = np.vstack(y_train_list)

#     # Преобразование меток с OneHotEncoder
#     encoder = OneHotEncoder()
#     y_train = encoder.fit_transform(y_train).toarray()
#     y_test = encoder.transform(y_test).toarray()

#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)

#     x_train = np.expand_dims(x_train, axis=2)
#     x_test = np.expand_dims(x_test, axis=2)

#     return x_train, x_test, y_train, y_test

def preprocess_features(Features, output_folder):
    Features = Features.fillna(0)

    X = Features.iloc[:, :-1].values
    Y = Features['labels'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    
    encoder_file_path = os.path.join(output_folder, 'encoder.pickle')
    with open(encoder_file_path, 'wb') as f:
        pickle.dump(encoder, f)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    scaler_file_path = os.path.join(output_folder, 'scaler.pickle')
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("Encoder и Scaler успешно загружены.")
    print("Тип объекта encoder:", type(encoder))
    print("Тип объекта scaler:", type(scaler))

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    return x_train, x_test, y_train, y_test

def save_data(x_train, x_test, y_train, y_test, output_folder):
    file_path_npz = os.path.join(output_folder, 'dataset_splits.npz')
    np.savez(file_path_npz, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    file_path_csv = os.path.join(output_folder, 'features.csv')
    Features.to_csv(file_path_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предобработка данных и извлечение признаков для обучения модели распознавания эмоций по голосу.")
    parser.add_argument("--data_folder", type=str, required=True, help="Путь к папке с аудиофайлами.")
    parser.add_argument("--output_folder", type=str, required=True, help="Путь к папке для сохранения результатов.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Используем обновленную функцию с параллельной обработкой для извлечения признаков
    crema_df = load_and_preprocess_data(args.data_folder)
    Features = parallel_feature_extraction(crema_df)
    x_train, x_test, y_train, y_test = preprocess_features(Features, args.output_folder)
    save_data(x_train, x_test, y_train, y_test, args.output_folder)
    print(f"Предобработка данных завершена. Данные сохранены в папку '{args.output_folder}'")