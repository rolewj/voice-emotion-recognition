import numpy as np
import pandas as pd
import librosa
import argparse
import pickle
from tensorflow.keras.models import load_model
import glob
import os

emotions_mapping = {
    'ANG': 'Anger', 'DIS': 'Disgust', 'FEA': 'Fear',
    'HAP': 'Happy', 'NEU': 'Neutral', 'SAD': 'Sad'
}

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Функция для загрузки и предобработки аудиофайлов из датасета
def load_samples(data_folder, num_samples_per_emotion=10, random_state=42):
    all_files = glob(os.path.join(data_folder, "*.wav"))
    samples = {emotion: [] for emotion in emotions_mapping.values()}
    
    np.random.seed(random_state)
    np.random.shuffle(all_files)
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        emotion_short = parts[2] if len(parts) > 2 else None
        emotion = emotions_mapping.get(emotion_short)
        if emotion and len(samples[emotion]) < num_samples_per_emotion:
            samples[emotion].append(file_path)
        
        if all(len(files) == num_samples_per_emotion for files in samples.values()):
            break
    
    return samples

def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfccs(data, sr, n_mfcc=13, frame_length=2048, hop_length=512, flatten: bool = True):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=frame_length)
    return np.squeeze(mfccs.T) if not flatten else np.ravel(mfccs.T)

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    features = np.array([])

    features = np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfccs(data, sr, frame_length=frame_length, hop_length=hop_length)
    ))
    return features

def predict_emotion_from_file(file_path, scaler2):
    data, _ = librosa.load(file_path, sr=22050, duration=2.5, offset=0.6)
    raw_features = extract_features(data)
    print("Размер признаков до reshape:", raw_features.shape)
    # Проверяем, если количество признаков меньше 1620, то делаем padding нулями
    if raw_features.size < 1620:
        # Вычисляем количество нулей для добавления
        padding = 1620 - raw_features.size
        # Добавляем нули с конца массива
        features_padded = np.pad(raw_features, (0, padding), 'constant')
    else:
        features_padded = raw_features[:1620]
    
    # Преобразуем в 2D массив
    features_reshaped = features_padded.reshape(1, -1)
    print("Размер признаков после reshape:", features_reshaped.shape)
    scaled_features  = scaler2.transform(features_reshaped)
    model_input_features = np.expand_dims(scaled_features, axis=2)
    print("Формат входных данных для модели:", model_input_features.shape)
    return model_input_features

def predict_and_report(model, samples, output_path, emotion_labels, scaler2, encoder2, num_samples_per_emotion=10):
    reports = []
    emotion_counts = {emotion: 0 for emotion in emotion_labels}
    
    for emotion, files in samples.items():
        for file_path in files[:num_samples_per_emotion]:
            model_input_features = predict_emotion_from_file(file_path, scaler2)
            predictions = model.predict(model_input_features)
            predicted_emotion_index = np.argmax(predictions, axis=1)
            # Преобразование индексов в one-hot векторы
            num_classes = len(emotion_labels)
            one_hot_predictions = np.zeros((predictions.shape[0], num_classes))
            one_hot_predictions[np.arange(predictions.shape[0]), predicted_emotion_index] = 1
            # Получение исходных меток классов
            predicted_emotion_labels = encoder2.inverse_transform(one_hot_predictions)
            predicted_emotion = predicted_emotion_labels[0][0]
            emotion_counts[predicted_emotion] += 1

            report = f"File: {file_path}\nPredicted Emotion: {predicted_emotion}\nProbabilities:\n"
            class_probabilities  = predictions.flatten()  # Преобразуем предсказания для удобства отчета
            for label, prob in zip(emotion_labels, class_probabilities):
                report += f"{label}: {prob:.4f}\n"
            report += "\n"
            reports.append(report)
    
    # Затем записываем всё в файл отчёта
    output_path_txt = os.path.join(output_path, 'model_evaluating_report.txt')
    with open(output_path_txt, 'w') as report_file:
        # Сначала записываем подсчёты эмоций
        report_file.write("Emotion Counts:\n")
        for emotion, count in emotion_counts.items():
            report_file.write(f"{emotion}: {count}\n")
        report_file.write("\nDetailed Predictions:\n\n")
        
        # После подсчётов записываем детализированные предсказания
        for report in reports:
            report_file.write(report)


def predict_and_report_modified(model, path, output_path, emotion_labels, scaler2, encoder2):
    reports = []
    emotion_counts = {emotion: 0 for emotion in emotion_labels}
    
    # Проверяем, является ли путь директорией
    if os.path.isdir(path):
        # Если это директория, обходим все файлы в ней и её поддиректориях
        file_paths = glob.glob(os.path.join(path, '**/*.wav'), recursive=True)
    else:
        # Если это файл, обрабатываем его как одиночный файл
        file_paths = [path]
    
    for file_path in file_paths:
        model_input_features = predict_emotion_from_file(file_path, scaler2)
        predictions = model.predict(model_input_features)
        predicted_emotion_index = np.argmax(predictions, axis=1)
        # Преобразование индексов в one-hot векторы
        num_classes = len(emotion_labels)
        one_hot_predictions = np.zeros((predictions.shape[0], num_classes))
        one_hot_predictions[np.arange(predictions.shape[0]), predicted_emotion_index] = 1
        # Получение исходных меток классов
        predicted_emotion_labels = encoder2.inverse_transform(one_hot_predictions)
        predicted_emotion = predicted_emotion_labels[0][0]
        emotion_counts[predicted_emotion] += 1

        report = f"File: {file_path}\nPredicted Emotion: {predicted_emotion}\nProbabilities:\n"
        class_probabilities = predictions.flatten()  # Преобразуем предсказания для удобства отчета
        for label, prob in zip(emotion_labels, class_probabilities):
            report += f"{label}: {prob:.4f}\n"
        report += "\n"
        reports.append(report)
        
        # Затем записываем всё в файл отчёта
        output_path_txt = os.path.join(output_path, 'model_evaluating_report.txt')
        with open(output_path_txt, 'w') as report_file:
            # Сначала записываем подсчёты эмоций
            report_file.write("Emotion Counts:\n")
            for emotion, count in emotion_counts.items():
                report_file.write(f"{emotion}: {count}\n")
            report_file.write("\nDetailed Predictions:\n\n")
            
            # После подсчётов записываем детализированные предсказания
            for report in reports:
                report_file.write(report)


def load_pickle(pickle_path):
    scaler_file_path = os.path.join(pickle_path, 'scaler.pickle')
    encoder_file_path = os.path.join(pickle_path, 'encoder.pickle')
    with open(scaler_file_path, 'rb') as f:
        scaler2 = pickle.load(f)
    with open(encoder_file_path, 'rb') as f:
        encoder2 = pickle.load(f)
    print("Файлы pickle успешно загружены.")
    return scaler2, encoder2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распознавание эмоций из аудиофайлов и генерация отчета")
    parser.add_argument("--data_folder", type=str, required=True, help="Путь к папке с датасетом.")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к обученной .h5 модели.")
    parser.add_argument("--pickle_path", type=str, required=True, help="Путь к .pickle файлам.")
    parser.add_argument("--output_folder", type=str, required=True, help="Путь к папке для сохранения результатов.")
    
    args = parser.parse_args()
    
    # Загрузка модели
    model = load_model(args.model_path)
    
    # Загрузка и предобработка данных
    # samples = load_samples(args.data_folder)
    
    # Загрузка pickle файлов
    scaler2, encoder2 = load_pickle(args.pickle_path)
    
    # Выполнение предсказаний и генерация отчета
    # predict_and_report(model, samples, args.output_folder, emotion_labels, scaler2, encoder2)
    
    predict_and_report_modified(model, args.data_folder, args.output_folder, emotion_labels, scaler2, encoder2)
    
    print(f"Оценка прошла успешно. Данные сохранены в папку '{args.output_folder}'")
