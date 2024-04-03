import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import io
from contextlib import redirect_stdout
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(num_emotions, input_shape):
    cnn = tf.keras.Sequential([
        L.Conv1D(256,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(x_train.shape[1],1)),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3,strides=2,padding='same'),

        L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3,strides=2,padding='same'),
        Dropout(0.2),

        L.Conv1D(128,kernel_size=5,strides=1,padding='same',activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3,strides=2,padding='same'),

        L.Conv1D(128,kernel_size=5,strides=1,padding='same',activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3,strides=2,padding='same'),
        Dropout(0.2),

        L.Conv1D(64,kernel_size=3,strides=1,padding='same',activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3,strides=2,padding='same'),
        Dropout(0.2),

        L.Flatten(),
        L.Dense(256,activation='relu'),
        L.BatchNormalization(),
        L.Dense(num_emotions,activation='softmax')
    ])
    cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
    cnn.summary()
    return cnn

# Функция для захвата вывода .summary() в строку
def capture_model_summary(model):
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        return buf.getvalue()

def train_and_evaluate_model(x_train, y_train, x_test, y_test, emotions, emotion_labels, output_folder):
    cnn = build_model(len(emotions), input_shape=(x_train.shape[1], x_train.shape[2]))
    
    # Захватываем summary модели
    model_summary = capture_model_summary(cnn)
    
    # Callbacks
    model_checkpoint = ModelCheckpoint(os.path.join(output_folder, 'cnn_model.h5'), monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    test_paths_file = os.path.join(output_folder, 'test_paths_with_emotions.txt')
    with open(test_paths_file, 'r') as f:
        test_files = [line.strip().split(',') for line in f]
        
    # Извлечение путей к файлам и меток эмоций из файла test_paths.txt
    _, test_file_emotions = zip(*test_files)

    # Получение уникальных меток эмоций
    emotion_labels = list(set(test_file_emotions))

    history=cnn.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test),
                  batch_size=64, callbacks=[model_checkpoint, early_stop, lr_reduction])

    # Путь к файлу отчета
    report_path = os.path.join(output_folder, 'model_training_report.pdf')

    # Генерация отчета классификации
    pred_test = cnn.predict(x_test)
    pred_test_labels = np.argmax(pred_test, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Вычисление количества правильно угаданных эмоций и общей точности
    num_correct = np.sum(pred_test_labels == y_test_labels)
    total_samples = len(y_test_labels)
    accuracy = num_correct / total_samples

    classification_report_str = classification_report(y_test_labels, pred_test_labels, target_names=emotions, digits = 4, output_dict=False)
    
    # Запись распределения вероятностей по эмоциям для каждого файла
    output_path_txt = os.path.join(output_folder, 'test_predictions_report.txt')
    with open(output_path_txt, 'w') as file:
        # Создание строки с информацией о количестве эмоций, правильно угаданных эмоциях и точности
        info_str = "General information about the test set:\n"
        info_str += f"Number of different emotions : {len(emotions)}\n"
        info_str += f"Number of correctly predicted emotions: {num_correct} out of {total_samples}\n"
        info_str += f"Overall accuracy: {accuracy * 100:.4f}%\n"
        file.write(f"{info_str}\n")
        
        # Запись полного отчета classification_report
        file.write(f"Detailed Classification Report for the test set:\n{classification_report_str}\n\n")
        
        file.write("Classification results for emotion recognition on each file in the test set:\n")
        for i in range(len(x_test)):
            file_path, emotion = test_files[i]
            predicted_emotion_index = np.argmax(pred_test[i])
            predicted_emotion = emotion_labels[predicted_emotion_index]
            
            file.write(f"File path: {file_path}\n")
            file.write(f"True Emotion: {emotion}\n")
            file.write(f"Predicted Emotion: {predicted_emotion}\n")
            file.write("Probabilities:\n")
            
            for emotion, prob in zip(emotion_labels, pred_test[i]):
                file.write(f"{emotion}: {prob:.4f}\n")
            
            file.write("\n")

    # Сохранение всего отчета в один PDF файл
    with PdfPages(report_path) as pdf:        
        confusion_matrix_arr = confusion_matrix(y_test_labels, pred_test_labels)
        
        confusion_matrix_df = pd.DataFrame(confusion_matrix_arr, index=emotions, columns=emotions)
        
        fig, ax = plt.subplots(figsize=(8, 11))
        text = fig.text(0.05, 0.95, model_summary, ha='left', va='top', fontsize=8, family='monospace')
        ax.axis('off')
        pdf.savefig(fig)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 11))
        text = fig.text(0.05, 0.95, classification_report_str, ha='left', va='top', fontsize=10, family='monospace')
        ax.axis('off')
        pdf.savefig(fig)
        plt.close()
        
        # Графики обучения
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Сохранение матрицы ошибок
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues")
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        pdf.savefig(fig)
        plt.close()

    print(f"Отчет сохранен в '{report_path}'")

def load_data(file_path):
    data = np.load(file_path)
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тренировка и оценка CNN модели для распознавания эмоций по голосу.")
    parser.add_argument("--data_file", type=str, required=True, help="Путь к файлу с обучающими и тестовыми данными.")
    parser.add_argument("--emotions", nargs='+', required=True, help="Список эмоций для распознавания.")
    parser.add_argument("--output_folder", type=str, required=True, help="Путь к папке для сохранения обученной модели и отчетов.")
    
    args = parser.parse_args()

    # Загрузка данных
    x_train, x_test, y_train, y_test = load_data(args.data_file)

    # Убедитесь, что папка для вывода существует
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Тренировка и оценка модели
    train_and_evaluate_model(x_train, y_train, x_test, y_test, args.emotions, args.output_folder, args.output_folder)