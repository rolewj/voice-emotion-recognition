@echo off

REM Пример запуска 1

python datasets_split.py --datasets datasets\CREMA-D datasets\RAVDESS datasets\SAVEE datasets\TESS --count_per_class 10 --output_folder "splited_dataset"
python data_preparation.py --datasets splited_dataset\subset1\CREMA-D splited_dataset\subset1\RAVDESS splited_dataset\subset1\SAVEE splited_dataset\subset1\TESS --output_folder "assets" --emotions Anger Disgust Fear Happy Sad
python model_training.py --data_file "assets\dataset_splits.npz" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python model_evaluating.py --data_folder "splited_dataset\subset2" --model_path "assets\cnn_model.h5" --pickle_path "assets" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python organize_script_outputs.py --input_folder "assets" --output_folder "results"

REM Пример запуска 2

python data_preparation.py --datasets datasets\CREMA-D datasets\RAVDESS datasets\TESS --output_folder "assets" --emotions Anger Disgust Fear Happy Sad
python model_training.py --data_file "assets\dataset_splits.npz" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python model_evaluating.py --data_folder "datasets\SAVEE" --model_path "assets\cnn_model.h5" --pickle_path "assets" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python organize_script_outputs.py --input_folder assets --output_folder results

python data_preparation.py --datasets datasets\CREMA-D datasets\RAVDESS datasets\SAVEE --output_folder "assets" --emotions Anger Disgust Fear Happy Sad
python model_training.py --data_file "assets\dataset_splits.npz" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python model_evaluating.py --data_folder "datasets\TESS" --model_path "assets\cnn_model.h5" --pickle_path "assets" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python organize_script_outputs.py --input_folder assets --output_folder results

python data_preparation.py --datasets datasets\CREMA-D datasets\SAVEE datasets\TESS --output_folder "assets" --emotions Anger Disgust Fear Happy Sad
python model_training.py --data_file "assets\dataset_splits.npz" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python model_evaluating.py --data_folder "datasets\RAVDESS" --model_path "assets\cnn_model.h5" --pickle_path "assets" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python organize_script_outputs.py --input_folder assets --output_folder results

python data_preparation.py --datasets datasets\RAVDESS datasets\SAVEE datasets\TESS --output_folder "assets" --emotions Anger Disgust Fear Happy Sad
python model_training.py --data_file "assets\dataset_splits.npz" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python model_evaluating.py --data_folder "datasets\CREMA-D" --model_path "assets\cnn_model.h5" --pickle_path "assets" --emotions Anger Disgust Fear Happy Sad --output_folder "assets"
python organize_script_outputs.py --input_folder assets --output_folder results