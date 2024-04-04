#!/bin/bash

python data_preparation.py --datasets datasets\CREMA-D datasets\RAVDESS --output_folder "assets" --emotions Anger Disgust Fear Happy Neutral Sad
python model_training.py --data_file "assets\dataset_splits.npz" --emotions Anger Disgust Fear Happy Neutral Sad --output_folder "assets"
python model_evaluating.py --data_folder "datasets\RAVDESS\Actor_05" --model_path "assets\cnn_model.h5" --pickle_path "assets" --emotions Anger Disgust Fear Happy Neutral Sad --output_folder "assets"