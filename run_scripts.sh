#!/bin/bash

python data_preparation.py --data_folder "datasets\CREMA-D\AudioWAV" --output_folder "assets"
python model_training.py --data_file "assets\dataset_splits.npz" --output_folder "assets"
python model_evaluating.py --data_folder "datasets\RAVDESS\Actor_19" --model_path "assets\cnn_model.h5" --pickle_path "assets" --output_folder "assets"