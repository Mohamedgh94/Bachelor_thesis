# Bachelor Thesis: Person and Soft Biometric Identification Using IMU Data

## Overview

This repository contains the code and resources for my Bachelor thesis titled **"Person and Soft Biometric Identification Using IMU Data"**. The project aims to identify individuals based on their unique movement patterns using data from Inertial Measurement Units (IMUs).

In this project, I accomplished the following:

- **Person re-Identification**: Achieved high accuracy in identifying individuals across different datasets (MobiAct, UniMib, and SisFall).
- **Soft-Biometric Identification**: Successfully identified soft-biometric attributes such as age, height, weight, and gender using neural network models.
- **Transfer Learning**: Explored the application of transfer learning to improve model performance and knowledge transfer across different datasets.


## Table of Contents

- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Project Structure

The repository is organized as follows:

- `CNN-LSTM/`: Contains the implementation of the CNN-LSTM model.
- `GatedTransformer/`: Contains the implementation of the Gated Transformer model.
- `Preprocessing/`: Scripts for preprocessing the datasets.
- `Test_Model/`: Includes test scripts for model evaluation.
- `Cross_valid.py`: Script for cross-validation.
- `subjects_check.py`: Script to check subject data integrity.
- `subjects_info.csv`: CSV file containing subject information.
- `README.md`: This file.

## Datasets

The following datasets are used in this project:

- **MobiAct**: A dataset for human activity recognition.
- **UniMib**: A dataset for mobile-based human activity recognition.
- **SisFall**: A dataset focused on fall detection.

## Preprocessing

Each dataset has its own preprocessing script to clean and prepare the data for model training. The preprocessing scripts are located in the `Preprocessing/` directory and include:

- `Mobi_act_preprocessing.py`
- `UniMib_preprocessing.py`
- `sisFall_to_csv.py`
- `sis_Fall_preprocessing.py`

## Models

Several models are implemented and tested in this project, including:

- **CNN-LSTM**: A hybrid model combining Convolutional Neural Networks and Long Short-Term Memory networks.
- **Gated Transformer**: An implementation of the Transformer model with gating mechanisms to enhance performance.


