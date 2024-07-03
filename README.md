# Bachelor Thesis: Person and Soft Biometric Identification Using IMU Data

## Overview

This repository contains the code and resources for my Bachelor thesis titled **"Person and Soft Biometric Identification Using IMU Data"**. The project aims to identify individuals based on their unique movement patterns using data from Inertial Measurement Units (IMUs).

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

- `GatedTransformer/`: Contains the implementation of the Gated Transformer model.
- `Test_Model/`: Includes test scripts for model evaluation.
- `CNN-LSTM.py`: Implementation of the CNN-LSTM model.
- `Cross_valid.py`: Script for cross-validation.
- `Mobi_act_preprocessing.py`: Preprocessing script for the MobiAct dataset.
- `Network.py`: Network architecture definitions.
- `UniMib_preprocessing.py`: Preprocessing script for the UniMib dataset.
- `sisFall_to_csv.py`: Script to convert SisFall dataset to CSV format.
- `sis_Fall_preprocessing.py`: Preprocessing script for the SisFall dataset.
- `subjects_check.py`: Script to check subject data integrity.
- `subjects_info.csv`: CSV file containing subject information.

## Datasets

The following datasets are used in this project:

- **MobiAct**: A dataset for human activity recognition.
- **UniMib**: A dataset for mobile-based human activity recognition.
- **SisFall**: A dataset focused on fall detection.

## Preprocessing

Each dataset has its own preprocessing script to clean and prepare the data for model training. The preprocessing scripts include:

- `Mobi_act_preprocessing.py`
- `UniMib_preprocessing.py`
- `sisFall_to_csv.py`
- `sis_Fall_preprocessing.py`

## Models

Several models are implemented and tested in this project, including:

- **CNN-LSTM**: A hybrid model combining Convolutional Neural Networks and Long Short-Term Memory networks.
- **Gated Transformer**: An implementation of the Transformer model with gating mechanisms to enhance performance.

