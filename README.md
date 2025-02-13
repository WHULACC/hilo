## Overview

This project contains the code for the paper **Multimodal Emotion-Cause Pair Extraction with Holistic Interaction and Label Constraint** published in ACM Transactions on Multimedia (ToMM).


## Dependencies

This project is based on PyTorch and Transformers. 
You can create the conda environment using the following command:
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate hilo 
```

## Configuration

The configurations for the model and training process are stored in `src/config.yaml`. You can modify this file to adjust the settings.

## Data
The dataset is located in data/dataset. Please follow the instructions in [data/dataset/README.md](data/dataset/README.md) to download the audio and video features, and place them in the data/dataset directory.


## Usage
You can run the following command to train `&` evaluate the model:  
`python main.py`