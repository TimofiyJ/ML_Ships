# ML_Ships (Ship Image Segmentation)

## Project Overview

The project consists of the following files:

- **.gitignore**: Git ignore file.
- **data-analysis.ipynb**: Jupyter Notebook containing Exploratory Data Analysis (EDA), data preparation, model pretraining, and experimentation. Key discoveries and explanations are documented in this file.
- **model-training.py**: Script for preparing the dataset and training the model based on the findings from data analysis.
- **model-testing.py**: Script for testing the trained model.
- **inference.py**: Script for making predictions. Creates a folder named "results" to store the model predictions.
- **requirements.txt**: List of project dependencies.

## Requirements

Recommended Python version: 3.11

## Usage

To run the inference script, use the following command:

```bash
py path-to-inference.py path-to-folder-with-images
