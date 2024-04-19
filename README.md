# IS4242 Group 1

# Introduction
The project is **Leveraging Machine Learning Models for Efficient email classification**. This repository includes the codebase for the 5 following sections:
- [Exploratory Data Analysis](data_exploration.ipynb)
- [Logistic Regression Model](LRModel.ipynb)
- [T5-Small Model](train-t5-hf-email.py)
- [BERT Model](BERTModel.ipynb)
- [Naive Bayes Model](NaiveBayesModel.ipynb)

# Exploratory Data Analysis
## Overview
The data_exploration.ipynb python notebook contains the code which our team has used to generate the descriptive statistics and data visualisations used in our EDA.

## Prerequisites
Before using this script, ensure that you have the following python packages installed:
- pandas
- json
- seaborn
- matplotlib

## Usage
Follow these steps to use the python notebook:

1. Choose an appropriate python kernel.
2. Ensure that the email_training_data.json, email_testing_data.json and emaildata.json files are in the same root directory as the python notebook.
3. Run the notebook sequentially cell by cell as written.

## Output
The python notebook should generate the following:
- Data types of each column
- Number of null values for each column
- Descriptive statistics
- Pie chart of distribution of data
- Bar plots of distribution of email length and word count
- Coefficients of email length and word count features after fitting a Logistic Regression Model

# Logistic Regression Model
## Overview
The LRModel.ipynb python notebook contains the code which our team has used to run a Logistic Regression Model using a bags-of-words approach to classify the emails.

## Prerequisites
Before using this script, ensure that you have the following python packages installed:
- pandas
- json
- scikit-learn
- numpy

## Usage
Follow these steps to use the python notebook:

1. Choose an appropriate python kernel.
2. Ensure that the email_training_data.json, email_testing_data.json and emaildata.json files are in the same root directory as the python notebook.
3. Run the notebook sequentially cell by cell as written.

## Output
The python notebook should generate the following:
- Accuracy of the model using CountVectorizer
- Accuracy of the model using TFID
- Accuracy of the model using CountVectorizer and text length and word count as features

# T5-Small Model
## Overview

## Prerequisites

## Usage

## Output

# BERT Model
## Overview

## Prerequisites

## Usage

## Output

# Naive Bayes Model
## Overview

## Prerequisites

## Usage

## Output
