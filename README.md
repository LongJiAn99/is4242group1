# IS4242 Group 1

# Introduction
The project is **Leveraging Machine Learning Models for Efficient email classification**. This repository includes the codebase for the 5 following sections:
- [Exploratory Data Analysis](data_exploration.ipynb)
- [Logistic Regression Model](LRModel.ipynb)
- [T5-Small Model](train-t5-hf-email.py)
- [BERT Model](BERTModel.ipynb)
- [Naive Bayes Model](NaiveBayesModel.ipynb)

After comparing the 4 different models listed above, we found out that the T5-small model performs best and would be using that model for future unseen data. We have come up with a python script as a demonstration for our overall project.

# Working Demo
1) Download and run [classify-email-chat-mode.py](classify-email-chat-mode.py) python script
2) Input "Your email subject" and "Your email content" respectively
3) Model will output the email's label accordingly, whether it is classified as "Personal", "Marketing" or "Updates"

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
The train-t5-hf-email.py python file contains the code which our team has used to run a T5-small Model on our training dataset.

## Prerequisites
Before using this script, ensure that you have the following python packages installed:
- json
- transformers
- torch
- numpy

## Usage
Run finetune.sh file to start training. You can modify the hyperparameters in the file to seek for better performance.

## Output
Run answer-email.py to use the trained model to generate answers for the testing dataset.

# BERT Model
## Overview
The BERTModel.ipynb file contains the code which our tea has used to train a BERT model on our training dataset.

## Prerequisites
- json
- string
- nltk
- scikit-learn
- transformers
- torch
## Usage
Follow these steps to use the python notebook:

1. Choose an appropriate python kernel.
2. Ensure that the email_training_data.json, email_testing_data.json and emaildata.json files are in the same root directory as the python notebook.
3. Run the notebook sequentially cell by cell as written.

## Output
The python notebook should generate the following:
- Accuracy of the model
- Accuracy of the model with train/test/validation split

# Naive Bayes Model
## Overview
The NaiveBayesModel.ipynb python notebook contains the code which our team has used to run a Naive Bayes Model.

## Prerequisites
Before using this script, ensure that you have the following python packages installed:
- json
- re
- string
- nltk
- scikit-learn

## Usage
Follow these steps to use the python notebook:

1. Choose an appropriate python kernel.
2. Ensure that the email_training_data.json, email_testing_data.json and emaildata.json files are in the same root directory as the python notebook.
3. Run the notebook sequentially cell by cell as written.

## Output
The python notebook should generate the following:
- Accuracy of the model
- Accuracy of the model with train/test/validation split
