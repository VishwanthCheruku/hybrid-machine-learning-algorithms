# hybrid-machine-learning-algorithms

Dataset Overview

This dataset contains information used for the simulation and analysis of various machine learning models, including both classification and regression tasks. The dataset has been used to train and evaluate models like Ridge Regression, XGBoost, LightGBM, Random Forest, k-Nearest Neighbors (kNN), Support Vector Machine (SVM), Gradient Boosting, and Linear Regression. The dataset includes features related to academic and employer reputation, faculty-to-student ratios, citations per faculty, international faculty and student metrics, research network scores, and sustainability scores, among others.

File Descriptions

QS world University ranking dataset.csv: This is the primary dataset file used for training and testing the machine learning models. It includes the following columns:

Academic Reputation Score

Academic Reputation Rank

Employer Reputation Score

Employer Reputation Rank

Faculty Student Score

Faculty Student Rank

Citations per Faculty Score

Citations per Faculty Rank

International Faculty Score

International Faculty Rank

International Students Score

International Students Rank

International Research Network Score

International Research Network Rank

Employment Outcomes Score

Employment Outcomes Rank

Sustainability Score

Sustainability Rank

Overall SCORE: The target variable for regression tasks.

Abbreviations and Codes

RMSE: Root Mean Squared Error

MSE: Mean Squared Error

R2 Score: Coefficient of Determination

PSO: Particle Swarm Optimization

LSTM: Long Short-Term Memory (a type of neural network model)

SVM: Support Vector Machine

kNN: k-Nearest Neighbors

XGBoost: Extreme Gradient Boosting

LightGBM: Light Gradient Boosting Machine

Software and Dependencies

The following software and libraries are required to run the models and analyses described in this submission:

Programming Language and Frameworks

Python: The primary programming language used for this project.

Frameworks and Libraries:

scikit-learn

xgboost

lightgbm

tensorflow

numpy

pandas

pyswarm (for PSO)

Dependencies

Please ensure the following dependencies are installed:

scikit-learn==0.24.2

xgboost==1.5.0

lightgbm==3.2.1

tensorflow==2.7.0

numpy==1.21.2

pandas==1.3.3

pyswarm==0.6.1

Evaluation Metrics

Metrics Used

For classification models: Accuracy, Precision, Recall, F1 Score

For regression models: RMSE, MSE, R2 Score

Experimental Setup

Hardware Configurations:

Minimum of 4GB RAM

Minimum of 2.26GHz Processor

Minimum of 512MB storage

Cloud Environment:

Experiments were conducted using Google Collaboratory and Jupyter Notebook by Anaconda.

Usage Instructions

Load the dataset: Use pandas to load the dataset (QS world University ranking dataset.csv) into a DataFrame.

Preprocess the data: Ensure the data is clean and properly encoded.

Split the data: Use train-test split to separate the data into training and testing sets.

Train the model: Select the desired machine learning model and train it using the training data.

Evaluate the model: Use the appropriate evaluation metrics (RMSE, MSE, R2 Score) to assess model performance.

Optimize the model: Optional, but hyperparameter optimization (e.g., using PSO) can be conducted to improve model performance.

Contact Information

For any questions or issues related to this dataset or the accompanying code, please contact [Ch. Vishwanth Kumar Goud] at [cherukuvishwanth@gmail.com].

This README file is intended to provide all necessary details for understanding, interpreting, and reusing the dataset and related code.


