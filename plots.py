# -*- coding: utf-8 -*-
"""plots.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xsm4WI6WlNIlWGvJgYewRb6rPZeTF0c5
"""

import matplotlib.pyplot as plt

# Define the algorithms and their corresponding values
algorithms = [
    "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"
]

rmse = [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883]
mae = [2.489116327921549,
1.985846444805863,
1.8018765635369778,
2.066873620923074,
1.9385199938456235,
2.0153390069011166,
2.1147864603241566
]
r2_score = [0.12143217982921384,
0.2847185188944267,
0.2809791387411822,
0.2611454864319901,
0.3222017264660062,
0.2791985462358334,
0.23686713600258014
]
accuracy = [89.06,
90.13,
90.11,
89.97,
100.0,
100.0,
100.0,
]

# Create scatter plots for each metric
plt.figure(figsize=(12, 8))

# RMSE Scatter Plot
plt.subplot(2, 2, 1)
plt.scatter(algorithms, rmse, label="RMSE", color="blue")
plt.title("RMSE Scatter Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

# MAE Scatter Plot
plt.subplot(2, 2, 2)
plt.scatter(algorithms, mae, label="MAE", color="green")
plt.title("MAE Scatter Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

# R2 Score Scatter Plot
plt.subplot(2, 2, 3)
plt.scatter(algorithms, r2_score, label="R2 Score", color="red")
plt.title("R2 Score Scatter Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

# Accuracy Scatter Plot
plt.subplot(2, 2, 4)
plt.scatter(algorithms, accuracy, label="Accuracy", color="purple")
plt.title("Accuracy Scatter Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

algorithms = [
     "Hybrid Model-4",
    "Hybrid Model-5",
    "Hybrid Model-6",
    "Hybrid Model-7"

]

rmse = [3.64,
3.29,
3.30,
3.34
]
mae = [2.48,
1.98,
1.80,
2.06
]
r2_score = [0.12,
0.28,
0.28,
0.26
]
accuracy = [89.06,
90.13,
90.11,
89.97
]

x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for RMSE, MAE, R2 Score, and Accuracy
bar1 = ax.bar(x - 1.5 * width, rmse, width, label='RMSE', color="b")
bar2 = ax.bar(x - 0.5 * width, mae, width, label='MAE', color="g")
bar3 = ax.bar(x + 0.5 * width, r2_score, width, label='R2 Score', color="r")
bar4 = ax.bar(x + 1.5 * width, accuracy, width, label='Accuracy', color="c")

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics for Different Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.grid(axis="y", linestyle="--")
plt.show()

# Create a line plot for all metrics
plt.figure(figsize=(10, 8))
plt.plot(algorithms, rmse, marker='o', label='RMSE', color="blue", linestyle='-', linewidth=2)
plt.plot(algorithms, mae, marker='o', label='MAE', color="green", linestyle='-', linewidth=2)
plt.plot(algorithms, r2_score, marker='o', label='R2 Score', color="red", linestyle='-', linewidth=2)
plt.plot(algorithms, accuracy, marker='o', label='Accuracy', color="purple", linestyle='-', linewidth=2)
plt.title("Performance Metrics Line Plot")
plt.xticks(rotation=45)
plt.xlabel("Algorithms")
plt.ylabel("Metrics Value")
plt.legend(loc='best')
plt.grid(axis="y", linestyle="--")
plt.show()

# Create a box plot for RMSE, MAE, R2 Score, and Accuracy
data = [rmse, mae, r2_score, accuracy]
labels = ["RMSE", "MAE", "R2 Score", "Accuracy"]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels, patch_artist=True)
plt.title("Performance Metrics Box Plot")
plt.ylabel("Value")
plt.grid(axis="y", linestyle="--")
plt.show()

import matplotlib.pyplot as plt

# Data
algorithms = [
     "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"
]

rmse = [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883]
mae = [2.489116327921549,
1.985846444805863,
1.8018765635369778,
2.066873620923074,
1.9385199938456235,
2.0153390069011166,
2.1147864603241566
]
r2_score = [0.12143217982921384,
0.2847185188944267,
0.2809791387411822,
0.2611454864319901,
0.3222017264660062,
0.2791985462358334,
0.23686713600258014
]
accuracy = [89.06,
90.13,
90.11,
89.97,
100.0,
100.0,
100.0,
]

# Create a pie chart
plt.figure(figsize=(12, 8))
plt.pie([sum(rmse), sum(r2_score), sum(accuracy)], labels=['RMSE', 'R2 Score', 'Accuracy'], autopct='%1.1f%%', startangle=140)
plt.title("Combined Metrics Pie Chart")
plt.axis('equal')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with your data
data = {
    "Algorithm": [
         "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"
    ],
    "RMSE": [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883],
    "MAE": [2.489116327921549,
1.985846444805863,
1.8018765635369778,
2.066873620923074,
1.9385199938456235,
2.0153390069011166,
2.1147864603241566],
    "R2 Score": [0.12143217982921384,
0.2847185188944267,
0.2809791387411822,
0.2611454864319901,
0.3222017264660062,
0.2791985462358334,
0.23686713600258014],
    "Accuracy": [89.06,
90.13,
90.11,
89.97,
100.0,
100.0,
100.0,]
}

df = pd.DataFrame(data)

# Create a correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Plot of Metrics")
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with your data
data = {
    "Algorithm": [
         "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"
    ],
    "RMSE": [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883],
    "MAE": [2.489116327921549,
1.985846444805863,
1.8018765635369778,
2.066873620923074,
1.9385199938456235,
2.0153390069011166,
2.1147864603241566],
    "R2 Score": [0.12143217982921384,
0.2847185188944267,
0.2809791387411822,
0.2611454864319901,
0.3222017264660062,
0.2791985462358334,
0.23686713600258014],
    "Accuracy": [89.06,
90.13,
90.11,
89.97,
100.0,
100.0,
100.0,]
}

df = pd.DataFrame(data)

# Create a KDE plot for RMSE
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="RMSE", hue="Algorithm", common_norm=False)
plt.title("Kernel Density Estimation (KDE) Plot of RMSE")
plt.xlabel("RMSE")
plt.ylabel("Density")
plt.show()

# Create a KDE plot for MAE
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="MAE", hue="Algorithm", common_norm=False)
plt.title("Kernel Density Estimation (KDE) Plot of MAE")
plt.xlabel("MAE")
plt.ylabel("Density")
plt.show()

# Create a KDE plot for R2 Score
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="R2 Score", hue="Algorithm", common_norm=False)
plt.title("Kernel Density Estimation (KDE) Plot of R2 Score")
plt.xlabel("R2 Score")
plt.ylabel("Density")
plt.show()

# Create a KDE plot for Accuracy
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="Accuracy", hue="Algorithm", common_norm=False)
plt.title("Kernel Density Estimation (KDE) Plot of Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Density")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with your data
data = {
    'Algorithm': [ "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"],
    'Metric': ['RMSE', 'RMSE', 'RMSE', 'RMSE'],
    'Value': [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883]
}

# Create a categorical scatter plot for RMSE
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.swarmplot(x="Algorithm", y="Value", hue="Metric", data=data, palette="Set2", dodge=True, size=10)
plt.title("Categorical Scatter Plot of RMSE")
plt.xlabel("Algorithm")
plt.ylabel("Value (RMSE)")
plt.legend(title="Metric", loc="upper right")
plt.grid(axis="y")
plt.xticks(rotation=15)

# Create a DataFrame with MAE data
data_mae = {
    'Algorithm': [ "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"],
    'Metric': ['MAE', 'MAE', 'MAE', 'MAE'],
    'Value': [2.489116327921549,
1.985846444805863,
1.8018765635369778,
2.066873620923074,
1.9385199938456235,
2.0153390069011166,
2.1147864603241566]
}

# Create a categorical scatter plot for MAE
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.swarmplot(x="Algorithm", y="Value", hue="Metric", data=data_mae, palette="Set2", dodge=True, size=10)
plt.title("Categorical Scatter Plot of MAE")
plt.xlabel("Algorithm")
plt.ylabel("Value (MAE)")
plt.legend(title="Metric", loc="upper right")
plt.grid(axis="y")
plt.xticks(rotation=15)

# Create a DataFrame with R2 Score data
data_r2 = {
    'Algorithm': [ "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"],
    'Metric': ['R2 Score', 'R2 Score', 'R2 Score', 'R2 Score'],
    'Value': [0.12143217982921384,
0.2847185188944267,
0.2809791387411822,
0.2611454864319901,
0.3222017264660062,
0.2791985462358334,
0.23686713600258014]
}

# Create a categorical scatter plot for R2 Score
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.swarmplot(x="Algorithm", y="Value", hue="Metric", data=data_r2, palette="Set2", dodge=True, size=10)
plt.title("Categorical Scatter Plot of R2 Score")
plt.xlabel("Algorithm")
plt.ylabel("Value (R2 Score)")
plt.legend(title="Metric", loc="upper right")
plt.grid(axis="y")
plt.xticks(rotation=15)

# Create a DataFrame with Accuracy data
data_accuracy = {
    'Algorithm': [ "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"],
    'Metric': ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy'],
    'Value': [89.06,
90.13,
90.11,
89.97,
100.0,
100.0,
100.0,]
}

# Create a categorical scatter plot for Accuracy
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.swarmplot(x="Algorithm", y="Value", hue="Metric", data=data_accuracy, palette="Set2", dodge=True, size=10)
plt.title("Categorical Scatter Plot of Accuracy")
plt.xlabel("Algorithm")
plt.ylabel("Value (Accuracy)")
plt.legend(title="Metric", loc="upper right")
plt.grid(axis="y")
plt.xticks(rotation=15)

# Show all the plots
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a dataframe with your data
data = {
    'Algorithm': [ "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"],
     "RMSE": [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883],
    "MAE": [2.489116327921549,
1.985846444805863,
1.8018765635369778,
2.066873620923074,
1.9385199938456235,
2.0153390069011166,
2.1147864603241566],
    "R2 Score": [0.12143217982921384,
0.2847185188944267,
0.2809791387411822,
0.2611454864319901,
0.3222017264660062,
0.2791985462358334,
0.23686713600258014],
    "Accuracy": [89.06,
90.13,
90.11,
89.97,
100.0,
100.0,
100.0,]
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to have all metrics in one column
df_melted = pd.melt(df, id_vars=['Algorithm'], var_name='Metric', value_name='Value')

# Create a categorical scatterplot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.stripplot(x="Algorithm", y="Value", hue="Metric", data=df_melted, size=10, jitter=True, dodge=True, palette="Set2")

# Customize the appearance
plt.title("Categorical Scatterplots of Metrics")
plt.xticks(rotation=90)
plt.xlabel("Algorithm")
plt.ylabel("Value")

# Show the plot
plt.legend(title="Metric")
plt.show()

import matplotlib.pyplot as plt

# Define the algorithms and their corresponding values
algorithms = [
    "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"
]

rmse = [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883]
mae = [2.489116327921549,
       1.985846444805863,
       1.8018765635369778,
       2.066873620923074,
       1.9385199938456235,
       2.0153390069011166,
       2.1147864603241566
       ]
r2_score = [0.12143217982921384,
            0.2847185188944267,
            0.2809791387411822,
            0.2611454864319901,
            0.3222017264660062,
            0.2791985462358334,
            0.23686713600258014
            ]
accuracy = [89.06, 90.13, 90.11, 89.97, 100.0, 100.0, 100.0]

# Create bar plots for each metric
plt.figure(figsize=(12, 8))

# RMSE Bar Plot
plt.subplot(2, 2, 1)
plt.bar(algorithms, rmse, color="blue")
plt.title("RMSE Bar Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

# MAE Bar Plot
plt.subplot(2, 2, 2)
plt.bar(algorithms, mae, color="green")
plt.title("MAE Bar Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

# R2 Score Bar Plot
plt.subplot(2, 2, 3)
plt.bar(algorithms, r2_score, color="red")
plt.title("R2 Score Bar Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

# Accuracy Bar Plot
plt.subplot(2, 2, 4)
plt.bar(algorithms, accuracy, color="purple")
plt.title("Accuracy Bar Plot")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the algorithms and their corresponding values
algorithms = [
    "XGBoost and Neural Network",
    "Gradient Boosting and k-Nearest Neighbors",
    "Random Forest and SVM",
    "Support Vector Machine (SVM), Neural Network and Gradient Boosting model",
    "RandomForest and XGBoost",
    "LinearRegression and RandomForest",
    "LinearRegression and XGBoost"
]

rmse = [3.649015459878686, 3.29250579412059, 3.301100928942298, 3.3463204130358974, 3.205075662974626, 3.305185845912358, 3.4008551570769883]
mae = [2.489116327921549, 1.985846444805863, 1.8018765635369778, 2.066873620923074, 1.9385199938456235, 2.0153390069011166, 2.1147864603241566]
r2_score = [0.12143217982921384, 0.2847185188944267, 0.2809791387411822, 0.2611454864319901, 0.3222017264660062, 0.2791985462358334, 0.23686713600258014]
accuracy = [89.06, 90.13, 90.11, 89.97, 100.0, 100.0, 100.0]

# Create a bar plot for all metrics
width = 0.2
x = np.arange(len(algorithms))

plt.figure(figsize=(12, 8))

plt.bar(x - 1.5 * width, rmse, width, label="RMSE", color="blue")
plt.bar(x - 0.5 * width, mae, width, label="MAE", color="green")
plt.bar(x + 0.5 * width, r2_score, width, label="R2 Score", color="red")
plt.bar(x + 1.5 * width, accuracy, width, label="Accuracy", color="purple")

plt.title("Combined Metrics for Hybrid Algorithms")
plt.xticks(x, algorithms, rotation=45)
plt.grid(axis="y", linestyle="--")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your dataset file path)
data = pd.read_csv('/content/encoded2_dataset.csv')

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()