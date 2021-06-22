import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
dataset = load_wine()
# print(dataset.DESCR)


# print(dataset.target_names)
# print(dataset.target)

# print(dataset.feature_names)
# print(dataset.data)


"""Prepare explanatory variable as DataFrame in pandas"""
df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
"""Add the target variable to df"""
df["target"] = dataset.target
print(df.head())


"""
Visualize the dataset
"""
show_data = False # True or False
if show_data:
    plt.figure(figsize=(6, 6))
    # class 0
    plt.scatter(df[df['target']==0]['alcohol'],
                df[df['target']==0]['magnesium'],
                label='class 0', color='red')
    # class 1
    plt.scatter(df[df['target']==1]['alcohol'],
                df[df['target']==1]['magnesium'],
                label='class 1', color='blue')
    # class 2
    plt.scatter(df[df['target']==2]['alcohol'],
                df[df['target']==2]['magnesium'],
                label='class 2', color='green')
    plt.xlabel("alcohol")
    plt.ylabel("magnesium")
    plt.show()








