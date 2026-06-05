from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

import argparse

parser = argparse.ArgumentParser(description='lda')
parser.add_argument('--sim_df', type=str, required=True)
parser.add_argument('--data_df', type=str, required=True)
parser.add_argument('--output_plot', type=str, required=True)

args = parser.parse_args()

sim_df = pd.read_csv(args.sim_df)
data_df = pd.read_csv(args.data_df)
sim_df = sim_df[data_df.columns]

X = np.vstack((data_df.values, sim_df.values))
y = np.array([0]*len(data_df["max_corr"]) + [1]*len(sim_df["max_corr"]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

data_lda_scores = lda.transform(data_df.values).flatten()
sim_lda_scores = lda.transform(sim_df.values).flatten()

X_train_lda = lda.transform(X_train)

all_values = np.concatenate([
    np.array(X_train_lda[y_train == 0]).flatten(),
    np.array(X_train_lda[y_train == 1]).flatten()
])


bins = np.linspace(math.floor(np.min(all_values)),
                   math.ceil(np.max(all_values)),
                   30)

plt.figure()
plt.hist(X_train_lda[y_train==0], bins=bins, alpha=0.6, color='blue', label='Data', density = True, cumulative = -1, histtype = "step", linewidth = 5)
plt.hist(X_train_lda[y_train==1], bins=bins, alpha=0.6, color='red', label='Sim', density = True, cumulative = -1, histtype = "step", linewidth = 5)
plt.xlabel('Linear Discriminant')
plt.ylabel('CDF')
plt.yscale('log')
plt.legend(loc = "upper right")
plt.savefig(args.output_plot)
plt.close()

