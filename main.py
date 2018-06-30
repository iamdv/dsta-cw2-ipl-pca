
# coding: utf-8
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ipl = pd.read_csv('./IPL_DATA/deliveries.csv')

list(ipl)

batsman_total = ipl.groupby("batsman").total_runs.sum()

batsman_total = pd.DataFrame(batsman_total)

batsman_data = batsman_total.merge(batsman_strike_rate, on='batsman', how='left')

batsman_data = batsman_data.rename(columns = {0:'strike_rate'})

plt.scatter(batsman_data.batsman_total, batsman_data.batsman_strike_rate)

ipl_reduced = PCA(n_components=2).fit_transform(batsman_data)

fig = plt.figure(1, figsize=(8, 6))

ipl_x = Axes3D(fig, elev=-100, azim=100)

ipl_reduced

ipl_x.scatter(ipl_reduced[:, 0], ipl_reduced[:, 1],
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ipl_x.set_title("Two PCA directions for IPL Batsman")

ipl_x.set_xlabel("1st eigenvector")

ipl_x.w_xaxis.set_ticklabels([])

ipl_x.set_ylabel("2nd eigenvector")

ipl_x.w_yaxis.set_ticklabels([])

plt.show()

