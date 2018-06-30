
# coding: utf-8

# In[70]:


import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[59]:


ipl = pd.read_csv('./IPL_DATA/deliveries.csv')


# In[60]:


list(ipl)


# In[83]:


batsman_total = ipl.groupby("batsman").total_runs.sum()


# In[85]:


batsman_total = pd.DataFrame(batsman_total)


# In[63]:


batsman_data = batsman_total.merge(batsman_strike_rate, on='batsman', how='left')


# In[66]:


batsman_data = batsman_data.rename(columns = {0:'strike_rate'})


# In[ ]:


plt.scatter(batsman_data.batsman_total, batsman_data.batsman_strike_rate)


# In[87]:


ipl_reduced = PCA(n_components=2).fit_transform(batsman_data)


# In[91]:


fig = plt.figure(1, figsize=(8, 6))


# In[109]:


ipl_x = Axes3D(fig, elev=-100, azim=100)


# In[110]:


ipl_reduced


# In[111]:


ipl_x.scatter(ipl_reduced[:, 0], ipl_reduced[:, 1],
           cmap=plt.cm.Set1, edgecolor='k', s=40)


# In[112]:


ipl_x.set_title("Two PCA directions for IPL Batsman")


# In[113]:


ipl_x.set_xlabel("1st eigenvector")


# In[114]:


ipl_x.w_xaxis.set_ticklabels([])
ipl_x.set_ylabel("2nd eigenvector")
ipl_x.w_yaxis.set_ticklabels([])


# In[115]:


plt.show()

