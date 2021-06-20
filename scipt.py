import numpy as np
import pandas as pd

df = pd.read_csv("SUV_Purchase.csv")
df.head()

df.info()

# Remove User ID
df.drop('User ID', axis = 1, inplace = True)
# Encoding Gender
df.replace({'Male':1, 'Female':0}, inplace = True)
# Featire Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Age','EstimatedSalary']] = sc.fit_transform(df.loc[:, ['Age','EstimatedSalary']])

# variable Separation
X = df.drop('Purchased', axis = 1).values
y = df.loc[:, 'Purchased'].values

# Train test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
## results of 
print('Variance Captured by PCA')
for i in np.arange(pca.n_components_):
    print("PC{0:d}: {1:.1f}%".format(i+1, 100*pca.explained_variance_ratio_[i]))

