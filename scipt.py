import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("SUV_Purchase.csv")
df.head()

df.info()
# Class Names
class_names = {'Not Purchased':0, 'Purchased':1}
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
X_test = pca.transform(X_test)
## results of 
print('Variance Captured by PCA')
for i in np.arange(pca.n_components_):
    print("PC{0:d}: {1:.1f}%".format(i+1, 100*pca.explained_variance_ratio_[i]))


def AUC_ROC(model,X_train, X_test, y_train, y_test) :
    # Training Data
    y_train_prob = model.predict_proba(X_train)
    fpr1, tpr1, thres1 = roc_curve(y_train, y_train_prob[:,1], pos_label = 1)
    auc1 = auc(fpr1, tpr1)
    # testing Data
    y_test_prob = model.predict_proba(X_test)
    fpr2, tpr2, thres2 = roc_curve(y_test, y_test_prob[:,1], pos_label = 1)
    auc2= auc(fpr2, tpr2)
    #plotting
    plt.figure(figsize = (8,6))
    plt.plot(fpr1, tpr1, label = "Training: AUC-ROC = {0:.3f}".format(auc1),
             color = "tab:orange")
    plt.plot(fpr2, tpr2, label = "Testing: AUC-ROC = {0:.3f}".format(auc2),
             color = 'tab:blue')
    plt.plot(fpr1, fpr1, label = "FPR = TPR", color = 'black', linewidth=0.5)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title("ROC Curve for " + model.__class__.__name__)
    plt.legend(loc = "lower right")
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    pass

def cm_Heatmap(y_true, y_pred, class_name = np.unique(y_train)) :
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm, columns = class_name, index = class_name)
    # Plotting
    plt.figure(figsize = (8,6))
    sns.heatmap(cm, annot = True, annot_kws = {'size': 20}, fmt = 'd', cmap = 'Greens')
    plt.xlabel('Observed Class')
    plt.ylabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.show()
    pass


# Logistic Regression
from sklearn.linear_model import LogisticRegression
## Training the model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Training report
print(classification_report(y_train, log_reg.predict(X_train), target_names = class_names))

# test data
cm_Heatmap(y_test, log_reg.predict(X_test), class_names)
# Classification report
print(classification_report(y_test, log_reg.predict(X_test), target_names = class_names))
# AUC-ROC
AUC_ROC(log_reg, X_train, X_test, y_train, y_test)

# k-NearestNeighbours
from sklearn.neighbors import KNeighborsClassifier
## Training model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
# Training report
print(classification_report(y_train, knn.predict(X_train)))

# test data
cm_Heatmap(y_test, knn.predict(X_test), class_names)
# Classification report
print(classification_report(y_test, knn.predict(X_test), target_names = class_names))
# AUC-ROC
AUC_ROC(knn, X_train, X_test, y_train, y_test)

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
## Training model
dtc = DecisionTreeClassifier(max_depth = 4)
dtc.fit(X_train, y_train)
## Trainign report
print(classification_report(y_train, dtc.predict(X_train), target_names = class_names))

# test data
cm_Heatmap(y_test, dtc.predict(X_test), class_names)
# Classification report
print(classification_report(y_test, dtc.predict(X_test), target_names = class_names))
# AUC-ROC
AUC_ROC(dtc, X_train, X_test, y_train, y_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
## Training model
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 3)
rfc.fit(X_train, y_train)
# Training report
print(classification_report(y_train, rfc.predict(X_train), target_names = class_names))

# test data
cm_Heatmap(y_test, rfc.predict(X_test), class_names)
# Classification report
print(classification_report(y_test, rfc.predict(X_test), target_names = class_names))
# AUC-ROC
AUC_ROC(rfc, X_train, X_test, y_train, y_test)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
## Training model
nbc = GaussianNB(priors = np.array([0.5, 0.5]))
nbc.fit(X_train, y_train)
# Training report
print(classification_report(y_train, nbc.predict(X_train), target_names = class_names))

# test data
cm_Heatmap(y_test, nbc.predict(X_test), class_names)
# Classification report
print(classification_report(y_test, nbc.predict(X_test), target_names = class_names))
# AUC-ROC
AUC_ROC(nbc, X_train, X_test, y_train, y_test)