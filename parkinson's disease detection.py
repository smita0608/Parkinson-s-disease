import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('parkinsons.csv')
X=dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,12,14,15,16,18,19,20,21,22,23]].values
y=dataset.iloc[:,17].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

from sklearn.neighbors import KNeighborsClassifier
classifi = KNeighborsClassifier(n_neighbors = 5,p=6,metric ='minkowski')
classifi.fit(X_train,y_train)

y_pred = classifi.predict(X_test)

from sklearn.svm import SVC
classifi2 = SVC()
classifi2.fit(X_train,y_train)

y2_pred = classifi2.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
classifi3 = RandomForestClassifier(n_estimators=10,criterion = "entropy",random_state=0)
classifi3.fit(X_train,y_train)

y3_pred = classifi3.predict(X_test)

from sklearn.tree import DecisionTreeClassifier as DT
classifi4 = DT(criterion='entropy', random_state=0)
classifi4.fit(X_train,y_train)

y4_pred = classifi4.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

cm2=confusion_matrix(y_test,y2_pred)
accuracy_score(y_test,y2_pred)

cm3=confusion_matrix(y_test,y3_pred)
accuracy_score(y_test,y3_pred)

cm4=confusion_matrix(y_test,y4_pred)
accuracy_score(y_test,y4_pred)