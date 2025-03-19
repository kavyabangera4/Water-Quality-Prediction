from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv("D:/MP_Team/Kavya/Dataset/water_potability.csv", encoding="latin1")
data = data.ffill()
x = data.drop('Potability', axis=1)
y = data['Potability']

scaler = StandardScaler()
features = x.columns
X = scaler.fit_transform(x)
X = pd.DataFrame(X, columns=features)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=404)

model = DecisionTreeClassifier(criterion='entropy', min_samples_split=9, splitter='best')
model.fit(x_train, y_train)
