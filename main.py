import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv('CollegeDistance.csv')


data = data.drop(columns=['rownames'])


missing_values = data.isnull().sum()
print("Brakujące wartości w każdej kolumnie:")
print(missing_values)


print("Statystyki opisowe zmiennych liczbowych:")
print(data.describe())


plt.figure(figsize=(15, 10))
numerical_cols = ['score', 'unemp', 'wage', 'distance', 'tuition', 'education']
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Rozkład zmiennej {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 12))
categorical_cols = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=data[col])
    plt.title(f'Rozkład zmiennej {col}')
plt.tight_layout()
plt.show()


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])


X = data.drop(columns=['score'])
y = data['score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Wyniki modelu Random Forest:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

with open("results/model_report.txt", "w") as f:
    f.write("Wyniki modelu Random Forest:\n")
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"R-squared (R2): {r2}\n")