# 4) Импортируйте необходимые пакеты и классы
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 5) Загрузите данные и распечатайте их
df = pd.read_excel("C:\\Users\\user\Downloads\\breast+cancer+coimbra\\dataR2.xlsx")
print("Размер df:", df.shape)
print(df.head())

# -------------------------------------------------------
# 6) Укажите переменную отклика и фактор(ы)
# Y: Classification (1=Healthy controls, 2=Patients)
# Переведём в 0/1: 0=Healthy, 1=Patient
df["Class01"] = (df["Classification"] == 2).astype(int)

# =======================================================
# 1. Однофакторная логистическая регрессия
# =======================================================

# 6) В однофакторной модели берём фактор X = Glucose
X = df[["Glucose"]].values
y = df["Class01"].values


# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9, stratify=y
)
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# 8) Обучение модели LogisticRegression
log1 = LogisticRegression()
log1.fit(X_train, y_train)

# 9) Проверка на нескольких наблюдениях тестовой выборки
print("\nПример истинных y_test[:5]:", y_test[:5])
print("Пример предсказаний model.predict(X_test[:5]):", log1.predict(X_test[:5]))

# 10) Коэффициенты модели
print("\nКоэффициенты (w):", log1.coef_)
print("Свободный член (b):", log1.intercept_)

# 11) Построение логистической кривой на фоне облака точек
x_grid = np.linspace(df["Glucose"].min() - 5, df["Glucose"].max() + 5, 200).reshape(
    -1, 1
)
p_grid = log1.predict_proba(x_grid)[:, 1]

plt.figure()
plt.scatter(df["Glucose"], df["Class01"], marker=".", label="data")
plt.plot(x_grid, p_grid, label="P(patient|Glucose)")
plt.axhline(0.5, label="threshold 0.5")
plt.xlabel("Glucose, mg/dL")
plt.ylabel("Probability / Class")
plt.title("Логистическая кривая (однофакторная модель)")
plt.grid(True)
plt.legend()
plt.show()

# 12) Прогноз для новых объектов
new_glucose = np.array([[70], [90], [110]])
new_pred = log1.predict(new_glucose)
new_prob = log1.predict_proba(new_glucose)[:, 1]
print("\nНовые объекты (Glucose):", new_glucose.ravel())
print("Предсказанный класс:", new_pred)
print("Вероятность patient:", np.round(new_prob, 3))

plt.figure()
plt.scatter(df["Glucose"], df["Class01"], marker=".", label="data")
plt.scatter(new_glucose.ravel(), new_pred, marker="x", s=60, label="pred class")
plt.xlabel("Glucose, mg/dL")
plt.ylabel("Class (0/1)")
plt.title("Прогноз для новых объектов (однофакторная модель)")
plt.grid(True)
plt.legend()
plt.show()

# 14) Ошибки и качество модели
pred_test = log1.predict(X_test)
acc = accuracy_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)
mse = mean_squared_error(y_test, pred_test)

print("\n=== Метрики (однофакторная модель) ===")
print("Accuracy:", round(acc, 3))
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))

# =======================================================
# 2. Множественная логистическая регрессия (2 фактора)
# =======================================================

# 6) Возьмём два фактора: Glucose и Leptin
Xm = df[["Glucose", "Leptin"]].values
y = df["Class01"].values

# 7) 3D диаграмма рассеивания (облако точек)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df["Glucose"], df["Leptin"], df["Class01"], marker=".")
ax.set_xlabel("Glucose, mg/dL")
ax.set_ylabel("Leptin, ng/mL")
ax.set_zlabel("Class (0/1)")
ax.set_title("3D облако точек: (Glucose, Leptin, Class)")
plt.show()

# 8) Train/test split
X_trainm, X_testm, y_trainm, y_testm = train_test_split(
    Xm, y, test_size=0.25, random_state=9, stratify=y
)
print("\nX_trainm:", X_trainm.shape, "y_trainm:", y_trainm.shape)
print("X_testm:", X_testm.shape, "y_testm:", y_testm.shape)

# 9) Обучение модели LogisticRegression
logm = LogisticRegression(max_iter=1000)
logm.fit(X_trainm, y_trainm)

# 10) Проверка на нескольких наблюдениях
print("\nПример истинных y_testm[:5]:", y_testm[:5])
print("Пример предсказаний model.predict(X_testm[:5]):", logm.predict(X_testm[:5]))

# 11) Коэффициенты модели
print("\nКоэффициенты (w1,w2):", logm.coef_)
print("Свободный член (b):", logm.intercept_)

# 12) Прогноз для новых объектов
new_multi = np.array([[70, 8], [90, 20], [110, 30]])
newm_pred = logm.predict(new_multi)
newm_prob = logm.predict_proba(new_multi)[:, 1]
print("\nНовые объекты (Glucose, Leptin):\n", new_multi)
print("Предсказанный класс:", newm_pred)
print("Вероятность patient:", np.round(newm_prob, 3))

# Визуализация: точки + линия p=0.5 (решающая граница)
plt.figure()
plt.scatter(df["Glucose"], df["Leptin"], c=df["Class01"], marker=".", label="data")
plt.scatter(new_multi[:, 0], new_multi[:, 1], marker="x", s=60, label="new objects")

gx = np.linspace(df["Glucose"].min() - 5, df["Glucose"].max() + 5, 200)
gy = np.linspace(df["Leptin"].min() - 2, df["Leptin"].max() + 2, 200)
xx, yy = np.meshgrid(gx, gy)
grid = np.c_[xx.ravel(), yy.ravel()]
pp = logm.predict_proba(grid)[:, 1].reshape(xx.shape)

cs = plt.contour(xx, yy, pp, levels=[0.5])
plt.clabel(cs, inline=1, fontsize=8)

plt.xlabel("Glucose, mg/dL")
plt.ylabel("Leptin, ng/mL")
plt.title("Двухфакторная логистическая регрессия (граница p=0.5)")
plt.grid(True)
plt.legend()
plt.show()

# 13) Ошибки и качество модели
pred_testm = logm.predict(X_testm)
accm = accuracy_score(y_testm, pred_testm)
maem = mean_absolute_error(y_testm, pred_testm)
msem = mean_squared_error(y_testm, pred_testm)

print("\n=== Метрики (множественная модель) ===")
print("Accuracy:", round(accm, 3))
print("MAE:", round(maem, 3))
print("MSE:", round(msem, 3))
