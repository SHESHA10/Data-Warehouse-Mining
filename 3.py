import numpy, pandas, matplotlib, seaborn
import scipy, sklearn, mlxtend
print("All required packages are installed.")
# Create the Dataset
import pandas as pd
data = {
"Student_ID": ["S%02d" % i for i in range(1, 16)],
"Study_Hours": [2,3,1,4,5,2,6,1,3,4,2,5,1,3,6],
"Attendance": [65,70,55,80,85,60,90,50,75,82,68,88,52,72,92],
"Internal_Mark": [48,55,40,65,72,45,80,35,58,68,50,75,38,56,82],
"Assignment_Mark": [50,58,42,68,75,48,82,38,60,70,52,78,40,59,85],
"Final_Mark": [52,60,45,70,78,50,85,40,63,73,55,80,42,61,88],
"Result": ["Pass","Pass","Fail","Pass","Pass","Pass","Pass","Fail",
"Pass","Pass","Pass","Pass","Fail","Pass","Pass"]}
pd.DataFrame(data).to_csv("student_performance.csv", index=False)
print("Dataset created")
# NumPy: Statistics of the Final Marks
import numpy as np
marks = np.array([52,60,45,70,78,50,85,40,63,73,55,80,42,61,88])
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Standard deviation:", round(np.std(marks), 2))
# NumPy: Working on the Whole Array
print("Marks + 5 grace:", marks + 5)
print("Above 70:", marks[marks > 70])
print("Count above 70:", np.sum(marks > 70))
print("Sorted:", np.sort(marks))
# Pandas: Load and Look at the Data
df = pd.read_csv("student_performance.csv")
print(df.head())
print("Shape:", df.shape)
# Pandas: Statistical Summary
print(df.describe().round(2))
# Pandas: Missing Values and Duplicates
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
# Pandas: Filter and Group
high = df[df["Final_Mark"] >= 70]
print(high[["Student_ID", "Final_Mark", "Result"]])
print(df.groupby("Result")["Final_Mark"].mean())
# Pictures of the data
# Matplotlib: Histogram
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.hist(df["Final_Mark"], bins=5, edgecolor="black")
plt.xlabel("Final Mark")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Marks")
plt.show()
# Matplotlib: Scatter Plot
plt.figure(figsize=(7, 5))
plt.scatter(df["Study_Hours"], df["Final_Mark"], color="green")
plt.xlabel("Study Hours")
plt.ylabel("Final Mark")
plt.title("Study Hours vs Final Mark")
plt.grid(True)
plt.show()
# Seaborn: Box Plot
import seaborn as sns
plt.figure(figsize=(7, 5))
sns.boxplot(x="Result", y="Final_Mark", data=df)
plt.title("Final Mark by Result")
plt.show()
# Seaborn: Correlation Heatmap
# corr computes the correlation of every pair of numeric columns. heatmap colours the table.
num = df[["Study_Hours", "Attendance", "Internal_Mark",
"Assignment_Mark", "Final_Mark"]]
plt.figure(figsize=(8, 6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Matrix")
plt.show()
# SciPy, Scikit-learn and Mlxtend- Statistics, learning and rules
# SciPy: Correlation and z-Scores
from scipy import stats
r, p = stats.pearsonr(df["Study_Hours"], df["Final_Mark"])
print("r =", round(r, 3), " p-value =", format(p, ".2e"))
z = stats.zscore(df["Final_Mark"])
print("z-scores:", np.round(z[:5], 2))
# Scikit-learn: Split the Data
from sklearn.model_selection import train_test_split
X = df[["Study_Hours", "Attendance",
"Internal_Mark", "Assignment_Mark"]]
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.30, random_state=42, stratify=y)
print("Training rows:", len(X_train), " Test rows:", len(X_test))
# Scikit-learn: Train and Evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted:", y_pred.tolist())
print("Actual: ", y_test.tolist())
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"]))
# Mlxtend: Frequent Itemsets
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
baskets = [["Milk","Bread"], ["Milk","Bread","Butter"],
["Bread","Butter"], ["Milk","Bread","Butter"],
["Milk","Butter"]]
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(baskets).transform(baskets),
columns=te.columns_)
print(apriori(onehot, min_support=0.6, use_colnames=True))