import pandas as pd
import numpy as np
rows = [
["S01","F",19,85,4,68,70,72], ["S02","M",20,70,3,55,58,60],
["S03","M",19,55,1,40,42,45], ["S04","F",21,80,4,65,68,70],
["S05","M",20,88,5,72,75,78], ["S06","F",19,60,2,45,48,50],
["S07","M",22,None,6,80,82,85], ["S08","F",20,50,1,35,38,40],
["S09","M",21,75,3,58,60,63], ["S10","F",19,82,4,68,70,73],
["S11","M",20,68,2,50,52,55], ["S12","F",21,90,5,None,78,80],
["S13","M",19,52,1,38,40,42], ["S14","F",20,72,3,56,59,61],
["S15","M",22,92,6,82,85,88], ["S16","F",19,78,4,62,66,70],
["S17","M",21,65,2,48,50,53], ["S18","F",20,95,14,88,90,92],
["S19","M",19,58,2,44,46,48], ["S20","F",21,84,5,70,74,76],
["S05","M",20,88,5,72,75,78]]
print(len(rows), "rows typed")
# Save the File
cols = ["Student_ID", "Gender", "Age", "Attendance", "Study_Hours",
"Internal_Mark", "Assignment_Mark", "Final_Mark"]
data = pd.DataFrame(rows, columns=cols)
data["Result"] = np.where(data["Final_Mark"] >= 50, "Pass", "Fail")
data.to_csv("student.csv", index=False)
print("student.csv created")
# Load the Data and See the First Rows
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
df = pd.read_csv("student.csv")
print(df.head())
print("Shape:", df.shape)
# Columns and Data Types
df.info()
# Statistical Summary
print(df.describe().round(1))
# Find the Missing Values
print(df.isnull().sum())
# Fill the Missing Values
df["Attendance"] = df["Attendance"].fillna(
round(df["Attendance"].mean(), 1))
df["Internal_Mark"] = df["Internal_Mark"].fillna(
df["Internal_Mark"].median())
print(df.loc[[6, 11], ["Student_ID", "Attendance", "Internal_Mark"]])
print("Missing values left:", df.isnull().sum().sum())
# Find and Remove Duplicate Rows
print("Duplicate rows:", df.duplicated().sum())
print(df[df.duplicated()][["Student_ID", "Final_Mark"]])
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)
# Statistics of Final_Mark
m = df["Final_Mark"]
print("Mean:", round(m.mean(), 2))
print("Median:", m.median())
print("Mode:", m.mode().tolist())
print("Variance:", round(m.var(), 2))
print("Standard deviation:", round(m.std(), 2))
print("Minimum:", m.min(), " Maximum:", m.max())
# Box Plot of Study_Hours
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7, 4))
sns.boxplot(x=df["Study_Hours"], color="lightblue")
plt.title("Box Plot of Study Hours")
plt.show()
# Outlier Detection by the IQR Rule
q1 = df["Study_Hours"].quantile(0.25)
q3 = df["Study_Hours"].quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("Q1 =", q1, " Q3 =", q3, " IQR =", iqr)
print("Allowed range:", low, "to", high)
out = df[(df["Study_Hours"] < low) | (df["Study_Hours"] > high)]
print(out[["Student_ID", "Study_Hours"]])
# Treat the Outlier
med = int(df["Study_Hours"].median())
df.loc[df["Study_Hours"] > high, "Study_Hours"] = med
print("Median used:", med)
print("Maximum Study_Hours now:", df["Study_Hours"].max())
df.to_csv("student_clean.csv", index=False)
# Histogram of Final Marks
plt.figure(figsize=(7, 4.5))
plt.hist(df["Final_Mark"], bins=6, color="skyblue", edgecolor="black")
plt.xlabel("Final Mark"); plt.ylabel("Number of Students")
plt.title("Distribution of Final Marks")
plt.show()
# Scatter Plot: Hours vs Marks
plt.figure(figsize=(7, 4.5))
sns.scatterplot(x="Study_Hours", y="Final_Mark", hue="Result",
data=df, s=90)
plt.title("Study Hours vs Final Mark")
plt.grid(True)
plt.show()
# Bar Chart: Pass and Fail Counts
counts = df["Result"].value_counts()
print(counts)
counts.plot(kind="bar", color=["green", "red"], figsize=(6, 4.5))
plt.ylabel("Number of Students"); plt.title("Result Count")
plt.xticks(rotation=0)
plt.show()
# Compare Groups: Gender
g = df.groupby("Gender")[["Attendance", "Study_Hours","Final_Mark"]].mean().round(2)
print(g)
# Correlation Heatmap
num = df[["Age", "Attendance", "Study_Hours",
"Internal_Mark", "Assignment_Mark", "Final_Mark"]]
print(num.corr().round(2)["Final_Mark"])
plt.figure(figsize=(8, 5.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()