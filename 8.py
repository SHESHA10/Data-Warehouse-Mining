# 8. Implementation of Naïve Bayes
# Naïve Bayes on Iris
# Create iris.csv
# Create assess.csv
import pandas as pd
data = {
"cgpa": [">=8", ">=8", "<8", ">=8", "<8",
">=8", ">=8", "<8", ">=8", "<8"],
"Interactiveness": ["Yes", "Yes", "No", "Yes", "No",
"Yes", "No", "Yes", "Yes", "No"],
"P_K": ["Good", "Good", "Average", "Good", "Average",
"Good", "Average", "Good", "Good", "Average"],
"Comm_Skills": ["Moderate", "Good", "Poor", "Moderate", "Poor",
"Good", "Poor", "Moderate", "Good", "Poor"],
"Job_Offer": ["Yes", "Yes", "No", "Yes", "No",
"Yes", "No", "Yes", "Yes", "No"]
}
ds = pd.DataFrame(data)
ds.to_csv("assess.csv", index=False)
print("assess.csv created successfully")
print(ds.head())
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
data = iris.frame
data["target"] = data["target"].map(dict(enumerate(iris.target_names)))
data.to_csv("iris.csv", index=False)
print(data.head(3))
# Load the File and Change the Header
pd.set_option("display.width", 200)
df = pd.read_csv("iris.csv")
print("Old header:", list(df.columns))
df.columns = ["sepal_length", "sepal_width", "petal_length",
"petal_width", "species"]
print(df.head())
# Preprocessing Checks
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df["species"].value_counts())
print(df.describe().round(2))
# Look at the Classes
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7, 4.5))
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Petal Length by Species")
plt.show()
# Split: 80% Training, 20% Test
from sklearn.model_selection import train_test_split
x = df.drop("species", axis="columns")
y = df["species"]
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.20, random_state=0, stratify=y)
print("Training rows:", len(x_train), " Test rows:", len(x_test))
# Train the Model
from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
nb = nbmodel.fit(x_train, y_train)
print("Classes:", nb.classes_)
print("Priors:", nb.class_prior_.round(3))
print(pd.DataFrame(nb.theta_.round(2), index=nb.classes_,
columns=x.columns))
# Predict and Measure the Accuracy
from sklearn.metrics import accuracy_score, classification_report
y_pred = nb.predict(x_test)
print("Naive Bayes Accuracy is", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))
# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()
# Predict the Class of New Flowers
new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5]],
columns=x.columns)
print(nb.predict(new))
print(pd.DataFrame(nb.predict_proba(new).round(3),
columns=nb.classes_))
# Naïve Bayes by Hand on Categorical Data
# Priors from Counts
ds = pd.read_csv("assess.csv")
n = len(ds)
yes = ds[ds["Job_Offer"] == "Yes"]
no = ds[ds["Job_Offer"] == "No"]
print("P(Yes) =", len(yes), "/", n, "=", round(len(yes) / n, 3))
print("P(No) =", len(no), "/", n, "=", round(len(no) / n, 3))
# Classify One Student by Hand
X = {"cgpa": ">=8", "Interactiveness": "Yes",
"P_K": "Good", "Comm_Skills": "Moderate"}
score = {}
for name, part in [("Yes", yes), ("No", no)]:
    p = len(part) / n
    for a, v in X.items():
        k = (part[a] == v).sum()
        print("P(%s=%s | %s) = %d/%d" % (a, v, name, k, len(part)))
        p = p * k / len(part)
    score[name] = p
for k, v in score.items():
    print("Score for", k, "=", round(float(v), 5))
# The Same with CategoricalNB
# OrdinalEncoder turns the categories into numbers; CategoricalNB then
# works from the counts, with smoothing so that no probability is
# exactly 0.
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
enc = OrdinalEncoder()
xa = enc.fit_transform(ds.drop("Job_Offer", axis="columns"))
ya = ds["Job_Offer"]
cnb = CategoricalNB().fit(xa, ya)
print("Accuracy on the 28 students:",
round(accuracy_score(ya, cnb.predict(xa)) * 100, 2))
xs = enc.transform(pd.DataFrame([X]))
print(cnb.classes_, cnb.predict_proba(xs).round(3))