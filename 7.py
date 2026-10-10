# 7. Implementation of Decision Tree:
# Type the Dataset
# Each string is one student: cgpa, Interactiveness, P_K, Comm_Skills, Job_Offer.
rows = [
">=9,Yes,Very good,Good,Yes", ">=8,No,Good,Moderate,Yes",
">=9,No,Average,Poor,No", "<8,No,Average,Good,No",
">=8,Yes,Good,Moderate,Yes", ">=9,Yes,Good,Moderate,Yes",
"<8,Yes,Good,Poor,No", ">=9,No,Very good,Good,Yes",
">=8,Yes,Good,Good,Yes", ">=8,Yes,Average,Good,Yes",
">=9,Yes,Very good,Good,Yes", ">=8,No,Good,Moderate,Yes",
">=9,No,Average,Poor,No", "<8,Yes,Average,Good,No",
">=8,Yes,Good,Moderate,Yes", ">=9,Yes,Average,Moderate,Yes",
"<8,Yes,Good,Poor,No", ">=9,No,Very good,Good,Yes",
">=8,Yes,Good,Good,No", ">=8,Yes,Average,Good,Yes",
">=9,No,Good,Good,Yes", ">=8,Yes,Good,Good,Yes",
">=8,Yes,Average,Good,Yes", ">=9,Yes,Very good,Good,Yes",
">=8,No,Good,Moderate,Yes", "<8,No,Average,Poor,Yes",
"<8,Yes,Average,Good,No", ">=8,Yes,Good,Moderate,Yes"]
print(len(rows), "rows typed")
import pandas as pd
with open("assess.csv", "w") as f:
    f.write("cgpa,Interactiveness,P_K,Comm_Skills,Job_Offer\n")
    f.write("\n".join(rows) + "\n")
ds = pd.read_csv("assess.csv")
print(ds.head())
print(ds.shape)
# Look at the Class Distribution
# value_counts counts the students in each class.
print(ds["Job_Offer"].value_counts())
print(pd.crosstab(ds["cgpa"], ds["Job_Offer"]))
# Information Gain of Each Attribute
# The function entropy follows the formula. The loop computes the gain of each attribute for the
# whole dataset.
import numpy as np
def entropy(col):
    p = col.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()
total = entropy(ds["Job_Offer"])
print("Entropy of Job_Offer:", round(total, 4))
for a in ["cgpa", "Interactiveness", "P_K", "Comm_Skills"]:
    after = sum(len(g) / len(ds) * entropy(g["Job_Offer"])
    for _, g in ds.groupby(a))
    print(a, "gain =", round(total - after, 4))
# Encode the Categories as Numbers
ds["cgpa"] = ds["cgpa"].map({">=9": 2, ">=8": 1, "<8": 0})
ds["Interactiveness"] = ds["Interactiveness"].map({"Yes": 1, "No": 0})
ds["P_K"] = ds["P_K"].map({"Very good": 2, "Good": 1, "Average": 0})
ds["Comm_Skills"] = ds["Comm_Skills"].map(
{"Good": 2, "Moderate": 1, "Poor": 0})
ds["Job_Offer"] = ds["Job_Offer"].map({"Yes": 1, "No": 0})
print(ds.head())
# Split into Training and Test Sets
from sklearn.model_selection import train_test_split
x = ds.drop("Job_Offer", axis="columns")
y = ds["Job_Offer"]
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.20, random_state=0)
print("Training rows:", len(x_train))
print("Test rows:", len(x_test))
# Train the Decision Tree
# fit builds the tree from the training rows. criterion="entropy" selects attributes by information
# gain; max_depth=3 limits the tree to three levels.
from sklearn.tree import DecisionTreeClassifier
dtmodel = DecisionTreeClassifier(criterion="entropy",
max_depth=3, random_state=0)
dt = dtmodel.fit(x_train, y_train)
print("Tree depth:", dt.get_depth())
print("Number of leaves:", dt.get_n_leaves())
# Predict and Measure the Accuracy
# predict gives the class of each test row. accuracy_score compares the predictions with the true
# classes.
from sklearn.metrics import accuracy_score, classification_report
y_pred = dt.predict(x_test)
print("Predicted:", y_pred.tolist())
print("Actual: ", y_test.tolist())
print("Accuracy is", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, zero_division=0))
# Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(
y_test, y_pred, display_labels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.show()
# Draw the Tree
from sklearn import tree
plt.figure(figsize=(13, 7))
tree.plot_tree(dtmodel, feature_names=list(x.columns),
class_names=["No", "Yes"], filled=True, fontsize=11)
plt.show()
# The Tree as Rules
print(tree.export_text(dtmodel, feature_names=list(x.columns)))
# Predict for a New Student
# new student: cgpa ≥ 9 (2), interactive (1), practical knowledge Good (1), communication
# Moderate (1).
new = pd.DataFrame([[2, 1, 1, 1]], columns=x.columns)
result = dt.predict(new)
print("Predicted class:", result[0])
if result[0] == 1:
    print("The student is likely to get a job offer")
else:
    print("The student is not likely to get a job offer")