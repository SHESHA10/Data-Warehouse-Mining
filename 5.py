# Implementation of the Apriori Algorithm
# Enter the Transactions
transactions = [
["Milk", "Egg", "Bread", "Butter"],
["Milk", "Butter", "Egg", "ketchup"],
["Bread", "Butter", "ketchup"],
["Milk", "Bread", "Butter"],
["Bread", "Butter", "Cookies"],
["Milk", "Bread", "Butter", "Cookies"],
["Milk", "Cookies"],
["Milk", "Bread", "Butter"],
["Bread", "Butter", "Egg", "Cookies"],
["Milk", "Butter", "Bread"],
["Milk", "Bread", "Butter"],
["Milk", "Bread", "Cookie", "ketchup"]]
print(len(transactions), "transactions")
# Preprocessing: Clean the Item Names
clean = []
for t in transactions:
    items = [i.strip().capitalize() for i in t]
    items = ["Cookies" if i == "Cookie" else i for i in items]
    clean.append(sorted(set(items)))
for n, t in enumerate(clean, 1):
    print(n, t)
# One-Hot Encoding
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(clean).transform(clean),
columns=te.columns_)
print(onehot.astype(int))
# Count Each Item
import matplotlib.pyplot as plt
counts = onehot.sum().sort_values(ascending=False)
print(counts)
counts.plot(kind="bar", color="steelblue", figsize=(7, 4.5))
plt.axhline(0.4 * len(onehot), color="red", linestyle="--")
plt.ylabel("Number of transactions"); plt.title("Item Counts")
plt.xticks(rotation=0)
plt.show()
# Generate the Frequent Itemsets
from mlxtend.frequent_patterns import apriori
freq = apriori(onehot, min_support=0.4, use_colnames=True)
freq["itemsets"] = freq["itemsets"].apply(
lambda s: ", ".join(sorted(s)))
print(freq.round(3))
# Generate Rules with Lift ≥ 1
from mlxtend.frequent_patterns import association_rules
freq = apriori(onehot, min_support=0.4, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1)
show = rules[["antecedents", "consequents", "support",
"confidence", "lift"]].copy()
for c in ["antecedents", "consequents"]:
    show[c] = show[c].apply(lambda s: ", ".join(sorted(s)))
print(show.round(3).to_string())
# Rules by Minimum Confidence
r2 = association_rules(freq, metric="confidence", min_threshold=0.75)
show = r2[["antecedents", "consequents", "support",
"confidence", "lift"]].copy()
for c in ["antecedents", "consequents"]:
    show[c] = show[c].apply(lambda s: ", ".join(sorted(s)))
print(show.round(3).to_string())