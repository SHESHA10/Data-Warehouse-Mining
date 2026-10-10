# 6. Implementation of FP-Growth Algorithm
# Enter the Transactions
# Each transaction is a list of the items bought together. All transactions are kept in one list.
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
# Preprocessing: One-Hot Encoding
# The algorithms need a table with one column per item: 1 if the item is in the transaction, 0 if not.
# TransactionEncoder builds it.
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(clean).transform(clean),
columns=te.columns_)
print(onehot.astype(int))
# Order the Items by Frequency
# The items are counted, the infrequent ones are dropped, and each transaction is rewritten in
# decreasing order of count. This is the order used to build the FP-tree.
counts = onehot.sum()
min_count = 0.4 * len(onehot)
order = counts[counts >= min_count].sort_values(
ascending=False, kind="stable").index.tolist()
print("Minimum count:", min_count)
print("Item order:", order)
for n, t in enumerate(clean, 1):
    print(n, [i for i in order if i in t])
# Generate the Frequent Patterns
from mlxtend.frequent_patterns import fpgrowth
fp = fpgrowth(onehot, min_support=0.4, use_colnames=True)
out = fp.copy()
out["itemsets"] = out["itemsets"].apply(
lambda s: ", ".join(sorted(s)))
out["count"] = (out["support"] * len(onehot)).round().astype(int)
print(out.round(3))
# Compare with Apriori
from mlxtend.frequent_patterns import apriori
ap = apriori(onehot, min_support=0.4, use_colnames=True)
same = set(ap["itemsets"]) == set(fp["itemsets"])
print("Apriori itemsets :", len(ap))
print("FP-Growth itemsets:", len(fp))
print("Same result?", same)
# Speed on a Larger Dataset
import numpy as np, time
rng = np.random.default_rng(1)
big = pd.DataFrame(rng.random((5000, 25)) < 0.35,
columns=["I%d" % i for i in range(25)])
for name, fn in [("Apriori", apriori), ("FP-Growth", fpgrowth)]:
    t0 = time.time()
    res = fn(big, min_support=0.02, use_colnames=True)
    print(name, "itemsets:", len(res),
    " time:", round(time.time() - t0, 2), "s")
# Generate the Association Rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(fp, metric="lift", min_threshold=1)
show = rules[["antecedents", "consequents", "support",
"confidence", "lift"]].copy()
for c in ["antecedents", "consequents"]:
    show[c] = show[c].apply(lambda s: ", ".join(sorted(s)))
print(show.round(3).to_string())