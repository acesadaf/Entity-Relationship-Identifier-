
from collections import Counter
from sklearn import svm
from joblib import load
from nltk.wsd import lesk
from Embedder_vn import get_embeddings
from CorpusReader import CorpusReader, Data
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import networkx as nx
from processor_verbnet import process, get_vec, pad, rel_dir, dim, f
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

datalenset = set()


reader = CorpusReader()
dataset, entities, relations = reader.read("semeval_train.txt")
test_set, test_e, test_r = reader.read("semeval_test.txt")


words = set()

for d in dataset:
    words |= set(d.sentence_filtered.split())


model = get_embeddings(dataset)


sentence_words = []

for data in dataset:
    sentence_words.append([data.id, data.sentence_filtered.split()])

inv_rel = {}
for key, value in rel_dir.items():
    inv_rel[value] = key

inv_dir = {}
for r in rel_dir:
    if r == "Other\n":
        inv_dir[rel_dir[r]] = r[:5]
    else:
        without = r[:-1]
        inv_dir[rel_dir[r]] = without


print("relation dictionary, ", rel_dir)
rels = len(rel_dir)


X_test, y_test, _, _, _ = process(
    len(test_set), test_set, True, model, dim)


maxes = 10


print("padding..")
X_test = pad(X_test, maxes, dim)
print("done..")


print("loading..")
clf = load('SVM_SPT_JobLib_100.joblib')
print("done..flattening")
X_tst = []
for xt in X_test:
    arr = []
    for feature in xt:
        for vector in feature:
            for v in vector:
                arr.append(v)
    X_tst.append(arr)

print("flattened")
y_pred = clf.predict(X_tst)
print("Testing time")

y_actual = []
for i in range(len(y_test)):
    max_val = max(y_test[i])
    for j in range(len(y_test[0])):
        if y_test[i][j] == max_val:
            y_actual.append(j)
            break

count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_actual[i]:
        print(y_pred[i])
        count += 1

count2 = 0
print(inv_dir)
for i in range(len(y_pred)):
    print(test_set[i].sentence_filtered,
          y_pred[i], inv_rel[y_pred[i]], inv_dir[y_pred[i]], "here")
    if inv_dir[y_pred[i]] == inv_dir[y_actual[i]]:
        # print(y_pred[i])
        count2 += 1


print("Both tasks accuracy:")
print(accuracy_score(y_actual, y_pred))
print([(i, inv_rel[i]) for i in range(19)])
# stats = precision_recall_fscore_support(y_actual, y_pred)
print("Both tasks P, R, F, per label:")
print(precision_recall_fscore_support(
    y_actual, y_pred, labels=[i for i in range(19)]))

print("Both tasks P, R, F, macro:")
print(precision_recall_fscore_support(
    y_actual, y_pred, average="macro", labels=[i for i in range(19)]))

print("Both tasks P, R, F, weighted:")
print(precision_recall_fscore_support(
    y_actual, y_pred, average="weighted", labels=[i for i in range(19)]))


y_aa = [inv_dir[actual] for actual in y_actual]
y_pp = [inv_dir[pred] for pred in y_pred]
print("Only relation accuracy: ")
print(accuracy_score(y_aa, y_pp))
print("Only relation matching P, R, F, per label:")
labels = list(set(list(inv_dir.values())))
print(labels)
print(precision_recall_fscore_support(
    y_aa, y_pp, labels=labels))
print("Only relation matching P, R, F, macro:")
print(precision_recall_fscore_support(
    y_aa, y_pp, average="macro", labels=labels))
print("Only relation matching P, R, F, weighted:")
print(precision_recall_fscore_support(
    y_aa, y_pp, average="weighted", labels=labels))
