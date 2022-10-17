
import os.path
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
from processor_verbnet import process, get_vec, pad, rel_dir, dim, f
import numpy as np
import spacy
from CorpusReader import CorpusReader, Data
from Embedder_vn import get_embeddings
from joblib import parallel_backend, dump, load
from sklearn import svm


datalenset = set()


reader = CorpusReader()
dataset, entities, relations = reader.read("semeval_train.txt")
test_set, test_e, test_r = reader.read("semeval_test.txt")
f = 30
# for data in dataset:
#     # print(data)


words = set()

for d in dataset:
    words |= set(d.sentence_filtered.split())

model = get_embeddings(dataset)


sentence_words = []

for data in dataset:
    sentence_words.append([data.id, data.sentence_filtered.split()])

lemmatized_words = []


vocabulary = set([""])

relations = set([sentence.relation for sentence in dataset])
inv_dir = {}
for r in rel_dir:
    if r == "Other\n":
        inv_dir[rel_dir[r]] = r[:5]
    else:
        without = r[:-1]
        inv_dir[rel_dir[r]] = without

print("relation dictionary, ", rel_dir)
rels = len(rel_dir)

# alternate
X_train, y_train, X_val, y_val, maxes_list = process(
    8000, dataset, False, model, dim)
X_test, y_test, _, _, _ = process(
    len(test_set), test_set, True, model, dim)

print("Number of features are................", len(X_train[0]))
# print(maxes_list)
maxes = max(maxes_list)
maxes = 10


# exit()
with parallel_backend('threading', n_jobs=-1):
    X_train = pad(X_train, maxes, dim)
    X_val = pad(X_val, maxes, dim)
    X_test = pad(X_test, maxes, dim)

    X_final = []

    for xt in X_train:
        arr = []
        for feature in xt:
            for vector in feature:
                for v in vector:
                    arr.append(v)
        X_final.append(arr)
    y_trn = []
    for i in range(len(y_train)):
        max_val = max(y_train[i])
        for j in range(len(y_train[0])):
            if y_train[i][j] == max_val:
                y_trn.append(j)
                break

    clf = svm.SVC(C=0.5, kernel="linear")
    # print(len(X_final[0]))
    print("Dataset loaded. Now training..")
    clf.fit(X_final, y_trn)

    X_tst = []
    for xt in X_test:
        arr = []
        for feature in xt:
            for vector in feature:
                for v in vector:
                    arr.append(v)
        X_tst.append(arr)

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
            # (y_pred[i])
            count += 1

    print("Accuracy on both tasks: ", count/len(y_pred))
    dump(clf, 'SVM_SPT_JobLib_100.joblib')
    count2 = 0
    for i in range(len(y_pred)):
        if inv_dir[y_pred[i]] == inv_dir[y_actual[i]]:
            # print(y_pred[i])
            count2 += 1

    print("Accuracy on relation classification only:", count2/len(y_pred))
    print("For detailed accuracy results use tester.py")
