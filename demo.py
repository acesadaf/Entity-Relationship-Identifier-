from processor_display import process, get_vec, pad, rel_dir, dim, f
from CorpusReader import Data, CorpusReader
from Embedder_vn import get_embeddings
import gensim
from joblib import parallel_backend, dump, load

try:
    print("loading word vector model...")
    model = gensim.models.Word2Vec.load(
        "word_vec_100.model")  # assumes model exists, if not please run main.py to create it before using the demo
    print("loading SVM classifier...")
    clf = load('SVM_SPT_JobLib_100.joblib')
except:
    print("Either the Word2Vec embedding model or the model itself does not exist. Please run main.py to create them before using the demo")
    exit()
while True:
    # "The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
    sentence_raw = input("Please give an input sentence or press X to exit\n")
    if sentence_raw == "X":
        exit()
    sentence_raw = sentence_raw.replace("\"", "")
    sentence_raw = sentence_raw.replace("\n", "")
    reader = CorpusReader()
    sentence_filtered, e1, e2, between, before_e1, after_e2 = reader.get_entities(
        sentence_raw)
    data = Data(id=0, sentence_raw=sentence_raw,
                sentence_filtered=sentence_filtered, e1=e1, e2=e2, text_between=between, text_before_e1=before_e1, text_after_e2=after_e2, relation="Test", direction="N\A")

    dataset = [data]
    X_test = process(size=1, dataset=dataset, model=model, dim=dim)
    X_test = pad(X_test, maxes=10, dim=dim)
    inv_rel = {}  # key = label number value = relation WITH dir, eg, key = 1, value = Product-Producer2
    for key, value in rel_dir.items():
        inv_rel[value] = key

    inv_dir = {}  # key = label number value = relation without dir, eg, key = 1, value = Product-Producer
    for r in rel_dir:
        if r == "Other\n":
            inv_dir[rel_dir[r]] = r[:5]
        else:
            without = r[:-1]
            inv_dir[rel_dir[r]] = without

    # print(rel_dir)
    # print(inv_dir)
    # print(inv_rel)

    X_tst = []
    for xt in X_test:
        arr = []
        for feature in xt:
            for vector in feature:
                for v in vector:
                    arr.append(v)
        X_tst.append(arr)
    y_pred = clf.predict(X_tst)
    this_point = y_pred[0]
    if inv_dir[this_point] == "Other":
        print("The relation is: ", inv_dir[this_point])
    else:
        print("The relation is: ", inv_dir[this_point])
        dirn = "(e1,e2)" if inv_rel[this_point][-1] == "1" else "(e2,e1)"
        print("The direction is: ", dirn)
