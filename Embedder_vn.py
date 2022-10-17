from nltk.corpus import verbnet
from nltk.corpus import framenet as fn
import gensim
import gensim.downloader as api
from os import path
from gensim.models import Word2Vec, KeyedVectors
import spacy
from nltk.corpus import wordnet
import nltk
# nltk.download('framenet_v17')
# nltk.download('verbnet')
TAG_LIST = [".", ",", "-LRB-", "-RRB-", "``", ":", "\"\"", "''", ",", "$", "#", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NIL", "NN", "NNP", "NNPS", "NNS",
            "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "ADD", "NFP", "GW", "XX", "BES", "HVS", "_SP"]
POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
DEP_LIST = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj",
            "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]
NER_LIST = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
            "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
FRAMENET_CONTENT = [f.name for f in fn.frames()] + [f.name for f in fn.fes()
                                                    ] + [l.name for l in fn.lus()]
LEVIN_CLASSES = verbnet.classids()
VERBNET_LEMMAS = []
for c in LEVIN_CLASSES:
    for l in verbnet.lemmas(c):
        VERBNET_LEMMAS.append(l)
nlp = spacy.load('en_core_web_sm')


def get_embeddings(dataset):
    if not path.isfile("word_vec_100.model"):
        print("Model doesn't exist")
        if not path.isfile("text8_model_100.model"):
            corpus = api.load("text8")
            model = Word2Vec(corpus, size=100, min_count=1)
            model.save("text8_model_100.model")

        else:
            print("text-8 model exists")
            model = gensim.models.Word2Vec.load("text8_model_100.model")
        print(len(list(model.wv.vocab)))
        print("integrating POS..")
        print("integrating Framenet")
        new_sentences = [d.sentence_filtered.split()
                         # unknown word token
                         for d in dataset] + [["$"]] + [[word.tag_ for word in nlp(d.sentence_filtered)] for d in dataset] + [TAG_LIST] + [POS_LIST] + [DEP_LIST] + [NER_LIST] + [FRAMENET_CONTENT] + [LEVIN_CLASSES] + [VERBNET_LEMMAS]
        wordnet_words = []
        print("integrating verbnet")
        print("integrating wordnet content..")
        print("integrating entities")
        print("integrating lemmas")

        for d in dataset:
            sf = d.sentence_filtered
            s = nlp(sf)
            for word in s:
                for synset in wordnet.synsets(word.text):
                    syn = [lemma.name() for lemma in synset.lemmas()]

                    hyper = [lemma.name() for hyper in synset.hypernyms()
                             for lemma in hyper.lemmas()]
                    hypo = [lemma.name() for hypo in synset.hyponyms()
                            for lemma in hypo.lemmas()]
                    mero = [lemma.name() for mero in synset.part_meronyms()
                            for lemma in mero.lemmas()]
                    holo = [lemma.name() for holo in synset.part_holonyms(
                    ) for lemma in holo.lemmas()]
                    if syn:
                        wordnet_words.append(syn)
                    if hyper:
                        wordnet_words.append(hyper)
                    if hypo:
                        wordnet_words.append(hypo)
                    if mero:
                        wordnet_words.append(mero)
                    if holo:
                        wordnet_words.append(holo)

            entities = [ent.label_ for ent in s.ents]
            if entities:
                # actually from Spacy, just for naming consistency
                wordnet_words.append(entities)

            lemmas = [token.lemma_ for token in s]
            if lemmas:
                wordnet_words.append(lemmas)

        new_sentences = new_sentences + wordnet_words

        model.build_vocab(new_sentences, update=True)
        model.train(new_sentences, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.save("word_vec_100.model")
    else:
        print("the full word2vec model exists.")
        model = gensim.models.Word2Vec.load("word_vec_100.model")

    return model
