from nltk.corpus import verbnet
from nltk.wsd import lesk
import spacy
from joblib import parallel_backend, dump, load
from spacy.lemmatizer import Lemmatizer
import numpy as np
from nltk.corpus import wordnet
import networkx as nx
from os import path
from nltk.corpus import framenet as fn
import nltk
import nltk
from terminaltables import AsciiTable
from nltk import Tree
# nltk.download('verbnet')
# nltk.download('framenet_v17')
dim = 100
f = 30
relations = set(['Product-Producer', 'Entity-Destination', 'Entity-Origin', 'Component-Whole', 'Cause-Effect',
                 'Instrument-Agency', 'Message-Topic', 'Member-Collection', 'Content-Container', 'Other\n'])

if not path.isfile("rel_dirs.joblib"):
    rel_dir = {}
    count = -1
    for relation in relations:
        if relation == "Other\n":
            rel_dir[relation] = count + 1
            count += 1

        else:
            rel_dir[relation+"1"] = count + 1
            rel_dir[relation+"2"] = count + 2
            count += 2
    dump(rel_dir, "rel_dirs.joblib")

else:
    rel_dir = load("rel_dirs.joblib")

rels = len(rel_dir)


def tok_format(tok):
    return "_".join([tok.orth_, tok.dep_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)


def get_vec(word, model):
    if word in model.wv:
        return model.wv[word]
    elif word.lower() in model.wv:
        return model.wv[word.lower()]
    elif word.upper() in model.wv:
        return model.wv[word.upper()]
    else:
        return model.wv["$"]


def pad(X, maxes, dim):
    for i in range(len(X)):
        for j in range(f):
            length = len(X[i][j])
            if length < maxes:
                if len(X[i][j]) > 0:
                    X[i][j].extend(
                        [[0]*dim for _ in range(maxes - length)])
                else:
                    X[i][j] = [[0]*dim for _ in range(maxes)]
            else:
                X[i][j] = X[i][j][:maxes]
    return X


def process(size, dataset, model, dim):
    X = []
    nlp = spacy.load('en_core_web_sm')
    for idx, sentence in enumerate(dataset[:size]):
        print("Sentence: ", sentence.sentence_filtered)
        print("Entity 1: ", sentence.e1)
        print("Entity 2: ", sentence.e2)
        print("Text between: ", sentence.text_between)
        print("-------------------------------------------------------------")

        e1, e2, tb, be, af = sentence.e1.split(
        ), sentence.e2.split(), sentence.text_between.split(), sentence.text_before_e1.split()[::-1][:5], sentence.text_after_e2.split()[:5]

        sentence.text_before_e1.split(), sentence.text_after_e2.split()
        data = [[get_vec(e, model) for e in e1] + [
            get_vec(e, model) for e in e2]]

        filtered = sentence.sentence_filtered
        s = nlp(filtered)

        print("Tokenization: Here is a list of the sentence tokens")
        print([token.text for token in s])
        print("-------------------------------------------------------------")

        print("Lemmatization: Here is a list of all sentence tokens, lemmatized")
        word_lemma = [["Token", "Lemma"]]
        for token in s:
            word_lemma.append([token, token.lemma_])
        #print([token.lemma_ for token in s])
        lemma_table = AsciiTable(word_lemma)
        print(lemma_table.table)
        print("-------------------------------------------------------------")

        print("POS Tagging:")
        word_tag = [["Token", "Tag"]]
        for token in s:
            word_tag.append([token, token.tag_])
        tag_table = AsciiTable(word_tag)
        print(tag_table.table)
        print("-------------------------------------------------------------")

        levin_classes = []
        levin_classes_print = []
        levin_lemmas = [[] for _ in range(3)]
        ll = 0
        s_tb = nlp(sentence.text_between)
        for token in s_tb:
            if token.pos_ == "VERB":
                disam_syn = lesk(sentence.sentence_filtered, token.text)
                for cid in verbnet.classids(token.lemma_):
                    levin_classes.append(get_vec(cid, model))
                    levin_classes_print.append(("Verb: " + token.text, cid))
                    for idx, v in enumerate(verbnet.lemmas(cid)):
                        if ll <= 2 and len(levin_lemmas[ll]) > 9:
                            ll += 1
                        if ll > 2:
                            # print("here")
                            break

                        # print(ll)
                        levin_lemmas[ll].append(get_vec(v, model))
        print("Levin Classes of the verbs in the text between the entities")
        print(levin_classes_print if levin_classes_print else "Not Applicable")
        print("-------------------------------------------------------------")
        data += [levin_classes]
        data += levin_lemmas

        sp_e1, sp_e2, sp_tb, sp_be, sp_af = nlp(" ".join(e1)), nlp(" ".join(
            e2)), nlp(" ".join(tb)), nlp(" ".join(be)), nlp(" ".join(af))
        # printing dependency parse tree
        print("Dependency parse tree:")
        [to_nltk_tree(sent.root).pretty_print() for sent in s.sents]

        edges = []
        for token in s:
            for child in token.children:
                edges.append(('{0}'.format(token),
                              '{0}'.format(child)))

        graph = nx.Graph(edges)

        first_entity_tokens = [token.text for token in sp_e1]
        second_entity_tokens = [token.text for token in sp_e2]
        fe, se = "", ""
        for fet in first_entity_tokens:
            if graph.has_node(fet):
                fe = fet
                break
        for sect in second_entity_tokens:
            if graph.has_node(sect):
                se = sect
                break
        if fe and se:
            # print("found path!")
            # print(fe, se)
            try:
                spt = nx.shortest_path(
                    graph, source=fe, target=se)
                spt = spt[:10]
            except:
                spt = []
        else:
            spt = []
        # print(len(spt))
        print("Shortest Dependency path: ")
        print("".join([element + "->" if i != len(spt) -
                       1 else element for i, element in enumerate(spt)]))
        print("-------------------------------------------------------------")
        spt_vec = [get_vec(w, model) for w in spt]
        # print(spt_vec)
        data += [spt_vec]

        print("Named Entity Recognition for Entity 1, {}:".format(sentence.e1),
              [ent.label_ for ent in sp_e1.ents])

        print("Named Entity Recognition for Entity 2, {}:".format(sentence.e2),
              [ent.label_ for ent in sp_e2.ents])

        e1_ner = ([get_vec(ent.label_, model)
                   for ent in sp_e1.ents] + [[0]*dim])[:1]
        e2_ner = ([get_vec(ent.label_, model)
                   for ent in sp_e2.ents] + [[0]*dim])[:1]
        e1_lem = ([get_vec(token.lemma_, model)
                   for token in sp_e1] + [[0]*dim for _ in range(3)])[:3]
        e2_lem = ([get_vec(token.lemma_, model)
                   for token in sp_e2] + [[0]*dim for _ in range(3)])[:3]

        e_additional = [e1_ner+e2_ner+e1_lem+e2_lem]
        data += e_additional

        be_lem = ([get_vec(token.lemma_, model)
                   for token in sp_be] + [[0]*dim for _ in range(5)])[:5]

        af_lem = ([get_vec(token.lemma_, model)
                   for token in sp_af] + [[0]*dim for _ in range(5)])[:5]

        be_af_lem = [be_lem+af_lem]
        data += be_af_lem

        tb_lem = ([get_vec(token.lemma_, model)
                   for token in sp_tb] + [[0]*dim for _ in range(10)])[:10]

        data += [tb_lem]
        print("FrameNet Frames: ")
        e1_fn_print = []
        e1_fn = []
        e1_fnf = []
        for token in sp_e1:
            # print([f.frame.name for f in fn.fes(token.lemma_)][:5])
            e1_fn_print += [f.frame.name for f in fn.fes(token.lemma_)][:5]
            temp = [get_vec(f.frame.name, model)
                    for f in fn.fes(token.lemma_)][: 5]
            tempf = [get_vec(f.name, model)
                     for f in fn.fes(token.lemma_)][: 5]
            if temp:
                e1_fn += temp
            if tempf:
                e1_fnf += tempf
        e1_fn = e1_fn[: 10]
        e1_fnf = e1_fnf[: 10]
        # print(e1_fn)

        e2_fn_print = []
        e2_fn = []
        e2_fnf = []
        for token in sp_e2:
            # print([f.frame.name for f in fn.fes(token.lemma_)][:5])
            e2_fn_print += [f.frame.name for f in fn.fes(token.lemma_)][:5]
            temp = [get_vec(f.frame.name, model)
                    for f in fn.fes(token.lemma_)][: 5]
            tempf = [get_vec(f.name, model)
                     for f in fn.fes(token.lemma_)][: 5]
            if temp:
                e2_fn += temp
            if tempf:
                e2_fnf += tempf
        e2_fn = e2_fn[: 10]
        e2_fnf = e2_fnf[: 10]
        # print(e2_fn)
        tb_fn_print = []
        tb_fn = []
        tb_fnf = []
        for token in sp_tb:
            if token.tag_ in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                tb_fn_print += [("Verb: " + token.text, f.frame.name)
                                for f in fn.fes(token.lemma_)][:5]
                # print([f.frame.name
                # for f in fn.fes(token.lemma_)][:3])
                temp = [get_vec(f.frame.name, model)
                        for f in fn.fes(token.lemma_)][: 3]
                if temp:
                    tb_fn += temp
                tempf = [get_vec(f.frame.name, model)
                         for f in fn.fes(token.lemma_)][: 3]
                if tempf:
                    tb_fnf += tempf

        print("Entity 1 Frames: ", None if not e1_fn_print else e1_fn_print)
        print("Entity 2 Frames: ", None if not e2_fn_print else e2_fn_print)
        print("In-between Verb Frames: ",
              None if not tb_fn_print else tb_fn_print)
        print("-------------------------------------------------------------")
        data += [e1_fn]
        data += [e1_fnf]
        data += [e2_fn]
        data += [e2_fnf]
        data += [tb_fn]
        data += [tb_fnf]

        sp_list = [sp_e1, sp_e2, sp_be, sp_af]  # leaving out sp_tb
        # 5th element is sp_tb which is added later after the for loop
        tag_content = [[] for _ in range(4)]
        # wordnet_content = [[] for _ in range(5)]
        print("Wordnet features [without disambiguation]")
        print("Entity 1: ", sentence.e1)
        hyp, hypo, holo, mero = [], [], [], []
        for i, word in enumerate(sp_e1):
            for synset in wordnet.synsets(word.text):
                hyp += [lemma.name() for hyper in synset.hypernyms()
                        for lemma in hyper.lemmas()]
                hypo += [lemma.name() for hypo in synset.hyponyms()
                         for lemma in hypo.lemmas()]
                holo += [lemma.name() for holo in synset.part_holonyms()
                         for lemma in holo.lemmas()]
                mero += [lemma.name() for mero in synset.part_meronyms()
                         for lemma in mero.lemmas()]

        print("Hypernyms: ", hyp[:10] if hyp else None)
        print("Hyponyms: ", hypo[:10] if hypo else None)
        print("Holonyms: ", holo[:10] if holo else None)
        print("Meronyms: ", mero[:10] if mero else None)

        print("Entity 2: ", sentence.e2)
        hyp, hypo, holo, mero = [], [], [], []
        for i, word in enumerate(sp_e2):
            for synset in wordnet.synsets(word.text):
                hyp += [lemma.name() for hyper in synset.hypernyms()
                        for lemma in hyper.lemmas()]
                hypo += [lemma.name() for hypo in synset.hyponyms()
                         for lemma in hypo.lemmas()]
                holo += [lemma.name() for holo in synset.part_holonyms()
                         for lemma in holo.lemmas()]
                mero += [lemma.name() for mero in synset.part_meronyms()
                         for lemma in mero.lemmas()]

        print("Hypernyms: ", hyp[:10] if hyp else None)
        print("Hyponyms: ", hypo[:10] if hypo else None)
        print("Holonyms: ", holo[:10] if holo else None)
        print("Meronyms: ", mero[:10] if mero else None)
        print("-------------------------------------------------------------")

        for j, sp in enumerate(sp_list):
            wordnet_content = [[] for _ in range(3)]
            # all tags will be in one vector of length maxes, called final_tag_list

            for i, word in enumerate(sp):
                disam_syn = lesk(sentence.sentence_filtered, word.text)
                if i > 3:
                    break
                for synset in wordnet.synsets(word.text):
                    if synset == disam_syn:
                        wordnet_content[0] += ([get_vec(lemma.name(), model)
                                                for lemma in synset.lemmas()]+[[0]*dim for _ in range(5)])[: 5]
                        wordnet_content[0] += ([get_vec(lemma.name(), model) for hyper in synset.hypernyms()
                                                for lemma in hyper.lemmas()]+[[0]*dim for _ in range(5)])[: 5]
                        wordnet_content[1] += ([get_vec(lemma.name(), model) for hypo in synset.hyponyms()
                                                for lemma in hypo.lemmas()]+[[0]*dim for _ in range(5)])[: 5]
                        wordnet_content[1] += ([get_vec(lemma.name(), model) for mero in synset.part_meronyms()
                                                for lemma in mero.lemmas()]+[[0]*dim for _ in range(5)])[: 5]
                        wordnet_content[2] += ([get_vec(lemma.name(), model) for holo in synset.part_holonyms(
                        ) for lemma in holo.lemmas()]+[[0]*dim for _ in range(5)])[: 5]

                tag_content[j] += [get_vec(word.tag_, model)]
                # print(tag_content[i])

            data += wordnet_content

        # print(tag_content)
        for i in range(len(tag_content)):
            tag_content[i] = (
                tag_content[i] + [[0]*dim for _ in range(3)])[:3]

        # sp_e1, sp_e2 in one, sp_be, sp_af in another
        final_tag_list = [[], []]
        # print(tag_content)
        for a, tc in enumerate(tag_content):
            # print(tc)
            for vec in tc:
                # print(final_tag_list, vec)
                if a <= 1:
                    final_tag_list[0] += [vec]
                else:
                    final_tag_list[1] += [vec]

        sp_tbm = " ".join(tb)  # modified
        length = len(sp_tbm)
        half = length//2
        sp_tb2 = nlp(sp_tbm[max(half-5, 0):min(half+5, length)])
        in_between_tags = [[get_vec(w.tag_, model) for w in sp_tb2]]
        # final_tag_list[0] +=

        # print(final_tag_list)
        data += final_tag_list
        data += in_between_tags
        X.append(data)

    return X
