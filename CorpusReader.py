class CorpusReader:
    path = "train.txt"

    def get_entities(self, s):
        e1_start, e1_end = "<e1>", "</e1>"
        e2_start, e2_end = "<e2>", "</e2>"
        e1 = s[s.find(e1_start)+len(e1_start):s.find(e1_end)
               ].strip()  # check if this is necessary idk
        e2 = s[s.find(e2_start)+len(e2_start):s.find(e2_end)].strip()
        before_e1 = s[:s.find(e1_start)]
        after_e2 = s[s.find(e2_end)+len(e2_end):]
        between = s[s.find(e1_end)+len(e1_end):s.find(e2_start)]
        filtered = s
        for f in [e1_start, e1_end, e2_start, e2_end]:
            filtered = filtered.replace(f, "")
        return filtered, e1, e2, between, before_e1, after_e2

    def get_relation(self, s):
        i = 0
        relation, direction = "", ""
        while i < len(s) and s[i] != "(":
            relation += s[i]
            i += 1
        direction = s[i:]
        return relation, direction

    def read(self, file):
        self.path = file
        dataset = []
        all_entities = {}
        all_relations = {}
        corpus = open(self.path)
        while True:
            sentence_data = corpus.readline()
            relation_data = corpus.readline()
            if not sentence_data:
                break
            if sentence_data == "\n":
                continue
            # print(sentence_data.split("\t"))
            tab_splitted = sentence_data.split("\t")
            if len(tab_splitted) < 2:
                continue
            id, sentence_raw = tab_splitted[0], tab_splitted[1]
            sentence_raw = sentence_raw.replace("\"", "")
            sentence_raw = sentence_raw.replace("\n", "")
            sentence_filtered, e1, e2, between, before_e1, after_e2 = self.get_entities(
                sentence_raw)
            relation, direction = self.get_relation(relation_data)
            all_entities[e1] = all_entities.get(e1, 0) + 1
            all_entities[e2] = all_entities.get(e2, 0) + 1
            all_relations[relation] = all_relations.get(relation, 0) + 1
            data = Data(id, sentence_raw, sentence_filtered, e1, e2,
                        between, before_e1, after_e2, relation, direction)
            dataset.append(data)
        return dataset, all_entities, all_relations


class Data:
    def __init__(self, id, sentence_raw, sentence_filtered, e1, e2,  text_between, text_before_e1, text_after_e2, relation, direction):
        self.id = id
        self.sentence_raw = sentence_raw
        self.sentence_filtered = sentence_filtered
        self.e1 = e1
        self.e2 = e2
        self.text_between = text_between
        self.text_before_e1 = text_before_e1
        self.text_after_e2 = text_after_e2
        self.relation = relation
        self.direction = direction

    def __str__(self):
        return f'id: {self.id}\nraw: {self.sentence_raw}\nfiltered: {self.sentence_filtered}\ne1: {self.e1}\ne2: {self.e2}\nbetween: {self.text_between}\nbefore_e1: {self.text_before_e1}\nafter_e2: {self.text_after_e2}\nrelation: {self.relation}\ndirection: {self.direction}'
