import numpy as np
import pandas as pd
import matchzoo
import collections
from tqdm import tqdm
from handlers.output_handler import FileHandler


class MatchInteraction(object):

    def __init__(self, data_pack: matchzoo.DataPack, **kargs):
        FileHandler.myprint("Converting DataFrame to Normal Dictionary of Data")
        self.unique_query_ids, \
        self.dict_query_contents, \
        self.dict_query_lengths, \
        self.dict_query_raw_contents, \
        self.dict_query_positions = self.convert_leftright(data_pack.left, text_key = "text_left",
                                                              length_text_key = "length_left",
                                                              raw_text_key = "raw_text_left")
        self.data_pack = data_pack
        assert len(self.unique_query_ids) == len(set(self.unique_query_ids)), "Must be unique ids"
        self.unique_doc_ids, \
        self.dict_doc_contents, \
        self.dict_doc_lengths, \
        self.dict_doc_raw_contents, \
        self.dict_doc_positions = self.convert_leftright(data_pack.right, text_key = "text_right",
                                                            length_text_key = "length_right",
                                                            raw_text_key = "raw_text_right")

        assert len(self.unique_doc_ids) == len(set(self.unique_doc_ids)), "Must be unique ids for doc ids"
        assert len(self.unique_query_ids) != len(self.unique_doc_ids)

        self.pos_queries, \
        self.pos_docs, \
        self.negatives, \
        self.unique_queries_test = self.convert_relations(data_pack.relation)

        # for queries, padded
        self.np_query_contents = np.array([self.dict_query_contents[q] for q in self.pos_queries])
        self.np_query_lengths = np.array([self.dict_query_lengths[q] for q in self.pos_queries])
        self.query_positions = np.array([self.dict_query_positions[q] for q in self.pos_queries])

        # for docs, padded
        self.np_doc_contents = np.array([self.dict_doc_contents[d] for d in self.pos_docs])
        self.np_doc_lengths = np.array([self.dict_doc_lengths[d] for d in self.pos_docs])
        self.doc_positions = np.array([self.dict_doc_positions[d] for d in self.pos_docs])

        assert self.np_query_lengths.shape == self.np_doc_lengths.shape
        self.padded_doc_length = len(self.np_doc_contents[0])
        self.padded_query_length = len(self.np_query_contents[0])

    def convert_leftright(self, part: pd.DataFrame, text_key: str, length_text_key: str, raw_text_key: str, **kargs):
        """ Converting the dataframe of interactions """
        ids, contents_dict, lengths_dict, position_dict = [], {}, {}, {}
        raw_content_dict = {}
        FileHandler.myprint("[NOTICE] MatchZoo use queryID and docID as index in dataframe left and right, "
                            "therefore, iterrows will return index which is left_id or right_id")
        for index, row in part.iterrows():
            ids.append(index)
            text_ = row[text_key]  # text_ here is converted to numbers and padded
            raw_content_dict[index] = row[raw_text_key]

            if length_text_key not in row: length_ = len(text_)
            else: length_ = row[length_text_key]
            assert length_ != 0
            assert index not in contents_dict
            contents_dict[index] = text_
            lengths_dict[index] = length_
            position_dict[index] = np.pad(np.arange(length_) + 1, (0, len(text_) - length_), 'constant')

        return np.array(ids), contents_dict, lengths_dict, raw_content_dict, position_dict

    def convert_relations(self, relation: pd.DataFrame):
        """ Convert relations.
        Retrieving positive interactions and negative interactions. Particularly,
        for every pair (query, doc) = 1, we get a list of negatives of the query q

        It is possible that a query may have multiple positive docs. Therefore, negatives[q]
        may vary the lengths but not too much.
        """
        queries, docs, negatives = [], [], collections.defaultdict(list)
        unique_queries = collections.defaultdict(list)

        for index, row in relation.iterrows():
            query = row["id_left"]
            doc = row["id_right"]
            label = row["label"]
            assert label == 0 or label == 1
            unique_queries[query] = unique_queries.get(query, [[], [], [], []]) #  doc, label, content, length
            a, b, c, d = unique_queries[query]
            a.append(doc)
            b.append(label)
            c.append(self.dict_doc_contents[doc])
            d.append(self.dict_doc_lengths[doc])

            if label == 1:
                queries.append(query)
                docs.append(doc)
            elif label == 0:
                negatives[query].append(doc)
        assert len(queries) == len(docs)
        return np.array(queries), np.array(docs), negatives, unique_queries


class MatchInteractionVisual(MatchInteraction):
    """Adding visual information. format is same as usual but now I loaded images as visual components"""

    def __init__(self, data_pack: matchzoo.DataPack):
        super(MatchInteraction, self).__init__()
        FileHandler.myprint("Converting DataFrame to Normal Dictionary of Data")
        self.unique_query_ids, \
        self.dict_query_contents, \
        self.dict_query_lengths, \
        self.dict_query_raw_contents, \
        self.dict_query_positions, \
        self.dict_query_imgages = self.convert_leftright(data_pack.left, text_key = "text_left",
                                                         length_text_key = "length_left",
                                                         raw_text_key = "raw_text_left", images_key = "images_left")
        self.data_pack = data_pack
        assert len(self.unique_query_ids) == len(set(self.unique_query_ids)), "Must be unique ids"

        self.unique_doc_ids, \
        self.dict_doc_contents, \
        self.dict_doc_lengths, \
        self.dict_doc_raw_contents, \
        self.dict_doc_positions, \
        self.dict_doc_imgages = self.convert_leftright(data_pack.right, text_key = "text_right",
                                                       length_text_key = "length_right",
                                                       raw_text_key = "raw_text_right", images_key="images_right")

        assert len(self.unique_doc_ids) == len(set(self.unique_doc_ids)), "Must be unique ids for doc ids"
        assert len(self.unique_query_ids) != len(self.unique_doc_ids)

        self.pos_queries, \
        self.pos_docs, \
        self.negatives, \
        self.unique_queries_test = self.convert_relations(data_pack.relation)

        # for queries, padded
        self.np_query_contents = np.array([self.dict_query_contents[q] for q in self.pos_queries])
        self.np_query_lengths = np.array([self.dict_query_lengths[q] for q in self.pos_queries])
        self.query_positions = np.array([self.dict_query_positions[q] for q in self.pos_queries])
        self.query_images = np.array([self.dict_query_imgages[q] for q in self.pos_queries])

        # for docs, padded
        self.np_doc_contents = np.array([self.dict_doc_contents[d] for d in self.pos_docs])
        self.np_doc_lengths = np.array([self.dict_doc_lengths[d] for d in self.pos_docs])
        self.doc_positions = np.array([self.dict_doc_positions[d] for d in self.pos_docs])
        self.doc_images = np.array([self.dict_doc_imgages[d] for d in self.pos_docs])

        assert self.np_query_lengths.shape == self.np_doc_lengths.shape
        self.padded_doc_length = len(self.np_doc_contents[0])
        self.padded_query_length = len(self.np_query_contents[0])
        self.padded_doc_images_len = len(self.doc_images[0])

    def convert_leftright(self, part: pd.DataFrame, text_key: str, length_text_key: str, raw_text_key: str, **kargs):
        """ Converting the dataframe of interactions """
        images_key = kargs["images_key"]
        ids, contents_dict, lengths_dict, position_dict, imgages_dict = [], {}, {}, {}, {}
        raw_content_dict = {}
        for index, row in part.iterrows():
            ids.append(index)
            text_ = row[text_key]  # text_ here is converted to numbers and padded
            raw_content_dict[index] = row[raw_text_key]

            if length_text_key not in row: length_ = len(text_)
            else: length_ = row[length_text_key]
            assert length_ != 0
            assert index not in contents_dict
            contents_dict[index] = text_
            lengths_dict[index] = length_
            position_dict[index] = np.pad(np.arange(length_) + 1, (0, len(text_) - length_), 'constant')
            imgages_dict[index] = row[images_key]  # added images
        return np.array(ids), contents_dict, lengths_dict, raw_content_dict, position_dict, imgages_dict
