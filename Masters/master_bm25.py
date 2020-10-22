import sys
sys.path.insert(0, '../../EMNLP2020')

from Fitting import bm25_fit
import time
import json
from handlers import load_data
import argparse
import os, sys
import datetime
from handlers.output_handler import FileHandler
from setting_keywords import KeyWordSettings
import pandas as pd

def get_query_docs(datapack: pd.DataFrame) -> dict:
    dict_queries = {}  # key is a queryID, value is [query_content, list_labels, list of docs and list of docIDs]
    for index, row in datapack.iterrows():
        query = row["id_left"]
        doc = row["id_right"]
        label = row["label"]
        query_content = row["text_left"]
        doc_content = row["text_right"]
        dict_queries[query] = dict_queries.get(query, [[], [], [], []])  # text, labels, docIDs, docContents
        s, labels, docIDs, docContents = dict_queries[query]
        if s: s[0] = query_content  # I do this to make it updated by reference
        else: s.append(query_content)
        assert query_content == s[0], "content must be matched"
        labels.append(label)
        docIDs.append(doc)
        docContents.append(doc_content)
    return dict_queries


def fit_models(args):
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    curr_date = datetime.datetime.now().timestamp()  # seconds
    # folder to store all outputed files of a run
    secondary_log_folder = os.path.join(args.log, "log_results_%s" % (int(curr_date)))
    if not os.path.exists(secondary_log_folder):
        os.mkdir(secondary_log_folder)

    logfolder_result = os.path.join(secondary_log_folder, "%s_result.txt" % int(curr_date))
    FileHandler.init_log_files(logfolder_result)
    settings = json.dumps(vars(args), sort_keys = True, indent = 2)
    FileHandler.myprint("Running script " + str(os.path.realpath(__file__)))
    FileHandler.myprint(settings)

    root = args.path
    train_pack = load_data.load_data2(root, 'train', prefix=args.dataset)
    valid_pack = load_data.load_data2(root, 'dev', prefix=args.dataset)
    predict_pack = load_data.load_data2(root, 'test', prefix=args.dataset)


    # print(train_pack.left)

    a = train_pack["text_left"].str.lower().str.split().apply(len).max()
    b = valid_pack["text_left"].str.lower().str.split().apply(len).max()
    c = predict_pack["text_left"].str.lower().str.split().apply(len).max()
    max_query_length = max([a, b, c])
    min_query_length = min([a, b, c])

    a = train_pack["text_right"].str.lower().str.split().apply(len).max()
    b = valid_pack["text_right"].str.lower().str.split().apply(len).max()
    c = predict_pack["text_right"].str.lower().str.split().apply(len).max()
    max_doc_length = max([a, b, c])
    min_doc_length = min([a, b, c])

    FileHandler.myprint("Min query length, " + str(min_query_length) + " Min doc length " + str(min_doc_length))
    FileHandler.myprint("Max query length, " + str(max_query_length) + " Max doc length " + str(max_doc_length))
    t1 = time.time()
    # get_query_docs(train_pack)
    dev_queries = get_query_docs(valid_pack)
    test_queries = get_query_docs(predict_pack)

    additional_data = {}
    if args.reranking:
        predict2_hard_pack = load_data.load_data2(root, 'test2_hard', prefix=args.dataset)
        predict3_hard_pack = load_data.load_data2(root, 'test3_hard', prefix=args.dataset)
        test2_queries = get_query_docs(predict2_hard_pack)
        test3_queries = get_query_docs(predict3_hard_pack)
        additional_data[KeyWordSettings.Test2Hard] = test2_queries
        additional_data[KeyWordSettings.Test3Hard] = test3_queries

    FileHandler.myprint('done extracting')
    t2 = time.time()
    FileHandler.myprint('loading data time: %d (seconds)' % (t2 - t1))
    params = {"b": args.b, "k1": args.k1}

    """ Many other things"""

    FileHandler.myprint("Fitting Model")
    fit_model = bm25_fit.BM25Fitter(params)

    try:
        fit_model.fit(None, verbose = True, topN = args.topk, val_queries = dev_queries, test_queries = test_queries, **additional_data)
    except KeyboardInterrupt:
        FileHandler.myprint('Exiting from training early')
    t10 = time.time()
    FileHandler.myprint('Total time:  %d (seconds)' % (t10 - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Description: Running Neural IR Search Models")
    parser.add_argument('--path', default = '../test_code/ben_data_10_no_body', help = 'Input data path', type = str)
    parser.add_argument('--log', default = "../logs/bm25", type = str, help = 'folder for logs and saved models')
    parser.add_argument('--dataset', default = "Snopes", type = str, help = '[Snopes, Politifact]]')
    parser.add_argument('--topk', type = int, default = 3, help = 'top K')
    parser.add_argument('--k1', type = float, default = 1.2, help = 'k1 hyper-param for bm25')
    parser.add_argument('--b', type = float, default = 0.75, help = 'b hyper-param for bm25')
    parser.add_argument('--reranking', type=int, default=1, help='Re-ranking or not')
    args = parser.parse_args()
    fit_models(args)
