import sys
sys.path.insert(0, '../EMNLP2020')
sys.path.insert(0, '../../EMNLP2020')

from Models import multimodal_attention_network
from Fitting import visual_fitter as fitter
from Fitting import contextualized_fitter
import time
import json
from interactions import MatchInteractionVisual
import matchzoo as mz
from handlers import load_data
import argparse
import random
import numpy as np
import torch
import os
import datetime
from handlers.output_handler import FileHandler
import matchzoo.datasets
import torch_utils


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
    FileHandler.myprint("Setting seed to " + str(args.seed))

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    index2queries = dict((y, x) for x, y in json.loads(open(args.query_mapped).read()).items())
    index2docs = dict((y, x) for x, y in json.loads(open(args.article_mapped).read()).items())
    root = args.path
    use_reranking = "reranking" in root
    t1 = time.time()

    elmo_queries_path = os.path.join(args.elmo_feats, "queries_feats.pth")
    elmo_docs_path = os.path.join(args.elmo_feats, "articles_feats.pth")
    elmo_loader = load_data.ElmoLoader(elmo_queries_path, elmo_docs_path, args.fixed_length_left, args.fixed_length_right)
    load_data_func = elmo_loader.elmo_load_data

    train_pack = load_data_func(root, 'train', prefix = args.dataset)
    valid_pack = load_data_func(root, 'dev', prefix = args.dataset)
    predict_pack = load_data_func(root, 'test', prefix = args.dataset)
    if use_reranking:
        FileHandler.myprint("Using Re-Ranking Dataset..........")
        predict2_hard_pack = load_data_func(root, 'test2_hard', prefix = args.dataset)

    a = train_pack.left["text_left"].str.lower().str.split().apply(len).max()
    b = valid_pack.left["text_left"].str.lower().str.split().apply(len).max()
    c = predict_pack.left["text_left"].str.lower().str.split().apply(len).max()
    max_query_length = max([a, b, c])
    min_query_length = min([a, b, c])

    a = train_pack.right["text_right"].str.lower().str.split().apply(len).max()
    b = valid_pack.right["text_right"].str.lower().str.split().apply(len).max()
    c = predict_pack.right["text_right"].str.lower().str.split().apply(len).max()
    max_doc_length = max([a, b, c])
    min_doc_length = min([a, b, c])

    FileHandler.myprint("Min query length, " + str(min_query_length) + " Min doc length " + str(min_doc_length))
    FileHandler.myprint("Max query length, " + str(max_query_length) + " Max doc length " + str(max_doc_length))

    if args.use_visual:
        image_loader = load_data.ImagesLoader(left_pth_file = args.left_images_features, max_num_left_images = args.n_img_in_query,
                                              right_pth_file = args.right_images_features, max_num_right_images = args.n_img_in_doc,
                                              use_cuda = args.cuda)
        data_packs = [train_pack, valid_pack, predict_pack]
        if use_reranking:
            data_packs.append(predict2_hard_pack)

        image_loader.fit(data_packs)  # memory-intensive (~10Gb RAM)
        train_pack = image_loader.transform(train_pack)
        valid_pack = image_loader.transform(valid_pack)
        predict_pack = image_loader.transform(predict_pack)
        if use_reranking:
            predict2_hard_pack = image_loader.transform(predict2_hard_pack)

        print(image_loader.left_tensor.size(), image_loader.right_tensor.size())

    preprocessor = mz.preprocessors.ElmoPreprocessor(args.fixed_length_left, args.fixed_length_right)
    print('parsing data')
    train_processed = preprocessor.fit_transform(train_pack)  # This is a DataPack
    valid_processed = preprocessor.transform(valid_pack)
    predict_processed = preprocessor.transform(predict_pack)

    train_interactions = MatchInteractionVisual(train_processed)
    valid_interactions = MatchInteractionVisual(valid_processed)
    test_interactions = MatchInteractionVisual(predict_processed)
    if use_reranking:
        predict2_processed = preprocessor.transform(predict2_hard_pack)
        predict2_interactions = MatchInteractionVisual(predict2_processed)

    FileHandler.myprint('done extracting')
    t2 = time.time()
    FileHandler.myprint('loading data time: %d (seconds)' % (t2 - t1))
    FileHandler.myprint("Building model")

    print("Loading word embeddings......")
    t1_emb = time.time()
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension = args.word_embedding_size,
                                                                  term_index = term_index)

    embedding_matrix = glove_embedding.build_matrix(term_index)
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis = 1))
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    t2_emb = time.time()
    print("Time to load word embeddings......", (t2_emb - t1_emb))

    match_params = {}
    match_params['embedding'] = embedding_matrix
    match_params["embedding_freeze"] = True  # freezing word embeddings
    match_params["fixed_length_left"] = args.fixed_length_left
    match_params["fixed_length_right"] = args.fixed_length_right
    match_params['dropout'] = 0.1
    match_params['filters'] = args.filters
    match_params["conv_layers"] = args.conv_layers
    match_params["filters_count_pacrr"] = args.filters_count_pacrr
    match_params["n_s"] = args.n_s
    match_params["max_ngram"] = args.max_ngram
    match_params["head_cnn_type"] = args.head_cnn_type
    match_params["use_visual"] = args.use_visual
    match_params["use_average_dcompositional_att"] = args.use_average_dcompositional_att
    match_params["attention_type"] = args.attention_type
    # contextualized part
    match_params["left_elmo_tensor"] = elmo_loader.left_tensor_feats
    match_params["right_elmo_tensor"] = elmo_loader.right_tensor_feats
    match_params["elmo_vec_size"] = 1024

    if args.use_visual:
        match_params["visual_feature_size"] = image_loader.visual_features_size
        image_loader.left_tensor = torch_utils.gpu(image_loader.left_tensor, args.cuda)
        image_loader.right_tensor = torch_utils.gpu(image_loader.right_tensor, args.cuda)
        match_params["full_left_images_tensor"] = image_loader.left_tensor
        match_params["full_right_images_tensor"] = image_loader.right_tensor

    match_model = multimodal_attention_network.MultiModalAttentionNetwork(match_params)
    FileHandler.myprint("Fitting Model")
    if args.use_visual:
        FileHandler.myprint("Using both Textual and Visual features.......")
        fit_model = fitter.VisualFitter(net = match_model, loss = args.loss_type, n_iter = args.epochs,
                                        batch_size = args.batch_size, learning_rate = args.lr,
                                        early_stopping = args.early_stopping, use_cuda = args.cuda,
                                        num_negative_samples = args.num_neg, logfolder = secondary_log_folder,
                                        curr_date = curr_date, use_visual = args.use_visual,
                                        image_loader = image_loader, index2queries = index2queries,
                                        index2docs = index2docs)
    else:
        FileHandler.myprint("Using Textual content only....")
        fit_model = contextualized_fitter.ContextualizedFitter(net=match_model, loss=args.loss_type,
                                                          n_iter=args.epochs, batch_size=args.batch_size,
                                                          learning_rate=args.lr, early_stopping=args.early_stopping,
                                                          use_cuda=args.cuda, num_negative_samples=args.num_neg,
                                                          logfolder=secondary_log_folder, curr_date=curr_date)

    try:
        fit_model.fit(train_interactions, verbose = True,
                      topN = args.topk,
                      val_interactions = valid_interactions,
                      test_interactions = test_interactions)
        fit_model.load_best_model(valid_interactions, test_interactions, topN = args.topk)
        if use_reranking:
            fit_model.load_best_model_test2_test3(predict2_interactions, None, topN = args.topk)

    except KeyboardInterrupt:
        FileHandler.myprint('Exiting from training early')
    t10 = time.time()
    FileHandler.myprint('Total time:  %d (seconds)' % (t10 - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Description: Running Neural IR Search Models")
    parser.add_argument('--path', default = '..', help = 'Input data path', type = str)
    parser.add_argument('--dataset', type = str, default = 'Snopes', help = '[Snopes, Politifact]')
    parser.add_argument('--epochs', default = 100, help = 'Number of epochs to run', type = int)
    parser.add_argument('--batch_size', default = 16, help = 'Batch size', type = int)
    parser.add_argument('--num_neg', default = 3, type = int,  help = 'num negs per positive candidate')
    parser.add_argument('--lr', default = 0.001, type = float, help = 'Learning rate')
    parser.add_argument('--early_stopping', default = 10, type = int, help = 'The number of step to stop training')
    parser.add_argument('--log', default = "", type = str, help = 'folder for logs and saved models')
    parser.add_argument('--optimizer', nargs = '?', default = 'adam', help = 'optimizer')
    parser.add_argument('--loss_type', nargs = '?', default = 'hinge',
                        help = 'Specify a loss function: bce, pointwise, bpr, hinge, adaptive_hinge, pce')
    parser.add_argument('--word_embedding_size', default = 300, help = 'the dimensions of word embeddings', type = int)
    parser.add_argument('--topk', type = int, default = 3, help = 'top K')
    parser.add_argument('--cuda', type = int, default = 1, help = 'using cuda or not')
    parser.add_argument('--seed', type = int, default = 123456, help = 'random seed')
    parser.add_argument('--fixed_length_left', type = int, default = 100, help = 'Maximum length of each query')
    parser.add_argument('--fixed_length_right', type = int, default = 1000, help = 'Maximum length of each document')
    parser.add_argument('--head_cnn_type', type=str, default="pacrr_plane", help='Head CNN part for text')
    parser.add_argument('--use_visual', type=int, default=1, help='Using images or not as matching')
    parser.add_argument('--n_img_in_query', type=int, default=4, help='Number of images in queries or tweets')
    parser.add_argument('--n_img_in_doc', type=int, default=17, help='Number of images in docs')
    parser.add_argument('--left_images_features', type=str, default=".", help='pth path to features of images of queries')
    parser.add_argument('--right_images_features', type=str, default=".", help='pth path to extracted full images of docs')
    parser.add_argument('--query_mapped', type=str, default=".", help='query_mapped.json')
    parser.add_argument('--article_mapped', type=str, default=".", help='article_mapped.json')
    parser.add_argument('--max_ngram', type = int, default = 1, help = "ngram level for sim matrices")
    parser.add_argument('--filters', type = int, default = 256, help = "Number of filters of 1D CNN")
    parser.add_argument('--conv_layers', type=int, default=3, help='number of conv layers for extracting features')
    parser.add_argument('--filters_count_pacrr', type=int, default=16, help='Number of filters for a conv layer')
    parser.add_argument('--n_s', type=int, default=32, help='Top k strongest signals for each plane')
    parser.add_argument('--use_average_dcompositional_att', type=int, default=0, help='0: using 2*sigmoid, 1: using average')
    parser.add_argument('--attention_type', type=int, default=1, choices=[1, 2, 3, 4], help='Attention Types in generating matrices')
    parser.add_argument('--use_elmo', type = int, default = 1, help = 'using elmo chars or not')
    parser.add_argument('--elmo_feats', type = str, default = "", help = 'path to pretrained features')

    args = parser.parse_args()
    fit_models(args)
