from parse import parse_args
import json
import os
import pickle as pkl
from graph_construct import tfidf_kw_extract, wiki_spacy_extract, \
    kw_process_graph, tagme_extract, knn_graph_construct, knn_embs_construct, \
    mdr_embs_construct, knn_mdr_graph_construct
import multiprocessing as mp
import torch

def set_cuda_device():

    # Check and set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon MPS support
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    # Set the start method before any other multiprocessing code


    args = parse_args()
    args.path = os.getcwd()

    # args.kg = 'MDR-KNN'
    # args.dataset='2WikiMQA' 
    # args.n_processes=64
    # args.k_knn=1

    # test_docs = json.load(open('./Data-Collect//{}/test_docs.json'.format(args.dataset), 'r'))

    test_docs = json.load(open('./{}/test_docs.json'.format(args.dataset), 'r'))

    if args.kg == 'TF-IDF':
        test_docs = tfidf_kw_extract(test_docs, args.n_kw, args.min_n, args.max_n, args.n_processes)
        test_docs_graphs = kw_process_graph(test_docs)

        info = 'TF-IDF_{}_{}_{}'.format(args.n_kw, args.min_n, args.max_n)

    elif args.kg == 'wiki_spacy':
        test_docs = wiki_spacy_extract(test_docs, args.n_processes)
        test_docs_graphs = kw_process_graph(test_docs)

        info = 'wiki_spacy'

    elif args.kg == 'TAGME':
        test_docs = tagme_extract(test_docs, args.n_processes, args.prior_prob)
        test_docs_graphs = kw_process_graph(test_docs)

        info = 'TAGME_{}'.format(args.prior_prob)

    elif args.kg == 'KNN':
        if not os.path.exists('./{}/knn_embs.pkl'.format(args.dataset)):
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            # mp.set_start_method('spawn')
            knn_embs_construct(args.dataset, test_docs, args.n_processes)

        test_docs_graphs = knn_graph_construct(args.dataset, test_docs, args.k_knn, args.n_processes)

        info = 'KNN_{}'.format(args.k_knn)

    elif args.kg == 'MDR-KNN':
        # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = set_cuda_device()

        if not os.path.exists('./{}/mdr_embs.pkl'.format(args.dataset)):
            mdr_embs_construct(args.dataset, test_docs, args)

        test_docs_graphs = knn_mdr_graph_construct(args.dataset, test_docs, args.k_knn, args.n_processes)
        info = 'MDR-KNN_{}'.format(args.k_knn)


    pkl.dump(test_docs_graphs, open('./{}/KG_{}.pkl'.format(args.dataset, info), 'wb'))

