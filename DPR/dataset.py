import json
import random
import os

def load_dataset(dataset, rel_path='./'):
    if dataset == 'HotpotQA':
        train_data = [json.loads(_) for _ in open('{}/HotpotQA/train_with_neg_v0.json').readlines() if len(json.loads(_)['neg_paras']) >= 2]
        val_data = [json.loads(_) for _ in open('{}/HotpotQA/dev_with_neg_v0.json').readlines()]

        chunk_pool = []
    

    elif dataset in ['2WikiMQA', 'IIRC', 'MuSiQue']:

        train_data = json.load(open(f'{rel_path}/train_data/{dataset}/train.json', 'r'))
        val_data = json.load(open(f'{rel_path}/train_data/{dataset}/val.json', 'r'))

        chunks = json.load(open(f'{rel_path}/train_data/{dataset}/chunks.json', 'r'))
        titles = json.load(open(f'{rel_path}/train_data/{dataset}/titles.json', 'r'))

        chunk_pool = [(chunk, title) for chunk, title in zip(chunks, titles)]

    return train_data, val_data, chunk_pool


def load_dataset_inf(dataset, rel_path='./'):
    return json.load(open(f'{rel_path}/QA-data/{dataset}/test_docs.json', 'r'))


if __name__ == "__main__":
    dataset = 'wikimultihop'
    
    load_dataset(dataset = dataset)
    
