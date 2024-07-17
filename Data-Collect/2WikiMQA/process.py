import json
import requests
import bs4
import re
import unicodedata
from text_split import get_text_splitter
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import random
import warnings
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import pickle as pkl
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')


MY_GCUBE_TOKEN = '07e1bd33-c0f5-41b0-979b-4c9a859eec3f-843339462'


def get_paragraphs(page_name):
    #Wikipedia API to get articles

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        r = session.get('https://en.wikipedia.org/wiki/{0}'.format(page_name))
        soup = bs4.BeautifulSoup(r.content)
        html_paragraphs = soup.find_all('p')
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error when requesting page {page_name}: {e}")
        return ""

    cleaned_texts = []
    for p in html_paragraphs:
        cleaned_text = re.sub('(\[[0-9]+\])', '', unicodedata.normalize('NFKD', p.text)).strip()

        if cleaned_text:
            cleaned_texts.append(cleaned_text)

    return ''.join(cleaned_texts)



def process_entry(d, splitter):
    question = d['question']
    answer = d['answer']

    tmp = {c[0]: c[1] for c in d['context']}

    
    supports = list(set([(s[0], tmp[s[0]][s[1]]) for s in d['supporting_facts'] if s[1] < len(tmp[s[0]])]))
    titles = list(set([s[0] for s in supports]))
    
    docs_chunks = []
    title_chunks = []
    docs = []
    for t in titles:
        doc = get_paragraphs(t)
        docs.append(doc)

        doc_chunks = splitter.split_text(doc)

        tmp_s = [s[1] for s in supports if s[0] == t]
        doc_chunks.extend(tmp_s)
        doc_chunks = list(set(doc_chunks))

        docs_chunks.append(doc_chunks)

        for chunk in doc_chunks:
            title_chunks.append((t, chunk))
    
    title_chunks = list(set(title_chunks))
    
    return {'question': question, 'answer': answer, 'titles': titles, 'docs_chunks': docs_chunks, \
            'docs': docs, 'title_chunks': title_chunks, 'supports': supports}


def process(data):
    splitter = get_text_splitter(splitter = "spacy", chunk_size = 250, separator = " ", chunk_overlap = 50)

    pool = mp.Pool(6)
    process_entry2 = partial(process_entry, splitter = splitter)
    process_data = []

    with tqdm(total=len(data)) as pbar:
        for i, res in enumerate(pool.imap_unordered(process_entry2, data)):
            process_data.append(res)
            pbar.update()

    pool.close()
    pool.join()

    return process_data



def negative_pair(process_train_data, process_val_data):
    # if os.path.exists('all_docs.json'):
    #     all_docs = json.load(open('all_docs.json', 'r'))
    # else:
    all_docs = []
    all_titles = set()
    for d in process_train_data:
        all_docs.extend([(title, doc, docs_chunks) for title, doc, docs_chunks in zip(d['titles'], d['docs'], d['docs_chunks']) if title not in all_titles])
        all_titles = all_titles.union(set(d['titles']))
    for d in process_val_data:
        all_docs.extend([(title, doc, docs_chunks) for title, doc, docs_chunks in zip(d['titles'], d['docs'], d['docs_chunks']) if title not in all_titles])
        all_titles = all_titles.union(set(d['titles']))

    json.dump(all_docs, open('./all_docs.json', 'w'))
    
    for d in process_train_data:
        while 1:
            negative_docs = random.sample(all_docs, 10)
            negative_docs_title = [title for title, _, _ in negative_docs]

            if not set(d['titles']).intersection(set(negative_docs_title)):
                break
        
        for title, doc, docs_chunks in negative_docs:
            d['titles'].append(title)
            d['docs'].append(doc)
            d['docs_chunks'].append(docs_chunks)

            for dc in docs_chunks:
                d['title_chunks'].append([title, dc])
    
    for d in process_val_data:
        while 1:
            negative_docs = random.sample(all_docs, 10)
            negative_docs_title = [title for title, _, _ in negative_docs]

            if not set(d['titles']).intersection(set(negative_docs_title)):
                break
        
        for title, doc, docs_chunks in negative_docs:
            d['titles'].append(title)
            d['docs'].append(doc)
            d['docs_chunks'].append(docs_chunks)

            for dc in docs_chunks:
                d['title_chunks'].append([title, dc])
    
    
        
    return process_train_data, process_val_data



class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']


    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }


def wat_entity_linking(text):
    # Main method, text annotation with WAT entity linking system
    wat_url = 'https://wat.d4science.org/wat/tag/tag'
    payload = [("gcube-token", MY_GCUBE_TOKEN),
               ("text", text),
               ("lang", 'en'),
               ("tokenizer", "nlp4j"),
               ('debug', 9),
               ("method",
                "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]

    response = requests.get(wat_url, params=payload)
    return [WATAnnotation(a) for a in response.json()['annotations']]


def wat_annotations(wat_annotations):
    json_list = [w.json_dict() for w in wat_annotations]
    
    return json_list

def wiki_kw_extract_chunk(chunk):
    title, chunk = chunk

    wat_annotations = wat_entity_linking(chunk)
    json_list = [w.json_dict() for w in wat_annotations]
    kw2chunk = defaultdict(set)
    chunk2kw = defaultdict(set)
    
    for wiki in json_list:
        if wiki['wiki_title'] != '' and wiki['prior_prob'] > 0.8:
            kw2chunk[wiki['wiki_title']].add(chunk)
            chunk2kw[chunk].add(wiki['wiki_title'])
    
    kw2chunk[title].add(chunk)
    chunk2kw[chunk].add(title)

    return kw2chunk, chunk2kw

def wiki_kw_extract(data):
    for d in tqdm(data):
        kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
        chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_chunk = {executor.submit(wiki_entity_chunk, chunk): chunk for chunk in d['title_chunks']}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk, inv_chunk = future.result()
                for key, value in chunk.items():
                    kw2chunk[key].update(value)
                for key, value in inv_chunk.items():
                    chunk2kw[key].update(value)

        for key in kw2chunk:
            kw2chunk[key] = list(kw2chunk[key])

        for key in chunk2kw:
            chunk2kw[key] = list(chunk2kw[key])

        d['kw2chunk'] = kw2chunk
        d['chunk2kw'] = chunk2kw

    return data

           
def strip_string(string, only_stopwords = False):
    if only_stopwords:
        return ' '.join([str(t) for t in nlp(string) if not t.is_stop])
    else:
        return ' '.join([str(t) for t in nlp(string) if t.pos_ in ['NOUN', 'PROPN']])



def tfidf_kw_extract_chunk(d, n_kw, ngram_l, ngram_h):
    kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
    chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

    chunks = []
    titles = set()
    for title, chunk in d['title_chunks']:
        chunks.append(strip_string(nlp(chunk)))
        titles.add(title)


    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (ngram_l, ngram_h))
    X = tfidf_vectorizer.fit_transform(chunks)
    term = tfidf_vectorizer.get_feature_names_out()
    score = X.todense()
    kws = list(set(list(term[(-score).argsort()[:, :n_kw]][0]) + list(titles)))

    vec = CountVectorizer(vocabulary = kws, binary=True, ngram_range = (ngram_l, ngram_h), token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!/-]+')
    bow = vec.fit_transform(chunks).toarray()

    bow_tile = np.tile(bow, (bow.shape[0], 1))
    bow_repeat = np.repeat(bow, bow.shape[0], axis = 0)
    common_kw = (bow_tile * bow_repeat).reshape(bow.shape[0], bow.shape[0], -1)
    node1, node2, kw_id = common_kw.nonzero()

    for n1, n2, kw in zip(node1, node2, kw_id):
        if n1 != n2:
            kw2chunk[kws[kw]].add(d['title_chunks'][n1][1])
            kw2chunk[kws[kw]].add(d['title_chunks'][n2][1])

            chunk2kw[d['title_chunks'][n1][1]].add(kws[kw])
            chunk2kw[d['title_chunks'][n2][1]].add(kws[kw])

    for key in kw2chunk:
        kw2chunk[key] = list(kw2chunk[key])

    for key in chunk2kw:
        chunk2kw[key] = list(chunk2kw[key])

    d['kw2chunk'] = kw2chunk
    d['chunk2kw'] = chunk2kw

    return d

def tfidf_kw_extract(data, n_kw, ngram_l, ngram_h, num_processes):
    # partial assign parameter to process_d
    func = partial(tfidf_kw_extract_chunk, n_kw = n_kw, ngram_l = ngram_l, ngram_h = ngram_h)

    with Pool(num_processes) as p:
        data = list(tqdm(p.imap(func, data), total=len(data)))

    return data

def wiki_spacy_extract_chunk(d):
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe("entityLinker", last=True)

    kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
    chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

    for title, chunk in d['title_chunks']:
        doc = nlp(chunk)

        for entity in doc._.linkedEntities:
            entity = entity.get_span().text

            kw2chunk[entity].add(chunk)
            chunk2kw[chunk].add(entity)
        
        kw2chunk[title].add(chunk)
        chunk2kw[chunk].add(title)

    for key in kw2chunk:
        kw2chunk[key] = list(kw2chunk[key])

    for key in chunk2kw:
        chunk2kw[key] = list(chunk2kw[key])

    d['kw2chunk'] = kw2chunk
    d['chunk2kw'] = chunk2kw

    return d


def wiki_spacy_extract(data, num_processes):
    # partial assign parameter to process_d
    func = partial(wiki_spacy_extract_chunk)

    with Pool(num_processes) as p:
        data = list(tqdm(p.imap(func, data), total=len(data)))

    return data


def graph_construct(i_d):
    idx, d = i_d

    G = nx.MultiGraph()

    chunk2id = {}
    for i, chunk in enumerate(d['title_chunks']):
        _, chunk = chunk

        G.add_node(i, chunk = chunk)
        chunk2id[chunk] = i
    
    for kw, chunks in d['kw2chunk'].items():
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                G.add_edge(chunk2id[chunks[i]], chunk2id[chunks[j]], kw = kw)
    
    return idx, G


def process_graph(docs):
    pool = mp.Pool(mp.cpu_count())
    graphs = [None] * len(docs)

    for idx, G in tqdm(pool.imap_unordered(graph_construct, enumerate(docs)), total=len(docs)):
        graphs[idx] = G

    pool.close()
    pool.join()

    return graphs
    
        

if __name__ == '__main__':
    warnings.filterwarnings("ignore")


    train_data = json.load(open('train.json', 'r'))[:4000]
    val_data = json.load(open('dev.json', 'r'))[:1000]


    process_train_data = process(train_data)
    process_val_data = process(val_data)


    json.dump(process_train_data, open('./process_train_data.json', 'w'))
    json.dump(process_val_data, open('./process_val_data.json', 'w'))


    train_docs, val_docs = negative_pair(process_train_data, process_val_data)
    

    length = len(val_docs)
    val_docs, test_docs = val_docs[:int(length/2)], val_docs[int(length/2):]


    json.dump(train_docs, open('./train_docs.json', 'w'))
    json.dump(val_docs, open('./val_docs.json', 'w'))
    json.dump(test_docs, open('./test_docs.json', 'w'))