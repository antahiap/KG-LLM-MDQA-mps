from evaluation import f1_score, exact_match_score, f1, ems
import pickle as pkl
import json
import numpy as np
import itertools

def evaluate(dataset, retriever, k = None, round = 5, topks = [1, 5, 10, 20, 30], k_emb = 15):    
    if retriever != 'knn':
        res = json.load(open('./Pipeline/result/{}/{}_{}.json'.format(dataset, retriever, k), 'rb'))
    else:
        res = json.load(open('./Pipeline/result/{}/{}_{}_{}.json'.format(dataset, retriever, k_emb, k), 'rb'))
    
    
    filter_res = [r for r in res if r['prediction'] != 'System mistake']
    
    f1s, emss, accs = [], [], []
    
    if retriever not in ['golden', 'no']:
        recall, precision, sp_em = [], [], []
    
    for r in filter_res:
        accs.append(('1' in r['grade'])*1.0)
        
        if dataset in ['hotpotqa', '2WikiMQA', 'musique']:
            f1s.append(f1_score(r['prediction'], r['answer']))
            emss.append(exact_match_score(r['prediction'], r['answer']))
            
        elif dataset in ['iirc']:
            f1s.append(f1(r['prediction'], r['answer']))
            emss.append(ems(r['prediction'], r['answer']))
        
        r['corpus'] = list(itertools.chain(*[_.split('\n') for _ in r['corpus']]))
        if retriever not in ['golden', 'no']:
            evi = set([_[1] for _ in r['supports']])
            
            tmp_recall = []
            tmp_precision = []    
            tmp_sp_em = []
            for kk in topks:
                if kk <= k:
                    tmp = set(r['corpus'][:kk])

                    tmp_recall.append(len(evi.intersection(tmp))/len(evi))
                    tmp_precision.append(len(evi.intersection(tmp))/kk)
                    
                    if evi.issubset(tmp):
                        tmp_sp_em.append(1)
                    else:
                        tmp_sp_em.append(0)
                
            
            recall.append(tmp_recall)
            precision.append(tmp_precision)
            sp_em.append(tmp_sp_em)

    
    print('Acc:', np.mean(accs))
    print('F1:', np.mean(f1s))
    print('EM:', np.mean(emss))
    

    if retriever not in ['golden', 'no']:
        print('Recall:', np.mean(np.array(recall), axis = 0))
        print('Precision:', np.mean(np.array(precision), axis = 0))
        print('SP_EM:', np.mean(np.array(sp_em), axis = 0))

evaluate(dataset = '2WikiMQA', retriever = 'Golden', k = 30)
evaluate(dataset = '2WikiMQA', retriever = 'no', k = 30)
# evaluate(dataset = '2WikiMQA', retriever = 'mhop', k = 30)
evaluate(dataset = '2WikiMQA', retriever = 'DPR', k = 30)
evaluate(dataset = '2WikiMQA', retriever = 'knn', k = 30, k_emb = 15)
evaluate(dataset = '2WikiMQA', retriever = 'bm25', k = 30)
evaluate(dataset = '2WikiMQA', retriever = 'tf-idf', k = 30)
# evaluate(dataset = '2WikiMQA', retriever = 't5', k = 30)