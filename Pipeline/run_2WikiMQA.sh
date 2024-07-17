echo "No"
python3 main.py --retriever='No' --dataset='2WikiMQA' --n_processes=16

echo "Golden"
python3 main.py --retriever='Golden' --dataset='2WikiMQA' --n_processes=16

echo "BM25"
python3 main.py --retriever='BM25' --k=30 --dataset='2WikiMQA' --n_processes=16

echo "TF-IDF"
python3 main.py --retriever='TF-IDF' --k=30 --dataset='2WikiMQA' --n_processes=16

echo "MDR"
python3 main.py --retriever='MDR' --dataset='2WikiMQA' --k=30 --n_processes=8

echo "DPR"
python3 main.py --retriever='DPR' --dataset='2WikiMQA' --k=30 --n_processes=8

echo "KNN"
python3 main.py --retriever='KNN' --k=30 --k_emb=15 --dataset='2WikiMQA'

echo "KGP w/o LLM"
python3 main.py --retriever='KG' --dataset='2WikiMQA' --k=30 --n_processes=8

echo "T5"
python3 main.py --retriever='T5' --dataset='2WikiMQA' --k=30 --n_processes=1 --port=5000

echo "LLaMA"
python3 main.py --retriever='LLaMA' --dataset='2WikiMQA' --k=30 --n_processes=1 --port=5000

echo "KGP-T5"
python3 main.py --retriever='KGP-T5' --dataset='2WikiMQA' --k=30 --n_processes=1 --port=5000 --kg="graph_TAGME_0.8"

echo "KGP-LLaMA"
python3 main.py --retriever='KGP-LLaMA' --dataset='2WikiMQA' --k=30 --n_processes=1 --port=5000  --kg="graph_TAGME_0.8"



