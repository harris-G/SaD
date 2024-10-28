
# python -u slimtest.py --dataset=Movielens1M --topk=[20,50] --alpha=0.01 --l1=0.0001 --mode origin #initial sparse
python dense.py --config_file ./config/sad_movielens1m_m1.ini #initial dense
# python evaluate.py --config_file ./config/sad_movielens1m_m1.ini --modes train
# python -u slimtest.py --dataset=Movielens1M --topk=[20,50] --alpha=0.01 --l1=0.0001 --mode ablation #obtain sparse
# python evaluate.py --config_file ./config/sad_movielens1m_m1.ini --modes test #prediction