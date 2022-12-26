source .env/bin/activate

python run.py --model NaiveBayes
python run.py --model NaiveBayes --feature-selection variance
python run.py --model NaiveBayes --feature-selection select_from_model --select-from svc
python run.py --model NaiveBayes --feature-selection select_from_model --select-from extra_trees

python run.py --model LogisticRegression
python run.py --model LogisticRegression --feature-selection variance
python run.py --model LogisticRegression --feature-selection select_from_model --select-from svc
python run.py --model LogisticRegression --feature-selection select_from_model --select-from extra_trees

python run.py --model GCN --h-feats 200
python run.py --model GCN --feature-selection variance --h-feats 200
python run.py --model GCN --feature-selection select_from_model --select-from svc --h-feats 200
python run.py --model GCN --feature-selection select_from_model --select-from extra_trees --h-feats 200

python run.py --model GCN --undirected --h-feats 200
python run.py --model GCN --feature-selection variance --undirected --h-feats 200
python run.py --model GCN --feature-selection select_from_model --select-from svc --undirected --h-feats 200
python run.py --model GCN --feature-selection select_from_model --select-from extra_trees --undirected --h-feats 200

python run.py --model GraphSAGE --h-feats 200
python run.py --model GraphSAGE --feature-selection variance --h-feats 200
python run.py --model GraphSAGE --feature-selection select_from_model --select-from svc --h-feats 200
python run.py --model GraphSAGE --feature-selection select_from_model --select-from extra_trees --h-feats 200

python run.py --model GraphSAGE --undirected --h-feats 200
python run.py --model GraphSAGE --feature-selection variance --undirected --h-feats 200
python run.py --model GraphSAGE --feature-selection select_from_model --select-from svc --undirected --h-feats 200
python run.py --model GraphSAGE --feature-selection select_from_model --select-from extra_trees --undirected --h-feats 200

python run.py --model GCN --h-feats 400
python run.py --model GCN --feature-selection variance --h-feats 400
python run.py --model GCN --feature-selection select_from_model --select-from svc --h-feats 400
python run.py --model GCN --feature-selection select_from_model --select-from extra_trees --h-feats 400

python run.py --model GCN --undirected --h-feats 400
python run.py --model GCN --feature-selection variance --undirected --h-feats 400
python run.py --model GCN --feature-selection select_from_model --select-from svc --undirected --h-feats 400
python run.py --model GCN --feature-selection select_from_model --select-from extra_trees --undirected --h-feats 400

python run.py --model GraphSAGE --h-feats 400
python run.py --model GraphSAGE --feature-selection variance --h-feats 400
python run.py --model GraphSAGE --feature-selection select_from_model --select-from svc --h-feats 400
python run.py --model GraphSAGE --feature-selection select_from_model --select-from extra_trees --h-feats 400

python run.py --model GraphSAGE --undirected --h-feats 400
python run.py --model GraphSAGE --feature-selection variance --undirected --h-feats 400
python run.py --model GraphSAGE --feature-selection select_from_model --select-from svc --undirected --h-feats 400
python run.py --model GraphSAGE --feature-selection select_from_model --select-from extra_trees --undirected --h-feats 400