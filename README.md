# Graph Embeddings: Node Classification and Link Prediction

**Semester 2 — Massive Graph Management and Analytics (MGMA)**  
*Completed by Olha Baliasina and Samuel Chapuis*

We compare 9 embedding methods across 3 structurally different graphs, evaluating on node classification and link prediction. Each unsupervised embedding is tested with multiple downstream predictors to disentangle what the embedding learns from what the classifier exploits.

## Datasets

| | HepTh | Amazon | LastFM Asia |
|---|---|---|---|
| Domain | Academic collaboration | Product co-purchasing | Social network |
| Nodes | ~10K | ~335K | ~7.6K |
| Edges | ~22K | ~926K | ~27K |
| Labels | 7 ArXiv subject class | 20 top product communities | 18 user country |

## Methods

**Shallow embeddings:** DeepWalk, Node2Vec (BFS-tuned), Spectral  
**GNNs (end-to-end):** GCN, GraphSAGE  
**Self-supervised:** DGI (Deep Graph Infomax)  
**KG-style:** TransE, DistMult  
**Generative:** VGAE  
**Novel:** GRU-Walk — replaces Skip-Gram with a GRU encoder over random walks, capturing sequential dependencies that word2vec-style objectives miss

## Evaluation

- **Node classification:** F1-macro with LogReg, Random Forest, and end-to-end GNN classifiers (60/20/20 stratified split)
- **Link prediction:** AUC, MRR, Hits@10 with dot-product and (for some methods) MLP decoders (10% test, 5% val edge holdout, negative sampling)
- **Visualization:** UMAP projections of embedding spaces colored by ground-truth labels

## Key Findings

1. **The downstream predictor matters as much as the embedding** — switching LogReg → Random Forest on the same Node2Vec embedding can jump F1 by 17+ points.
2. **GRU-Walk** shows competitive performance with shallow methods while learning richer walk representations.
3. **VGAE** dominates link prediction (it's trained on that objective) but doesn't transfer as well to node classification.
4. **Feature-rich graphs** (LastFM with 7842-dim artist vectors) strongly favor GNNs over structure-only embeddings.

## Stack

Python, PyTorch, PyTorch Geometric, node2vec, NetworkX, scikit-learn, UMAP
