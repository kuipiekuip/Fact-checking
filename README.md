# Fact-checking

This repo contains a collection of Kaggle notebooks that were used to train the models in our paper. 

- `train` and `test` contains skeleton code for training and testing the NLI models, which were adapted to suit different models and datasets for our experiments
- `bm25` contains the code used to generate the top-100 snippets for each claim
- `claimdecomp` contains code used to train the BART model to decompose the claims in QuanTemp, and the reranking code to extract the top-5 snippets from the bm25 top 100
- `temporal-reranking` contains the code used to rerank the top-5 snippets from the bm25 top-100 by taking into account temporal information
- `strategyqa` contains the code used to invistigate the use of a model trained on StrategyQA in order to decompose questions and answer them iteratively using a Deberta model trained on SQuAD