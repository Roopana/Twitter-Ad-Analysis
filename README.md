# Twitter-Ad-Analysis

## Objective
Identify tweets that refer to commercials played during super bowl 2020 game given a corpus of tweets and context of advertisement videos screened during the game. 

## Methods used
### Bag of Words
- Glove
- Word2Vec

### Topic Modelling
Topic Modelling is an unsupervised technique to identify topics in a given text corpus based on joint distribution of words. 
- LDA 
- Guided LDA
- [Gibbs Sampling algorithm for the Dirichlet Multinomial Mixture model](http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf) for short text clustering (GSDMM)

## BERT
 BERT stands for Bidirectional Encoder Representation from Transformers. It captures text context unlike bag of words and topic modelling approaches. BERT is trained for \
 1. Masked Language modelling and 
 2. Next Sentence prediction tasks. 
 
 It considers all words of an input sentence simultaneosuly and then uses an _attention_ mechanism to develop contextual meaning of words. The attention heads help to identify the words that contribute the most in a sentence. 

_BERT-base_ model has 12 llayers with 12 attention heads
_BERT-large_ model has 24 layers with 16 attention heads. 

In this project __BERT-base model__ is used as it is sufficient for the current problem

## Joint learning with BERT

Stand alone BERT did not perform well in identifying all the ~60 ads present due to sparse nature of data. Stratified sampling during training did not help as expected. In order to increase the data signal each sentence in the training data is appended with ad context and annotated as whether ad related or not. This exploded the dataset to __NK__ where __N__ is the initial number of tweets and __K__ is the number of ad classes. This method gave better performance than previous models and classical models - Logistic Regression, SVM and Naive Bayes TF-IDF. 

Below are the results for each model:



