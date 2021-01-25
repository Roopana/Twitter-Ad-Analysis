# Twitter-Ad-Analysis

## Objective
Identify tweets that refer to commercials played during super bowl 2020 game given a corpus of tweets and context of advertisement videos screened during the game. 

## Methods used
### Bag of Words
- Glove
- Word2Vec

### Topic Modelling
Topic Modelling is an unsupervised technique to identify topics in a given text corpus based on joint distribution of words. 
- __LDA__ 
- __Guided LDA__
- [Gibbs Sampling algorithm for the Dirichlet Multinomial Mixture model](http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf) for short text clustering (GSDMM)

__LDA__ and __GSDMM__ are unsupervised generative probabilistic models which analyze topic composition in text corpora. When LDA is run on the tweets data set, it gives only a brief idea of topic distribution in the data. But most of the topics identified by LDA are not ad-related as ad-related tweets have a minority presence in our dataset collected in the context of the Superbowl. 

Gibbs Sampling algorithm for the Dirichlet Multinomial Mixture model (Yin and Wang 2014) is a variation of the LDA method that overcomes the sparse and high dimensional nature of short text data sets. GSDMM forms topic clusters iteratively using dirichlet dis- tribution and provides a good balance between homogeneity and completeness of clusters.

In the GSDMM implementation, we observed better performance than LDA. However both the approaches of LDA and GSDMM, rely on good representation of classes available in the training dataset, and hence these approaches do not work well in our setting where tweets talking about commercials are very less (1%) in the whole dataset.

## BERT
 BERT stands for Bidirectional Encoder Representation from Transformers. It captures text context unlike bag of words and topic modelling approaches. BERT is trained for
 1. Masked Language modelling and 
 2. Next Sentence prediction tasks. 
 
 It considers all words of an input sentence simultaneosuly and then uses an _attention_ mechanism to develop contextual meaning of words. The attention heads help to identify the words that contribute the most in a sentence. 

_BERT-base_ model has 12 layers with 12 attention heads
_BERT-large_ model has 24 layers with 16 attention heads

In this project __BERT-base model__ is used as it is sufficient for the current problem

## Joint learning with BERT

Stand alone BERT did not perform well in identifying all the ~60 ads present due to sparse nature of data. Stratified sampling during training did not help as expected. In order to increase the data signal each sentence in the training data is appended with ad context and annotated as whether ad related or not. This exploded the dataset to __NK__ where __N__ is the initial number of tweets and __K__ is the number of ad classes. This method gave better performance than previous models and classical models - Logistic Regression, SVM and Naive Bayes TF-IDF. 

## Results
Below are the results for each model:

![](results/classification_performance.png =24x48)

![](results/graph_classifier_performance.png =24x48)
