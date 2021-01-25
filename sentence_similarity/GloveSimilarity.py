#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:48:27 2020

@author: vcroopana
"""
import numpy as np
import en_core_web_sm  # This is the default model ( vocabulary, syntax and entity)
nlp = en_core_web_sm.load()
from nltk.data import find
import gensim 
from DataPreprocessor import DataPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import os
        
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

class GloveSimilarity():
    # we will try to return lemma list of only the allowed pos in the text
    def lemma_token_pos(self, text, allowed_pos):
        text = text.lower()  # to take care of capital case word in glove
        doc=nlp(text)
        lemma_list = []
        for token in doc:
            if token.is_stop is False:
                #if (token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'adv'):
                if (token.pos_ in allowed_pos):
                    token_preprocessed = DataPreprocessor.preprocessor(token.lemma_)
                    if token_preprocessed != '':
                         lemma_list.append(token_preprocessed)
                         
        return lemma_list
    
    def getSimilarityDf(self, annotations_data, ads_lemma_tokens, tweets_lemma_tokens, model):
        ad_id = 1
        glove_pos_sim_df = pd.DataFrame(columns = annotations_data['Ad Name'])
        
        for ad in ads_lemma_tokens:
            glove_pos_sim_col =[]
            avg_vec_pos_ad = self.average_vec(ad, model)
            print("computing sim for ad:" + str(ad_id))
            for i in range(0, len(tweets_lemma_tokens)): 
                avg_vec_pos_twt = self.average_vec(tweets_lemma_tokens[i], model)     
                if  avg_vec_pos_twt.shape == avg_vec_pos_ad.shape:
                    glove_pos_sim = cosine_similarity(avg_vec_pos_twt,avg_vec_pos_ad).sum()
                else:
                    glove_pos_sim = 0
                glove_pos_sim_col.append(np.round(glove_pos_sim,3))
            
            glove_pos_sim_df[annotations_data['Ad Name'][ad_id]] = glove_pos_sim_col
            ad_id = ad_id + 1
            
        return glove_pos_sim_df
            
    def computeGlovePOSSimilarity(self, annotations_data, man_ann_data, model):
        ##### Data Init
        ad_keywords_clean = annotations_data['keywords_clean']
        
        ###### Preprocessing - tokenize and filter POS
        allowed_pos = ('NOUN', 'VERB', 'ADJ', 'adv')
        tweets_lemma_tokens = man_ann_data['tweet_clean'].apply(lambda x: self.lemma_token_pos(x, allowed_pos))
        print("Calculated pos tokens of tweets")
        ads_lemma_tokens = ad_keywords_clean.apply(lambda x: self.lemma_token_pos(x, allowed_pos))
        print("Calculated pos tokens of ads")
        
        ###### Similarity calculation
        glove_pos_sim_df = self.getSimilarityDf(annotations_data, ads_lemma_tokens, tweets_lemma_tokens, model)

        print("Glove POS Similarities Calculated")
        return glove_pos_sim_df
    
    def computeGlovePOSSynSimilarity(self, annotations_data, man_ann_data, model):
        ##### Data Init
        ad_keywords_clean = annotations_data['keywords_clean']
        
        ###### Preprocessing - tokenize and filter POS
        allowed_pos = ('NOUN', 'VERB', 'ADJ', 'adv')
        tweets_lemma_tokens = man_ann_data['tweet_clean'].apply(
            lambda x: self.lemma_token_pos_synonyms(x, allowed_pos, model))
        print("Calculated pos tokens of tweets")
        ads_lemma_tokens = ad_keywords_clean.apply(
            lambda x: self.lemma_token_pos_synonyms(x, allowed_pos, model))
        print("Calculated pos tokens of ads")
        
        ###### Similarity calculation
        glove_pos_syn_df = self.getSimilarityDf(annotations_data, ads_lemma_tokens, tweets_lemma_tokens, model)

        print("Glove POS Synonymn Similarities Calculated")
        return glove_pos_syn_df
    
    def computeGloveSimilarity(self, annotations_data, man_ann_data, model):
        ##### Data Init
        ad_keywords_clean = annotations_data['keywords_clean']
        
        ###### Preprocessing - tokenize
        tweets_lemma_tokens = man_ann_data['tweet_clean'].apply(lambda x: self.lemma_token(x))
        print("Calculated pos tokens of tweets")
        ads_lemma_tokens = ad_keywords_clean.apply(lambda x: self.lemma_token(x))
        print("Calculated pos tokens of ads")
        
        ###### Similarity calculation
        glove_sim_df = self.getSimilarityDf(annotations_data, ads_lemma_tokens, tweets_lemma_tokens, model)
        print("Glove Similarities Calculated")
        return glove_sim_df
    
    def compute_glove_pos_cross_similary(self, annotations_data, man_ann_data, model, flag):
        ad_keywords_clean = annotations_data['keywords_clean']
        tweets_lemma_tokens =[]
        ads_lemma_tokens = []
        if flag==True:
            allowed_pos = ('NOUN', 'VERB', 'ADJ', 'adv')
            tweets_lemma_tokens = man_ann_data['tweet_clean'].apply(lambda x: self.lemma_token_pos(x, allowed_pos))
            print("Calculated pos tokens of tweets")
            ads_lemma_tokens = ad_keywords_clean.apply(lambda x: self.lemma_token_pos(x, allowed_pos))
            print("Calculated pos tokens of ads")
        else:
            tweets_lemma_tokens = man_ann_data['tweet_clean'].apply(lambda x: self.lemma_token(x))
            print("Calculated pos tokens of tweets")
            ads_lemma_tokens = ad_keywords_clean.apply(lambda x: self.lemma_token(x))
            print("Calculated pos tokens of ads")
    
        ad_id = 1
        glove_pos_sim_df = pd.DataFrame(columns = annotations_data['Ad Name'])
            
        for ad in ads_lemma_tokens:
            ad_token_filtd = [w for w in ad if w in model.vocab]
            word_vecs_ad = [model.word_vec(w) for w in ad_token_filtd]
            glove_pos_sim_col =[]
            if(len(ad_token_filtd) == 0):
                glove_pos_sim_col = [0] * len(tweets_lemma_tokens) 
            else:
                print("computing sim for ad:" + str(ad_id))
                for i in range(0, len(tweets_lemma_tokens)): 
                    tweets_lemma_filtd = [ w for w in tweets_lemma_tokens[i] if w in model.vocab ]
                    if(len(tweets_lemma_filtd)!=0):
                        pairwise_cos_similarity = 0
                        cross_cos_similarity_12 = 0
                        for word in tweets_lemma_filtd:
                            word_vecs_tweet = model.word_vec(word).reshape(1, -1)
                            pairwise_cos_similarity = cosine_similarity(word_vecs_tweet ,word_vecs_ad).sum()
                            cross_cos_similarity_12 = cross_cos_similarity_12 + pairwise_cos_similarity 
                        norm_cross_cos_similarity = (cross_cos_similarity_12)/ (len(ad_token_filtd)*len(tweets_lemma_filtd)) 
    #                     print("norm_cross_cos_similarity : {}".format(norm_cross_cos_similarity))
                    else:
                        norm_cross_cos_similarity =0
                        
                    glove_pos_sim_col.append(np.round(norm_cross_cos_similarity,3))
                
            glove_pos_sim_df[annotations_data['Ad Name'][ad_id]] = glove_pos_sim_col
            ad_id = ad_id + 1
            
        return glove_pos_sim_df
    
    def lemma_token(self, text):
        text = text.lower()  # to take care of capital case word in glove
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        tokens = tokenizer(text)
        token_list = []
        lemma_list = []
        for token in tokens:
            if token.is_stop is False:
                token_preprocessed = DataPreprocessor.preprocessor(token.lemma_)
                if token_preprocessed != '':
                     lemma_list.append(token_preprocessed)
                     token_list.append(token.text)   
                     
        return lemma_list
    
    # we will try to return lemma list of only the POS in the text 
    # Extend lemma list with synonymns
    def lemma_token_pos_synonyms(self, text, allowed_pos, model):

        lemma_list = self.lemma_token_pos(text, allowed_pos)
        synonym_list = []

        for lemma in lemma_list:
            if lemma in model.vocab:
                tuple_list = model.most_similar(positive=[lemma], topn = 5)
                # print("tuple list:"+ str(tuple_list))
                for a_tuple in tuple_list:
                    synonym_list.append(a_tuple[0])
        
        lemma_list = lemma_list + synonym_list        
        # print("\n Extended lemma list : ")
        # print(lemma_list)
        # print("\n")
        return lemma_list
    
    def average_vec(self, words, model):
        #use unk word when word is not present in vocab to find a predesigned vector which is often the best vector
        word_vecs = [model.word_vec(w) if w in model.vocab else model.word_vec('unk') for w in words ]
        if(len(word_vecs) == 0):
            return (np.array(word_vecs).sum(axis=0)).reshape(1,-1)
        else:                
            return (np.array(word_vecs).sum(axis=0)/len(word_vecs)).reshape(1,-1)
    
    
    def getTwitterDataModel(path_glove_str, path_w2v_str):

        path_glove = os.path.abspath(path_glove_str) #'/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d.txt'
        path_w2v = os.path.abspath(path_w2v_str) #'/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d_w2v.txt'
        
        glove_file = datapath(path_glove)
        tmp_file = get_tmpfile(path_w2v)
        
        _ = glove2word2vec(glove_file, tmp_file)
        print('Done Glove to word2vec ')
        model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
        print('Loaded Word2Vec model')
        return model

    def getUnPrunedWord2VecModel(filename):
        # filename = '/Users/vcroopana/gensim-data/GoogleNews-vectors-negative300.bin'
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        return model
        
    def getPrunedWord2VecModel(path):
        # path = '/Users/vcroopana/nltk_data/models/word2vec_sample/pruned.word2vec.txt'
        word2vec_sample = str(find(path))
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
        return model
    
    compare = lambda a,b, model: cosine_similarity(GloveSimilarity.average_vec(a, model),
                                            GloveSimilarity.average_vec(b, model)).sum()
    
    compare_pos = lambda a,b, model: cosine_similarity(GloveSimilarity.average_vec_pos(a, model),
                                                GloveSimilarity.averagse_vec_pos(b, model)).sum()
    # ### outputs the average word2vec for words in this sentence
    # def average_vec_pos(self, words, model):
    #     #use unk word when word is not present in vocab to find a predesigned vector which is often the best vector
    #     word_vecs = [model.word_vec(w) if w in model.vocab else model.word_vec('unk') for w in words ]
    #     if(len(word_vecs) == 0):
    #         return (np.array(word_vecs).sum(axis=0)).reshape(1,-1)
    #     else:                
    #         return (np.array(word_vecs).sum(axis=0)/len(word_vecs)).reshape(1,-1)