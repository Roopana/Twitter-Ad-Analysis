#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:05:54 2020

@author: vcroopana
"""
import pandas as pd
from GloveSimilarity import GloveSimilarity
from DataPreprocessor import DataPreprocessor
from ConfusionMatrixUtility import ConfusionMatrixUtility

class Caller():
    annotations_data = None
    man_ann_data = None
    
    def compareTweetAndAds(self, tweet, annotations_data, model):
        ad_keywords = annotations_data['keywords_clean']
        ad_sim = []
        for ad in ad_keywords:
            compare_res = GloveSimilarity.compare(tweet.lower(), ad.lower(), model)
            ad_sim.append(compare_res)
        return ad_sim
    
    def get_clean_ad_tweet_data(self, tweet_file, ad_file):
        annotations_data = pd.read_csv(ad_file, index_col=0) 
        annotations_data['Keywords'] = annotations_data['Brand Name']\
                                        .str.cat(annotations_data['Ad Name'], sep=" ")\
                                        .str.cat(annotations_data['KeyTerms_Edited'], sep=" ")
        df = annotations_data.drop_duplicates()
        print(df.shape)
        man_ann_data = pd.read_csv(tweet_file)    
        
        annotations_data['keywords_clean'] = annotations_data['Keywords'].apply(lambda ad: DataPreprocessor.cleanTweet(ad))
        man_ann_data['tweet_clean'] = man_ann_data['tweet_text'].apply(lambda twt: DataPreprocessor.cleanTweet(twt))
        
        return annotations_data, man_ann_data
        
    def computeGloveSim(self):
        # glove_avg_sim_df = pd.DataFrame(columns = annotations_data['Ad Name'])
        filename = '/Users/vcroopana/gensim-data/GoogleNews-vectors-negative300.bin'
        model = GloveSimilarity.getUnPrunedWord2VecModel(filename)
        print("Extracted unpruned model from file")
        gloveObj = GloveSimilarity()
        return gloveObj.computeGloveSimilarity(self.annotations_data, self.man_ann_data, model)
        # self.man_ann_data.to_csv("/Users/vcroopana/Downloads/summer2020/superbowl/mann_ann_sb_temp.csv")
        
    def computeGlovePOSSim(self):
        gloveObj = GloveSimilarity()
        path_glove_str = '/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d.txt'
        path_w2v_str = '/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d_w2v.txt'
        model = GloveSimilarity.getTwitterDataModel(path_glove_str, path_w2v_str)
        print('Extracted Twitter Glove model')
        return gloveObj.computeGlovePOSSimilarity(self.annotations_data, self.man_ann_data, model)
    
    def computeGlovePOSSynSim(self):
        gloveObj = GloveSimilarity()
        path_glove_str = '/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d.txt'
        path_w2v_str = '/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d_w2v.txt'
        model = GloveSimilarity.getTwitterDataModel(path_glove_str, path_w2v_str)
        print('Extracted Twitter Glove model')
        return gloveObj.computeGlovePOSSynSimilarity(self.annotations_data, self.man_ann_data, model)
    
    def computeGlovePOSCrossSim(self, posFlag):
        gloveObj = GloveSimilarity()
        path_glove_str = '/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d.txt'
        path_w2v_str = '/Users/vcroopana/gensim-data/glove/glove.twitter.27B.200d_w2v.txt'
        model = GloveSimilarity.getTwitterDataModel(path_glove_str, path_w2v_str)
        print('Extracted Twitter Glove model')
        return gloveObj.compute_glove_pos_cross_similary(self.annotations_data, self.man_ann_data, model, posFlag)
        
    
    def mergeSimOpManAnnInp(self, nlargest, data):
        # nlargest = 5
        # data = glove_pos_sim
        result_con = ConfusionMatrixUtility.getTopNSimAds(nlargest, data)
        ## merge mann ann data and sim result
        man_ann_data_glove_pos = pd.concat([self.man_ann_data, result_con], axis =1)
        return man_ann_data_glove_pos

    def __init__(self):   
        
        tweet_file = '/Users/vcroopana/Downloads/summer2020/superbowl/mann_ann_sb.csv'
        ad_file = '/Users/vcroopana/Downloads/summer2020/superbowl/ip/SB_ad_annotations.csv'
        
        self.annotations_data, self.man_ann_data = self.get_clean_ad_tweet_data(tweet_file, ad_file)
        print('Extracted Tweets and ads from input files')
    
caller = Caller()

####### Compute Glove sim using GoogleNews-vectors-negative300 model

glove_sim = caller.computeGloveSim()
merged = caller.mergeSimOpManAnnInp(5,glove_sim)
print(merged.head(5))
merged['conf_matrix'] = merged.apply(lambda x: ConfusionMatrixUtility.get_conf_matrix(x['ad_manual'], x['top1_ad'], 
                                                          x['top1'], 0.8), axis =1)
# man_ann_data_glove_pos['conf_matrix'] = man_ann_data_glove_pos.apply(lambda x: get_conf_matrix_2(x['ad_manual'], x['top1_ad'], 
#                             x['top2_ad'], x['top3_ad'], x['top4_ad'], x['top5_ad']), axis =1)
ConfusionMatrixUtility.computeAccuracy(merged)
merged.to_csv("/Users/vcroopana/Downloads/summer2020/superbowl/sim_glove_0.8.csv")

###### Compute Glove sim using POS tags and glove.twitter.27B.200d model

# glove_pos_sim = caller.computeGlovePOSSim()
# glove_pos_sim.head(5)
# merged = caller.mergeSimOpManAnnInp(5,glove_pos_sim)
# print(merged.head(5))
# merged['conf_matrix'] = merged.apply(lambda x: ConfusionMatrixUtility.get_conf_matrix(x['ad_manual'], x['top1_ad'], 
#                                                           x['top1'], 0.8), axis =1)

# ConfusionMatrixUtility.computeAccuracy(merged)
# merged.to_csv("/Users/vcroopana/Downloads/summer2020/superbowl/man_ann_data_sb_glove_pos.csv")

###### Compute Glove sim using POS tags, synonymns and glove.twitter.27B.200d model

# glove_pos_syn_sim = caller.computeGlovePOSSynSim()
# glove_pos_syn_sim.head(5)
# merged_glove_pos_syn = caller.mergeSimOpManAnnInp(5,glove_pos_syn_sim)
# print(merged_glove_pos_syn.head(5))
# merged_glove_pos_syn['conf_matrix'] = merged_glove_pos_syn.apply(lambda x: ConfusionMatrixUtility.get_conf_matrix(x['ad_manual'], x['top1_ad'], 
#                                                           x['top1'], 0.8), axis =1)

# ConfusionMatrixUtility.computeAccuracy(merged_glove_pos_syn)
# merged_glove_pos_syn.to_csv("/Users/vcroopana/Downloads/summer2020/superbowl/sim_glove_pos_syn.csv")


###### Cross Similarity using Glove and optional POS

# glove_pos_cross_sim = caller.computeGlovePOSCrossSim(False)

# merged_glove_pos_cross_sim = caller.mergeSimOpManAnnInp(5,glove_pos_cross_sim)
# print(merged_glove_pos_cross_sim.head(5))
# merged_glove_pos_cross_sim['conf_matrix'] = merged_glove_pos_cross_sim.apply(lambda x: ConfusionMatrixUtility.get_conf_matrix(x['ad_manual'], x['top1_ad'], 
#                                                           x['top1'], 0.8), axis =1)

# ConfusionMatrixUtility.computeAccuracy(merged_glove_pos_cross_sim)
# merged_glove_pos_cross_sim.to_csv("/Users/vcroopana/Downloads/summer2020/superbowl/sim_glove_pos_cross_python.csv")






