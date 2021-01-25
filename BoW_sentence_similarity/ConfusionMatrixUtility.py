#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:41:48 2020

@author: vcroopana
"""
import numpy as np
import pandas as pd

class ConfusionMatrixUtility():

    def getTopNSimAds(nlargest, data):
        
        order = np.argsort(-data.values, axis=1)[:, :nlargest]
        result = pd.DataFrame(data.columns[order], 
                              columns=['top{}_ad'.format(i) for i in range(1, nlargest+1)],
                              index= data.index)
    
        order_vals = np.sort(-data.values, axis=1)[:, :nlargest]
        result_vals = pd.DataFrame(-order_vals,
                                  columns = ['top{}'.format(i) for i in range(1, nlargest+1)],
                                  index= data.index)
        
        result_con = pd.concat([result_vals, result], axis =1)
        return result_con
    
    def get_conf_matrix(ad_manual, ad_algo, ad_prob, thresh_prob):
        res = ""
        ad_manual = ad_manual.lower()
        ad_algo = ad_algo.lower()
    
        if(ad_manual == ad_algo and ad_prob >= thresh_prob):
            res = 'TP'
        elif(ad_manual == ad_algo and ad_prob < thresh_prob):
            res = 'FN'
        elif(ad_manual=='none' and ad_manual != ad_algo and ad_prob > thresh_prob):
            res = 'FP'
        elif(ad_manual=='none' and ad_manual != ad_algo and ad_prob < thresh_prob):
            res = 'TN'
        elif(ad_manual!='none' and ad_manual!= ad_algo):
            res = 'FN'
        return res
    
    def get_conf_matrix_2(ad_manual, ad_algo, ad_algo_2, ad_algo_3, ad_algo_4, ad_algo_5):
        res = ""
        ad_manual = ad_manual.lower()
        ad_algo = ad_algo.lower()
        ad_algo_2 = ad_algo_2.lower()
        ad_algo_3 = ad_algo_3.lower()
        ad_algo_4 = ad_algo_4.lower()
        ad_algo_5 = ad_algo_5.lower()
        
        if(ad_manual == ad_algo or ad_manual == ad_algo_2 or ad_manual == ad_algo_3 or ad_manual == ad_algo_4
          or ad_manual == ad_algo_5):
            res = 'TP'
        elif(ad_manual=='none'):
            res = 'FP'
        elif(ad_manual!='none' and ad_manual!= ad_algo and ad_manual!= ad_algo_2 and ad_manual!= ad_algo_3
            and ad_manual!= ad_algo_4 and ad_manual!= ad_algo_5):
            res = 'FN'
        elif(ad_manual!='none' and ad_manual!= ad_algo and ad_manual!= ad_algo_2 and ad_manual!= ad_algo_3
            and ad_manual!= ad_algo_4 and ad_manual!= ad_algo_5):
            res = 'TN'
        return res
    
    def computeAccuracy(result):    
        n_tp = result[result['conf_matrix'] == 'TP'].shape[0]
        n_fp = result[result['conf_matrix'] == 'FP'].shape[0]
        n_fn = result[result['conf_matrix'] == 'FN'].shape[0]
        n_tn = result[result['conf_matrix'] == 'TN'].shape[0]    
        print("n_tp:" + str(n_tp)+ " n_fp:" + str(n_fp)+ " n_fn:" + str(n_fn) + " n_tn:" + str(n_tn))
        precision = n_tp/(n_tp+ n_fp)
        recall = n_tp/(n_tp+ n_fn)
        f_measure = (2*precision*recall)/ (precision+recall)
    
        return precision, recall, f_measure


