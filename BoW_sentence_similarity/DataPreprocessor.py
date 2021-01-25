#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:59:11 2020

@author: vcroopana
"""
from nltk.corpus import stopwords
import re
import en_core_web_sm  # This is the default model ( vocabulary, syntax and entity)
nlp = en_core_web_sm.load()

class DataPreprocessor():
    
    stop = stopwords.words('english')
    print(len(stop))
    
    def removeMentions(text):
    
        textBeforeMention = text.partition("@")[0]
        textAfterMention = text.partition("@")[2]
        textAfterMention =  re.sub(r':', '', textAfterMention) #cadillac join the 31k
        # tHandle = textAfterMention.partition(" ")[0].lower() #cadillac    
        text = textBeforeMention+ " " + textAfterMention  
        return text

    
    def cleanTweet(strinp):
        strinp = re.sub(r'RT', "", strinp) # Remove RT
        strinp = strinp.lower()
        
        stop_removed_list = [word for word in strinp.split() if word not in (DataPreprocessor.stop)]
        stop_removed = ' '.join([str(elem) for elem in stop_removed_list])    
        text = re.sub('https?://[A-Za-z0-9./]+', ' ', stop_removed) # Remove URLs
        text = DataPreprocessor.removeMentions(text)
        text = re.sub('[^\x00-\x7F]+', ' ', text) # Remove non-ASCII chars.
        
        # remove punctuations except '_'
        punctuation = ['(', ')', '[',']','?', ':', ':', ',', '.', '!', '/', '"', "'", '@', '#', '&']
    #     text = re.sub('[^a-zA-Z]', ' ', text) # remove all other than alphabet chars 
        text = "".join((char for char in text if char not in punctuation))
        
    #     text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # remove all single characters     
        stop_removed_l = [word for word in text.split() if word not in (DataPreprocessor.stop)]
        stop_removed = ' '.join([str(elem) for elem in stop_removed_l]) 
        return stop_removed
    
    def preprocessor(text):
        if isinstance((text), (str)):
            text = re.sub('<[^>]*>', '', text)
            text = re.sub('[\W]+', '', text.lower())
            text = re.sub('[^\x00-\x7F]+', ' ', text) # removes non ascii chars
            text = re.sub('https?://[A-Za-z0-9./]+', ' ', text) # Remove URLs
            # remove punctuations except '_'
            punctuation = ['(', ')', '[',']','?', ':', ':', ',', '.', '!', '/', '"', "'", '@', '#', '&']
    #     text = re.sub('[^a-zA-Z]', ' ', text) # remove all other than alphabet chars 
            text = "".join((char for char in text if char not in punctuation))
            text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # remove all single characters   
            return text
        
        if isinstance((text), (list)):
            return_list = []
            for i in range(len(text)):
                temp_text = re.sub('<[^>]*>', '', text[i])
                temp_text = re.sub('[\W]+', '', temp_text.lower())
                return_list.append(temp_text)
            return(return_list)
        else:
            pass
        
    def lemma_token(text):
        text = text.lower()  # to take care of capital case word in glove
        # spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        # using spacy tokenizer 
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

    def find_emo(text):
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
        return emoticons