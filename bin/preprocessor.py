# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:57:13 2020

@author: Anaji
"""
import pandas as pd

class PreprocessData:
    """
    Module for preprocessing data
    """
    def preprocess(self,df):
        """
        Preprocess dataframe 
        Input:
            df = dataframe
        Output:
            df = cleaned dataframe
        """
        # Drop unusefull columns 
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],
                axis=1, inplace=True)
        # Column class map with for ham = 0 and spam = 1
        df['label'] = df['class'].map(
                                {'ham': 0, 'spam': 1})
        return df
    