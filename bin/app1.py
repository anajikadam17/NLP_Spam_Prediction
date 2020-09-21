# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:37:35 2020

@author: Anaji
"""

import pickle

filename1 = 'nlp_model.pkl'
clf = pickle.load(open(filename1, 'rb'))
filename2 = 'tranform.pkl'
cv = pickle.load(open(filename2,'rb'))
message = 'you won price'
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)

print(my_prediction)