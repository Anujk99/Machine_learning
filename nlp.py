# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:55:22 2020

@author: HP
"""

import nltk
from nltk.tokenize import  word_tokenize
text="mitrc is an engineering college in alwar and. jack,take admission in mitrc"
word_token=word_tokenize(text)
print(word_token)




from nltk.tokenize import  sent_tokenize
print(sent_tokenize(text))





text="NLP and NLU are comple and the data. the best"
t1=word_tokenize(text)
from nltk.util import bigrams
print(list(bigrams(t1)))




text="NLP and NLU are comple and the data. the best"
t1=word_tokenize(text)
from nltk.util import ngrams
print(list(ngrams(t1,4)))



#stemming
from nltk.stem import PorterStemmer
li=["sleeping","slept","sleeped"]
st=PorterStemmer()
for i in li:
    print(st.stem(i))



from nltk.corpus import stopwords
sw=stopwords.words("english")
print(sw)

text2="i want to market yesterday"
text2=word_tokenize(text2)

d=[]
n_ds=[]
for i in text2:
    if(i in sw):
        d.append(i) 
    else:
        n_ds.append(i)
print(d)
print(n_ds)




from nltk.stem import WordNetLemmatizer
w_l=WordNetLemmatizer()
text3="geese"
print(w_l.lemmatize(text3))






text4="the dog killed the bat"
t_w=word_tokenize(text4)

for i in t_w:
    print(nltk.pos_tag_sents(i))







