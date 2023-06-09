import streamlit as st
import pandas as pd
import pickle
import os
import contractions
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np
from xgboost import XGBClassifier

st.title("Quora Question Pair Similarity")

#Take input from user
question1=st.text_input("Enter Question 1")
question2=st.text_input("Enter Question 2")

balanced=pd.DataFrame({'question1':question1, 'question2':question2}, index=[0])

#preprocess the data
from text_clean import clean_test_for_sample_data #own module
balanced['question1']=balanced['question1'].apply(clean_test_for_sample_data)
balanced['question2']=balanced['question2'].apply(clean_test_for_sample_data)

#Extract basic features
# 1. Question length
balanced['que1_len']=balanced['question1'].str.len()
balanced['que2_len']=balanced['question2'].str.len()

# 2. Number of words
balanced['que1_num_words'] =balanced['question1'].apply(lambda sent: len(sent.split()))
balanced['que2_num_words'] =balanced['question2'].apply(lambda sent: len(sent.split()))

# 3. Total words in both question
def total_words(row):
    q1_w=set(map(lambda x: x.lower().strip(), row['question1'].split()))
    q2_w=set(map(lambda x: x.lower().strip(), row['question2'].split()))
    return len(q1_w) + len(q2_w)

balanced['total_words']=balanced.apply(total_words, axis=1)

# 4. Common words in both questions
def common_words(row):
    q1_w=set(map(lambda x: x.lower().strip(), row['question1'].split()))
    q2_w=set(map(lambda x: x.lower().strip(), row['question2'].split()))
    return len(q1_w)&len(q2_w)

balanced['common_words']=balanced.apply(common_words, axis=1)

# 5. Word sharing
balanced['shared_words'] = round(balanced['common_words']/balanced['total_words'], 2)

#Extract token features
from token_features import extract_token_features #own module

token_features=balanced.apply(extract_token_features, axis=1)
balanced["cwc_min"]=list(map(lambda x: x[0], token_features))
balanced["cwc_max"]=list(map(lambda x: x[1], token_features))
balanced["csc_min"]=list(map(lambda x: x[2], token_features))
balanced["csc_max"]=list(map(lambda x: x[3], token_features))
balanced["ctc_min"]= list(map(lambda x: x[4], token_features))
balanced["ctc_max"]=list(map(lambda x: x[5], token_features))
balanced["last_word_eq"]=list(map(lambda x: x[6], token_features))
balanced["first_word_eq"]=list(map(lambda x: x[7], token_features))

#Extract length features
from length_features import extract_length_features

length_features=balanced.apply(extract_length_features, axis=1)
balanced['abs_len_diff']=list(map(lambda x: x[0], length_features))
balanced['mean_len']=list(map(lambda x: x[1], length_features))
balanced['long_substr_ratio']=list(map(lambda x: x[2], length_features))

#Extract Fuzzy features
from fuzzy_features import extract_fuzzy_features

fuzzy_features=balanced.apply(extract_fuzzy_features, axis=1)
balanced['fuzz_ratio']=list(map(lambda x: x[0], fuzzy_features))
balanced['fuzz_partial_ratio']=list(map(lambda x: x[1], fuzzy_features))
balanced['token_sort_ratio']=list(map(lambda x: x[2], fuzzy_features))
balanced['token_set_ratio']=list(map(lambda x: x[3], fuzzy_features))

#BERT to document vectorization
model=SentenceTransformer('all-MiniLM-L6-v2')

#Vectorize
doc_vector1=list(balanced['question1'].apply(model.encode))
doc_vector2=list(balanced['question2'].apply(model.encode))

#Extracted features
extracted_features=['que1_len', 'que2_len',
       'que1_num_words', 'que2_num_words', 'total_words', 'common_words',
       'shared_words', 'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min',
       'ctc_max', 'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
       'long_substr_ratio', 'fuzz_ratio', 'fuzz_partial_ratio',
       'token_sort_ratio', 'token_set_ratio']
extracted_features=balanced[extracted_features]

#Concate vectors and extracted features
extracted_features_array=np.array(extracted_features)
embedded_document_array=np.hstack((doc_vector1, doc_vector2))
input_data=np.hstack((extracted_features_array, embedded_document_array))

#Load Model
FILE_DIR=os.path.dirname(os.path.abspath(__file__))
dir_of_interest=os.path.join(FILE_DIR, "resources")
model_path=os.path.join(dir_of_interest, 'xgboost.pkl')
xgb=pickle.load(open(model_path, 'rb'))

#Predict
prediction=(xgb.predict(input_data))[0]
button=st.button("Check Question Duplicate")
if button:
    if prediction==1:
        st.subheader("Questions Pair is Duplicate")
    else:
        st.subheader("Questions Pair is Not Duplicate")
