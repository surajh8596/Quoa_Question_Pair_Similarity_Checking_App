from nltk.corpus import stopwords

def extract_token_features(row):
    q1=row['question1']
    q2=row['question2']
    SAFE_DIV=0.0001
    STOP_WORDS=stopwords.words("english")   #Stopwords
    token_features=[0.0]*8
    
    q1_tokens=q1.split()    #tokens in question1
    q2_tokens=q2.split()    #tokens in question2
    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])     #non-stopwords in question1
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])     #non-stopwords in question2
    
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])         #stopwords in question1
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])         #stopwords in question2
    
    common_word_count = len(q1_words.intersection(q2_words))                   #non-stopword count
    common_stop_count = len(q1_stops.intersection(q2_stops))                   #common stopword count
    common_token_count=len(set(q1_tokens).intersection(set(q2_tokens)))        #common token count
    
    token_features[0]=common_word_count/(min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1]=common_word_count/(max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2]=common_stop_count/(min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3]=common_stop_count/(max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4]=common_token_count/(min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5]=common_token_count/(max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6]=int(q1_tokens[-1]==q2_tokens[-1])   #last word same or not
    token_features[7]=int(q1_tokens[0]==q2_tokens[0])     #first word same or not
    
    return token_features