import distance

def extract_length_features(row):
    q1=row['question1']
    q2=row['question2']
    length_features=[0.0]*3
    
    q1_tokens=q1.split()   #question1 token
    q2_tokens=q2.split()   #question2 token
    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return length_features
    
    length_features[0]=abs(len(q1_tokens) - len(q2_tokens))            #absolute length
    length_features[1]=(len(q1_tokens) + len(q2_tokens))/2             #average token length
    strs=list(distance.lcsubstrings(q1, q2))                           #longest substring
    length_features[2]=len(strs) / (min(len(q1), len(q2)) + 1)         #longest substring ratio
    
    return length_features