from fuzzywuzzy import fuzz
def extract_fuzzy_features(row):
    q1=row['question1']
    q2=row['question2']
    fuzzy_features=[0.0]*4
    
    fuzzy_features[0]=fuzz.QRatio(q1, q2)              #fuzzy ration
    fuzzy_features[1]=fuzz.partial_ratio(q1, q2)       #fuzzy partial_ratio
    fuzzy_features[2]=fuzz.token_sort_ratio(q1, q2)    #token sort ratio
    fuzzy_features[3]=fuzz.token_set_ratio(q1, q2)     #token set ratio

    return fuzzy_features