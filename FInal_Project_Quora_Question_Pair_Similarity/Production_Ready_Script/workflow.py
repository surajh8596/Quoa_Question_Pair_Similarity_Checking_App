from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
from prefect import task, flow
import re
import contractions
import distance
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix


@task
def load_data(path: str, unwanted_cols: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.drop(unwanted_cols, axis=1, inplace=True)
    return data

@task
def get_classes(target_data: pd.Series) -> List[str]:
    return list(target_data.unique())

@task
def drop_na_dup(dataframe):  #dropna and duplicate
    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates(inplace=True)
    return dataframe

@task
def clean_test_for_sample_data(text): #return clean text column
    text=str(text).lower()
    text=text.replace('%', ' percent')
    text=text.replace('$', ' dollar ')
    text=text.replace('₹', ' rupee ')
    text=text.replace('€', ' euro ')
    text=text.replace('@', ' at ')
    text=text.replace(',000,000,000 ', 'b ')
    text=text.replace(',000,000 ', 'm ')
    text=text.replace(',000 ', 'k ')
    text=re.sub(r'([0-9]+)000000000', r'\1b', text)
    text=re.sub(r'([0-9]+)000000', r'\1m', text)
    text=re.sub(r'([0-9]+)000', r'\1k', text)
    pattern=re.compile('\W')
    text=re.sub(pattern, ' ', text).strip()
    text=re.sub("<.*?>", "", text)
    text=contractions.fix(text)
    text=re.sub(" +", " ", text)
    return text

@task
def total_words(row): #Total words in both question
    q1_w=set(map(lambda x: x.lower().strip(), row['question1'].split()))
    q2_w=set(map(lambda x: x.lower().strip(), row['question2'].split()))
    return len(q1_w) + len(q2_w)

@task
def common_words(row): #Common words in both questions
    q1_w=set(map(lambda x: x.lower().strip(), row['question1'].split()))
    q2_w=set(map(lambda x: x.lower().strip(), row['question2'].split()))
    return len(q1_w)&len(q2_w)

@task
def extract_length_features(row): #tooken features
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

@task
def extract_token_features(row): #length features
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

@task
def extract_fuzzy_features(row): #fuzzy features
    q1=row['question1']
    q2=row['question2']
    fuzzy_features=[0.0]*4
    fuzzy_features[0]=fuzz.QRatio(q1, q2)              #fuzzy ration
    fuzzy_features[1]=fuzz.partial_ratio(q1, q2)       #fuzzy partial_ratio
    fuzzy_features[2]=fuzz.token_sort_ratio(q1, q2)    #token sort ratio
    fuzzy_features[3]=fuzz.token_set_ratio(q1, q2)     #token set ratio
    return fuzzy_features

@task
def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_train, 'Y_TRAIN': y_train, 'X_TEST': X_test, 'Y_TEST': y_test}

@task
def model_building(x_train: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, x_test: pd.Series) -> Any:
    with mlflow.start_run():   
        classifier=XGBClassifier()
        classifier.fit(x_train,y_train)
        
        y_pred=classifier.predict(x_test)
        acc=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)

        print("----------------------------------------------------------")
        print("Acuuracy={}\nPrecission={}\nRecall={}\nF1 Score={}".format(acc, precision, recall, f1))
        print("----------------------------------------------------------")
        print("Confusion Matrix=\n")
        sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, annot_kws={"fontsize":15}, linewidths=5, linecolor='blue',
                    yticklabels=["Non-Duplicate","Duplicate"], xticklabels=["Non-Duplicate","Duplicate"], cbar=None, fmt='g')
        plt.xlabel("Predicted",fontsize=18)
        plt.ylabel("Actual",fontsize=18)
        plt.show()
        print("----------------------------------------------------------")
        print("Out of {} testing values {} are mis-classified.".format(y_test.shape[0], sum(y_test!=y_pred)))
        print("----------------------------------------------------------")
        print("Classification Report=\n", classification_report(y_test,y_pred))
        print("----------------------------------------------------------")
        
        mlflow.log_metrics({"Accuracy":acc, "Precision":precision, "Recall":recall, "F1 Score":f1})
        mlflow.sklearn.log_model(classifier, artifact_path="models")
        return acc
    

# Workflow
@flow
def main(path: str):
    mlflow.set_experiment("Quora Question Pair Similarity")

    #define parameters
    TARGET_COL='is_duplicate'    
    UNWANTED_COLS=['id','qid1','qid2']
    TEST_DATA_RATIO=0.2
    DATA_PATH=path

    #Load data
    dataframe=load_data(path=DATA_PATH, unwanted_cols=UNWANTED_COLS)

    #dropna and duplicates
    drop_na_dup(dataframe)

    #clean_data
    dataframe['question1']=dataframe['question1'].apply(clean_test_for_sample_data)
    dataframe['question2']=dataframe['question2'].apply(clean_test_for_sample_data)

    #Extract features
    dataframe['total_words']=dataframe.apply(total_words, axis=1)
    dataframe['common_words']=dataframe.apply(common_words, axis=1)
    dataframe['que1_len']=dataframe['question1'].str.len() #length
    dataframe['que2_len']=dataframe['question2'].str.len()
    dataframe['que1_num_words'] =dataframe['question1'].apply(lambda sent: len(sent.split())) #words
    dataframe['que2_num_words'] =dataframe['question2'].apply(lambda sent: len(sent.split()))
    dataframe['shared_words'] = round(dataframe['common_words']/dataframe['total_words'], 2) #common words

    token_features=dataframe.apply(extract_token_features, axis=1) #token features
    dataframe["cwc_min"]=list(map(lambda x: x[0], token_features))
    dataframe["cwc_max"]=list(map(lambda x: x[1], token_features))
    dataframe["csc_min"]=list(map(lambda x: x[2], token_features))
    dataframe["csc_max"]=list(map(lambda x: x[3], token_features))
    dataframe["ctc_min"]= list(map(lambda x: x[4], token_features))
    dataframe["ctc_max"]=list(map(lambda x: x[5], token_features))
    dataframe["last_word_eq"]=list(map(lambda x: x[6], token_features))
    dataframe["first_word_eq"]=list(map(lambda x: x[7], token_features))

    length_features=dataframe.apply(extract_length_features, axis=1) #length features
    dataframe['abs_len_diff']=list(map(lambda x: x[0], length_features))
    dataframe['mean_len']=list(map(lambda x: x[1], length_features))
    dataframe['long_substr_ratio']=list(map(lambda x: x[2], length_features))

    fuzzy_features=dataframe.apply(extract_fuzzy_features, axis=1) #fuzzy features
    dataframe['fuzz_ratio']=list(map(lambda x: x[0], fuzzy_features))
    dataframe['fuzz_partial_ratio']=list(map(lambda x: x[1], fuzzy_features))
    dataframe['token_sort_ratio']=list(map(lambda x: x[2], fuzzy_features))
    dataframe['token_set_ratio']=list(map(lambda x: x[3], fuzzy_features))

    #dropna and duplicates
    drop_na_dup(dataframe)

    #Input and output
    y=dataframe[TARGET_COL]
    X=dataframe.drop([TARGET_COL], axis=1)
    classes = get_classes(target_data=y)

    #train test split
    train_test_dict=split_data(input_=X, output_=y, test_data_ratio=TEST_DATA_RATIO)
    
    #Creating vectors
    model=SentenceTransformer('all-MiniLM-L6-v2')
    x_train_doc_vector1=list(train_test_dict['X_TRAIN']['question1'].apply(model.encode))
    x_train_doc_vector2=list(train_test_dict['X_TRAIN']['question2'].apply(model.encode))

    x_test_doc_vector1=list(train_test_dict['X_TEST']['question1'].apply(model.encode))
    x_test_doc_vector2=list(train_test_dict['X_TEST']['question2'].apply(model.encode))


    extracted_features=['que1_len', 'que2_len',
       'que1_num_words', 'que2_num_words', 'total_words', 'common_words',
       'shared_words', 'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min',
       'ctc_max', 'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
       'long_substr_ratio', 'fuzz_ratio', 'fuzz_partial_ratio',
       'token_sort_ratio', 'token_set_ratio']
    
    x_train_extracted_features=train_test_dict['X_TRAIN'][extracted_features]
    x_test_extracted_features=train_test_dict['X_TEST'][extracted_features]
    x_train_extracted_features_array=np.array(x_train_extracted_features)
    x_test_extracted_features_array=np.array(x_test_extracted_features)
    x_train_embedded_document_array=np.hstack((x_train_doc_vector1, x_train_doc_vector2))
    x_test_embedded_document_array=np.hstack((x_test_doc_vector1, x_test_doc_vector2))
    x_train=np.hstack((x_train_extracted_features_array, x_train_embedded_document_array))
    x_test=np.hstack((x_test_extracted_features_array, x_test_embedded_document_array))
    y_train=np.array(train_test_dict['Y_TRAIN'])
    y_test=np.array(train_test_dict['Y_TEST'])

    # Model Training
    model_building(x_train, y_train, y_test, x_test)
    
    
# Run the main function
main(path='data\\sample.csv')