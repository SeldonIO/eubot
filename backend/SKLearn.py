import pandas as pd#import required dependencies
import numpy as np
import os
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC#, NuSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer

import sys
#import time

LEAVE = 'leave'#constants
REMAIN = 'remain'
TEST = 'test'

SOURCES = [
    ('Remain',  REMAIN),
    ('Leave',   LEAVE),
]

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def read_files(path):#read all the data and yield it in a usable format
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file in file_names:
            if file[0] != ".":
                file_path = os.path.join(root, file)
                text = ""
                r = open(file_path, encoding="UTF-8")
                
                count = 0
                
                for line in r:#convert data into sentences
                    text = ""
                    sentences = nltk.sent_tokenize(line)
                                    
                    for i in sentences:
                        count += 1
                        #print(i)
                        #time.sleep(0.01)
                        text = i
                        if i.strip() != "":
                            yield file_path + str(count), text
                
                r.close()
            
def build_data_frame(path, classification):#build a data frame from the data passed to it
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)
    
    data_frame = pd.DataFrame(rows, index=index)
    return data_frame
    
data = pd.DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))#build a dataframe containing all
                                                              #the data needed for sklearn

data = data.reindex(np.random.permutation(data.index))#shuffle the data

#print(data)

pipeline = Pipeline([#set up pipeline for vectorising, transforming & classifying
    ('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', LinearSVC())#using linearSVC as it seems to wield the most accurate results
])

folds = 10

k_fold = KFold(n=len(data), n_folds=folds)#arrays for cross-validation and performance metrics
leaveScores = []
remainScores = []
confusion = np.array([[0, 0], [0, 0]])

progressTracker = 0
progress(progressTracker, folds)

for train_indices, test_indices in k_fold:#start machine learning and predicting
    
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values
    
    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    leaveScore = f1_score(test_y, predictions, pos_label=LEAVE)
    remainScore = f1_score(test_y, predictions, pos_label=REMAIN)
    leaveScores.append(leaveScore)
    remainScores.append(remainScore)
    
    progressTracker += 1
    progress(progressTracker, folds)

print('\nTotal sentences classified:', len(data))#performance metrics
#print(scores, sum(scores), len(scores))
print('Leave Score:', sum(leaveScores)/len(leaveScores))
print('Remain Score:', sum(remainScores)/len(remainScores))
print('Confusion matrix:')
print(confusion)

while True:
    
    remainTrack = 0
    leaveTrack = 0    
    
    stringToPredict = input("Predict a string (type 'quit' to quit): ")
    
    if stringToPredict != "quit":
        predictSentences = nltk.sent_tokenize(stringToPredict)
        
        for i in predictSentences:          
            
            if pipeline.predict([i])[0]=="remain":
                remainTrack += 1
            else:
                leaveTrack += 1
        
        if remainTrack > leaveTrack:
            print("This string is pro-remain")
            
        elif remainTrack < leaveTrack:
            print("This string is pro-leave")
            
        else:
            print("This string is neutral")
    
    else:
        break