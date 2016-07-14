"""
    Copyright 2016 Seldon Technologies Ltd.
    ​
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    ​
        http://www.apache.org/licenses/LICENSE-2.0
    ​
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from flask import request, Flask#import required dependencies
from flask_cors import CORS, cross_origin

import pandas as pd
import numpy as np
import os
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer

import sys
#import time

result = "result"

app = Flask(__name__)
CORS(app, origin="http://localhost")

LEAVE = 'leave'#constants
REMAIN = 'remain'

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
    
pipeline = Pipeline([#set up pipeline for vectorising, transforming & classifying
    ('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', LogisticRegression())#using linearSVC as it seems to wield the most accurate results
])

#@app.route('/learn/', methods=['POST'])
#@cross_origin(origin='http://localhost')
def Learn():
    
    data = pd.DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))#build a dataframe containing all
                                                                  #the data needed for sklearn
    data = data.reindex(np.random.permutation(data.index))#shuffle the data
    
    #print(data)
    
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
    
    print('\nTotal sentences classified:', len(data), flush=False)#performance metrics
    #print(scores, sum(scores), len(scores), flush=False)
    print('Leave Score:', sum(leaveScores)/len(leaveScores), flush=False)
    print('Remain Score:', sum(remainScores)/len(remainScores), flush=False)
    print('Confusion matrix:', flush=False)
    print(confusion, flush=False)
    
    return "Success"

@app.route('/predict/', methods=['POST'])
@cross_origin(origin='http://localhost')
def Predict():
    
    stringToPredict = request.get_json('stringToPredict')["stringToPredict"]
    
    remainTrack = 0#scores for decision time
    leaveTrack = 0    
    
    predictSentences = nltk.sent_tokenize(stringToPredict)
    
    for i in predictSentences:#predict every sentence and score them
        
        print(pipeline.predict_proba([i])[0], flush=True)
        
        if pipeline.predict([i])[0]=="remain":
            remainTrack += 1
        else:
            leaveTrack += 1
    
    if remainTrack > leaveTrack:#decision time babee
        return "remain"
        
    elif remainTrack < leaveTrack:
        return "leave"
        
    else:
        return "neutral"

Learn()

if __name__ == "__main__":
    app.run(debug=False)