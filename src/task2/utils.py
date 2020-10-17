import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
import re
import string
from funcy import lflatten
from sklearn_crfsuite import metrics


class Task2Utils(object):

    def readDataAsDataFrame(self,path):
        df = pd.read_csv(path, delimiter='; ', engine='python', header=0)
        return df
    
    def getData(self,df):
        df['IdxSplit'] = df.Index.apply(lambda x: ''.join(x.split(".")[0:2]))
        df.set_index('IdxSplit', inplace=True)
        np.random.seed(0)
        testrows = np.random.choice(df.index.values, int(len(df) / 3))
        test = df.loc[testrows].drop_duplicates(subset='Index')
        train = df.drop(test.index)
        return train,test
    
    def makeDictionary(lines,lot):
        d = defaultdict(list)
        for line_, tag_ in zip(lines, lot):
            d[tag_] = line_

        return d

    def preprocessData(self,df):
        lodict_ = []
        for rows in df.itertuples():
            list_ = [rows[2], rows[3], rows[4]]
            map1 = ['sentence', 'cause', 'effect']
            dict_ = self.makeDictionary(list_, map1)
            lodict_.append(dict_)
        map_ = [('cause', 'C'), ('effect', 'E')]
        hometags = self.make_causal_input(lodict_, map_)
        postags = self.nltkPOS([i['sentence'] for i in lodict_])
        tokens = ((token for token in word_tokenize(sent)) for sent in df.Text.tolist())
        data = []
        for i, (j, k) in enumerate(zip(hometags, postags)):
            data.append([(w, pos, label) for (w, label), (word, pos) in zip(j, k)])

        X = [self.extract_features(doc) for doc in data]
        y = [self.get_multi_labels(doc) for doc in data]

        return X, y, tokens

    def make_causal_input(self,lod, map_, silent=True):
        dd = defaultdict(list)
        dd_ = []
        rx = re.compile(r"(\b[-']\b)|[\W_]")
        rxlist = [r'("\\)', r'(\\")']
        rx = re.compile('|'.join(rxlist))
        for i in range(len(lod)):
            line_ = lod[i]['sentence']
            line = re.sub(rx, '', line_)
            line = ' '.join(word.strip(string.punctuation) for word in line.split())
            caus = lod[i]['cause']
            caus = re.sub(rx, '', caus)
            caus = ' '.join(word.strip(string.punctuation) for word in caus.split())
            effe = lod[i]['effect']
            effe = re.sub(rx, '', effe)
            effe = ' '.join(word.strip(string.punctuation) for word in effe.split())

            d = defaultdict(list)
            index = 0
            for idx, w in enumerate(word_tokenize(line)):
                index = line.find(w, index)

                if not index == -1:
                    d[idx].append([w, index])
                    # print(w, index)hu
                    index += len(w)

            d_ = defaultdict(list)
            for idx in d:
                d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])

            init_e = line.find(effe)
            init_e = 0 if init_e == -1 else init_e
            init_c = line.find(caus)
            init_c = 0 if init_c == -1 else init_c

            for c, cl in enumerate(word_tokenize(caus)):
                # print('init_c', init_c)
                init_c = line.find(cl, init_c)
                # print('start Cause', init_c)
                stop = line.find(cl, init_c) + len(cl)
                word = line[init_c:stop]
                # print('word', word.upper(), 'el', cl.upper())

                for idx in d_:
                    if int(init_c) == int(d_[idx][0][1]):
                        und_ = defaultdict(list)
                        und_[idx].append([tuple([word, 'C']), line.find(word, init_c)])
                        d_[idx] = und_[idx]

                init_c += len(cl)
                # print('increment_c', init_c)

            for e, el in enumerate(word_tokenize(effe)):
                # print('init_e', init_e)
                init_e = line.find(el, init_e)
                # print('start Effect', init_e)
                stop = line.find(el, init_e) + len(el)
                word = line[init_e:stop]
                # print('word', word.upper(), 'el', el.upper())

                for idx in d_:
                    if int(init_e) == int(d_[idx][0][1]):
                        und_ = defaultdict(list)
                        und_[idx].append([tuple([word, 'E']), line.find(word, init_e)])
                        d_[idx] = und_[idx]

                init_e += len(word)
                # print('init_e', init_e)

            dd[i].append(d_)

        for dict_ in dd:
            dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])

        return dd_

    def nltkPOS(self,loft):
        su_pos = []
        rx = re.compile(r"(\b[-']\b)|[\W_]")
        rxlist = [r'("\\)', r'(\\")']
        rx = re.compile('|'.join(rxlist))

        for i, j in enumerate(loft):
            text = re.sub(rx, '', j)
            tokens = word_tokenize(text)
            pos_ = list(nltk.pos_tag(tokens))
            su_pos.append(pos_)

        return su_pos

    # ##  PREPARE MORE FEATURES

    def word2features(self,sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'BOS': True,
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1
            })
        else:
            features['EOS'] = True

        return features
    
    def extract_features(self,doc):
        return [self.word2features(doc, i) for i in range(len(doc))]
    
    def get_multi_labels(self,doc):
        return [label for (token, postag, label) in doc]
    
    def printResult(self,y_test,y_pred,tags,digits=3):
        print(metrics.flat_classification_report(
        y_test, y_pred, labels=['_', 'C', 'E'], digits=3
        ))