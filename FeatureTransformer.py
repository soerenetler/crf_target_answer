from sklearn.base import TransformerMixin
from tqdm import tqdm
import sys
import itertools

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import allennlp

#wget https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz

class FeatureTransformer(TransformerMixin):
    def __init__(self,
                 nlp,
                 pos_features=True,
                 ent_type_features=True,
                 lemma_features=True,
                 srl_features=True,
                 is_features=True,
                 position_features=True,
                 bias=True,
                 begin=-1,
                 end=1):
        self.nlp = nlp
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=1)
        self.pos_features = pos_features
        self.ent_type_features = ent_type_features
        self.lemma_features = lemma_features
        self.srl_features = srl_features
        self.is_features = is_features
        self.position_features = position_features
        self.bias = bias
        self.begin = begin
        self.end = end

    def get_params(self, deep=True):
        return {"nlp": self.nlp,
                "pos_features": self.pos_features,
                "ent_type_features": self.ent_type_features,
                "lemma_features": self.lemma_features,
                "is_features": self.is_features,
                "position_features": self.position_features,
                "bias": self.bias,
                "begin": self.begin,
                "end": self.end}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key=="nlp": self.nlp=value
            elif key=="pos_features": self.pos_features=value
            elif key=="ent_type_features": self.ent_type_features=value
            elif key=="lemma_features": self.lemma_features=value
            elif key=="is_features": self.is_features=value
            elif key=="is_features": self.is_features=value
            elif key=="position_features": self.position_features=value
            elif key=="bias": self.bias=value
            elif key=="begin": self.begin=value
            elif key=="end": self.end=value

    def fit(self, X):
        return self

    def transform(self, X):
        return [self.text2features(x) for x in tqdm(self.nlp.pipe(X))]

    def word2features(self, sent, srl_tags, srl_verb, i):
        features = {}
        if self.bias:
            features['bias'] = 1.0
        if self.position_features:
            features['AbsPos'] = str(i)
            #features['RelPos'] = i/len(sent)
            features['QuartilePos'] = str(int(4*(i/len(sent))))
        
        if sent[i].is_space:
            features['WHITE'] = True
        else:
            for n in range(self.begin, self.end+1):
                if i + n < 0:
                    features['{} BOS'.format(n)] = True
                elif i + n >= len(sent):
                    features['{} EOS'.format(n)] = True
                else:
                    if sent[i+n].is_space:
                        features['{}_WHITE'.format(n)] = True
                    else:
                        word = sent[i+n]
                        # POS and Dependency Tag
                        features['{}:pos'.format(n)] = word.pos_
                        if word.dep_ != word.pos_.lower():
                            features['{}:dep'.format(n)] = word.dep_
                        if word.tag_ != word.text:
                            features['{}:tag'.format(n)] = word.tag_
                        
                        # NER Features
                        features['{}:ent_iob'.format(n)] = word.ent_iob_
                        if word.ent_iob_ != "O":
                            features['{}:ent_type'.format(n)] = word.ent_type_
                        
                        #Text Feature
                        features['{}:word'.format(n)] = word.text.lower()
                        for fix in range(1,4):
                            if len(word) > fix and not word.is_punct:
                                features['{}:pre{}'.format(n, fix)] = word.text.lower()[:fix]
                                features['{}:suf{}'.format(n, fix)] = word.text.lower()[-fix:]
                        
                        #SRL Feature
                        if len(srl_verb) >0:
                            features['{}:srl_verb'] = srl_verb[0].lower()
                            features['{}:srl_iob'.format(n)] = srl_tags[0][i+n][0]
                            if srl_tags[0][i+n][0] != "O":
                                features['{}:srl_type'.format(n)] = srl_tags[0][i+n][2:]

                        for tags, verb in zip(srl_tags, srl_verb):
                            if tags[i+n][0] != "O":
                                if tags[i+n][2:] != "V":
                                    features['{}:srl_{}'.format(n, tags[i+n][0])] = True
                                features['{}:srl_{}'.format(n, tags[i+n][2:])] = tags[i+n][0]
                                features['{}:srl_v_{}'.format(n, tags[i+n][2:])] = verb.lower()
                        
                        #Morphological Feature
                        for key, value in word.morph.to_dict().items():
                            features['{}:morph_{}'.format(n, key)] = value
                        
                        #Reduced Shape
                        shape_reduced = ''.join(i for i, _ in itertools.groupby(word.shape_))
                        if shape_reduced != word.text:
                            features['{}:shape'.format(n)] = shape_reduced

                        #Additional Features
                        features.update({
                            '{}:alpha'.format(n): word.is_alpha,
                            #'{}:word.is_ascii()'.format(n): word.is_ascii,
                            #'{}:like_num'.format(n):word.like_num,
                            #'{}:punct'.format(n):word.is_punct,
                            #'{}:oov'.format(n):word.is_oov,
                            '{}:stop'.format(n):word.is_stop
                            #'{}:word.shape'.format(n):word.shape_,
                            #'{}:word.prob'.format(n):word.prob
                        })
        return features

    def text2features(self, sent):
        tokens = [token for token in sent]
        #allennlp.nn.util.move_to_device(token, 0)

        #print(allennlp.nn.util.get_device_of(token))
        #print(allennlp.nn.util.get_device_of(self.predictor))

        instances = self.predictor.tokens_to_instances(tokens)
        if not instances:
            srl_tags = []
            srl_verb = []
        else:
            srl_pred = self.predictor.predict_instances(instances)
            srl_tags = [element["tags"] for element in srl_pred["verbs"]]
            srl_verb = [element["verb"] for element in srl_pred["verbs"]]
        return [self.word2features(sent, srl_tags, srl_verb, i) for i in range(len(sent))]