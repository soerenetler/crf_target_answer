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
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=0)
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
            features['Whitespace'] = True
        else:
            for n in range(self.begin, self.end+1):
                if i + n < 0:
                    features['{} BOS'.format(n)] = True
                elif i + n >= len(sent):
                    features['{} EOS'.format(n)] = True
                else:
                    if sent[i+n].is_space:
                        features['{}_Whitespace'.format(n)] = True
                    else:
                        word = sent[i+n]
                        if self.pos_features:
                            features['{}:pos_'.format(n)] = word.pos_
                            features['{}:tag_'.format(n)] = word.tag_
                            features['{}:dep_'.format(n)] = word.dep_
                        if self.ent_type_features:
                            features['{}:ent_iob_'.format(n)] = word.ent_iob_
                            if word.ent_iob_ != "O":
                                features['{}:ent_type'.format(n)] = word.ent_type_
                        if self.lemma_features:
                            #features['{}:word.lemma'.format(n)] = word.lemma_
                            #features['{}:word.norm'.format(n)] = word.norm_
                            features['{}:word'.format(n)] = word.text.lower()
                            for fix in range(1,4):
                                if len(word) > fix and not word.is_punct:
                                    features['{}:prefix{}'.format(n, fix)] = word.text.lower()[:fix]
                                    features['{}:suffix{}'.format(n, fix)] = word.text.lower()[-fix:]
                        if self.srl_features:
                            #features['{}:srl'.format(n)] = srl_tags[i+n]
                            features['{}:srl_verb'] = srl_verb[0][i+n][0]
                            features['{}:srl_iob'.format(n)] = srl_tags[0][i+n][0]
                            if srl_tags[i+n][0] != "O":
                                features['{}:srl_type'.format(n)] = srl_tags[0][i+n][2:]

                            for tags, verb in zip(srl_tags, srl_verb):
                                if tags[i+n][0] != "O":
                                    features['{}:srl_{}'.format(n, tags[i+n][0])] = tags[i+n][2:]
                                    features['{}:srl_verb_{}'.format(n, tags[i+n])] = verb
                        if True:
                            for key, value in word.morph.to_dict().items():
                                features['{}:morph_{}'.format(n, key)] = value
                        if self.is_features:
                            features.update({
                                '{}:alpha'.format(n): word.is_alpha,
                                #'{}:word.is_ascii()'.format(n): word.is_ascii,
                                '{}:like_num'.format(n):word.like_num,
                                '{}:punct'.format(n):word.is_punct,
                                '{}:oov'.format(n):word.is_oov,
                                '{}:stop'.format(n):word.is_stop,
                                #'{}:word.shape'.format(n):word.shape_,
                                '{}:shape_reduced'.format(n):''.join(i for i, _ in itertools.groupby(word.shape_)),
                                #'{}:word.prob'.format(n):word.prob
                            })
        return features

    def text2features(self, sent):
        tokens = [token.text for token in sent]
        #allennlp.nn.util.move_to_device(token, 0)

        #print(allennlp.nn.util.get_device_of(token))
        #print(allennlp.nn.util.get_device_of(self.predictor))
        srl_pred = self.predictor.predict_tokenized(tokens)
        if not srl_pred["verbs"]:
            srl_tags = []
            srl_verb = []
        else:
            srl_tags = [element["tags"] for element in srl_pred["verbs"]]
            srl_verb = [element["verb"] for element in srl_pred["verbs"]]
        return [self.word2features(sent, srl_tags, srl_verb, i) for i in range(len(sent))]