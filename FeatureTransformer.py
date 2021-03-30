from sklearn.base import TransformerMixin
from tqdm import tqdm
import sys

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

    def transform(self, X):
        return [self.text2features(x) for x in tqdm(self.nlp.pipe(X))]

    def word2features(self, sent, srl_tags, srl_verb, i):
        features = {}
        if self.bias:
            features['bias'] = 1.0
        if self.position_features:
            features['AbsolutePosition'] = i
            features['RelativePosition'] = i/len(sent)
            features['QuatilePosition'] = int(4*(i/len(sent)))

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
                            features['{}:word.pos_'.format(n)] = word.pos_
                            features['{}:word.tag_'.format(n)] = word.tag_
                            features['{}:word.dep_'.format(n)] = word.dep_
                        if self.ent_type_features:
                            features['{}:word.ent_type'.format(n)] = word.ent_type_
                            features['{}:word.ent_iob_'.format(n)] = word.ent_iob_
                        if self.lemma_features:
                            features['{}:word.lemma'.format(n)] = word.lemma_
                            features['{}:word.norm'.format(n)] = word.norm_
                            features['{}:word.prefix'.format(n)] = word.prefix_
                            features['{}:word.suffix'.format(n)] = word.suffix_
                        if self.srl_features:
                            features['{}:srl'.format(n)] = srl_tags[i+n]
                            features['{}:srl_verb'.format(n)] = srl_verb
                        if True:
                            key, value for word.morph.to_dict().entries():
                                features['{}:morph_{}'.format(n, key)] = value
                        if self.is_features:
                            features.update({
                                '{}:word.is_alpha()'.format(n): word.is_alpha,
                                '{}:word.is_ascii()'.format(n): word.is_ascii,
                                '{}:word.is_digit()'.format(n): word.is_digit,
                                '{}:word.is_lower()'.format(n): word.is_lower,
                                '{}:word.is_upper()'.format(n): word.is_upper,
                                '{}:word.is_title()'.format(n): word.is_title,
                                '{}:word.is_punct'.format(n):word.is_punct,
                                '{}:word.is_space'.format(n):word.is_space,
                                '{}:word.is_bracket'.format(n):word.is_bracket,
                                '{}:word.is_quote'.format(n):word.is_quote,
                                '{}:word.is_currency'.format(n):word.is_currency,
                                '{}:word.like_num'.format(n):word.like_num,
                                '{}:word.is_oov'.format(n):word.is_oov,
                                '{}:word.is_stop'.format(n):word.is_stop,
                                '{}:word.shape'.format(n):word.shape_,
                                '{}:word.prob'.format(n):word.prob
                            })
        return features

    def text2features(self, sent):
        tokens = [token.text for token in sent]
        #allennlp.nn.util.move_to_device(token, 0)

        #print(allennlp.nn.util.get_device_of(token))
        #print(allennlp.nn.util.get_device_of(self.predictor))
        srl_pred = self.predictor.predict_tokenized(tokens)
        if not srl_pred["verbs"]:
            srl_tags = ["O"]*len(sent)
            srl_verb = "NONE"
        else:
            srl_tags = srl_pred["verbs"][0]["tags"]
            srl_verb = srl_pred["verbs"][0]["verb"]
        return [self.word2features(sent, srl_tags, srl_verb, i) for i in range(len(sent))]