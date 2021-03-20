from sklearn.base import TransformerMixin
from tqdm import tqdm
import sys

from ngrams import NGrams

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

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
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.pos_features = pos_features
        self.ent_type_features = ent_type_features
        self.lemma_features = lemma_features
        self.srl_features = srl_features
        self.is_features = is_features
        self.position_features = position_features
        self.bias = bias
        self.begin = begin
        self.end = end

    def fit(self, X, y=None):
        tokenized_list = []
        for sentence in tqdm(self.nlp.pipe(X)):
            tokenized_list.append([token.text for token in sentence])

        self.unigram_model = NGrams(tokenized_list, n=2)
        self.bigram_model = NGrams(tokenized_list, n=3)
        self.trigram_model = NGrams(tokenized_list, n=4)

        return self

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
                        if self.srl_features:
                            features['{}:srl'.format(n)] = srl_tags[i+n]
                            features['{}:srl_verb'.format(n)] = srl_verb
                        if self.is_features:
                            features.update({
                                '{}:word.is_alpha()'.format(n): word.is_alpha,
                                '{}:word.is_ascii()'.format(n): word.is_ascii,
                                '{}:word.is_digit()'.format(n): word.is_digit,
                                '{}:word.is_lower()'.format(n): word.is_lower,
                                '{}:word.is_upper()'.format(n): word.is_upper,
                                '{}:word.is_title()'.format(n): word.is_title,
                                '{}:word.is_punct'.format(n):word.is_punct,
                                '{}:word.is_left_punct'.format(n):word.is_left_punct,
                                '{}:word.is_right_punct'.format(n):word.is_right_punct,
                                '{}:word.is_space'.format(n):word.is_space,
                                '{}:word.is_bracket'.format(n):word.is_bracket,
                                '{}:word.is_quote'.format(n):word.is_quote,
                                '{}:word.is_currency'.format(n):word.is_currency,
                                '{}:word.like_url'.format(n):word.like_url,
                                '{}:word.like_num'.format(n):word.like_num,
                                '{}:word.like_email'.format(n):word.like_email,
                                '{}:word.is_oov'.format(n):word.is_oov,
                                '{}:word.is_stop'.format(n):word.is_stop,
                                '{}:word.shape'.format(n):word.shape_,
                                '{}:word.prob'.format(n):word.prob
                            })
                        if True:
                            try:
                                unigram_context = (sent[i+n-1].text.lower(),)
                                unigram_proba = self.unigram_model[unigram_context][sent[i+n].text.lower()]
                            except (KeyError, IndexError):
                                unigram_proba = 0
                            features['{}:word.unigram_proba'.format(n)] = unigram_proba
                            
                            try:
                                bigram_context = (sent[i+n-2].text.lower(), sent[i+n-1].text.lower(),)
                                bigram_proba = self.bigram_model[bigram_context][sent[i+n].text.lower()]
                            except (KeyError, IndexError):
                                bigram_proba = 0
                            features['{}:word.bigram_proba'.format(n)] = bigram_proba

                            try:
                                trigram_context = (sent[i+n-3].text.lower(), sent[i+n-2].text.lower(), sent[i+n-1].text.lower(),)
                                trigram_proba = self.trigram_model[trigram_context][sent[i+n].text.lower()]
                            except (KeyError, IndexError):
                                trigram_proba = 0
                            features['{}:word.trigram_proba'.format(n)] = trigram_proba

        return features

    def text2features(self, sent):
        print([token.text for token in sent])
        srl_pred = self.predictor.predict_tokenized([token.text for token in sent])
        if not srl_pred["verbs"]:
            srl_tags = ["O"]*len(sent)
            srl_verb = "NONE"
        else:
            srl_tags = self.predictor.predict_tokenized([token.text for token in sent])["verbs"][0]["tags"]
            srl_verb = srl_pred["verbs"][0]["verb"]
        return [self.word2features(sent, srl_tags, srl_verb, i) for i in range(len(sent))]