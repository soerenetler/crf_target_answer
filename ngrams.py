from collections import Counter
import random
import numpy as np

####################################
#     ngrams.py                    #
# Sprachmodellierung mit N-Grammen #
#                                  #
# Marius Gerdes      8.11.2015     #
# Sören Etler       28.11.2019     #
####################################

class NGrams:
    def __init__(self, ws, n=2, threshold=0):
        """Erschafft ein NGram Modell aus einer liste von Strings.
        ws - Eine Liste von strings
        n - Die Ordnung des NGram Modells als integer"""
        if n < 1:
            raise ValueError("Error during NGramModel creation: Cannot create NGramModel of order 0 or lower.")

        self._n = n
        self._ngrams = set()

        total_token_list = []
        for question in ws:
            for _ in range(n-1):
                total_token_list.append("")
            total_token_list.append("<BOS>")
            for token in question:
                total_token_list.append(token.lower())
            total_token_list.append("<EOS>")
        
        # wir zaehlen die Anzahl beobachteter n-grams
        # in einem dictionary von countern
        self._condFreqDist = self._count(total_token_list, n)

        # bedingte Wahrscheinlichkeitsverteilung als
        # dictionary von dictionaries
        self._condProbDist = self._mlEstimate(self._condFreqDist, threshold)

    def ngrams(self):
        return self._ngrams

    def keys(self):
        """Gibt alle gueltigen Werte fuer KEY in self[KEY] oder self.randomSampleFor(KEY) zurueck."""
        return self._condProbDist.keys()

    def items(self):
        """Gibt alle Wahrscheinlichkeitsverteilungen fuer beobachtete Kontexte als (KONTEXT, DICTIONARY) Paar zurueck."""
        return self._condProbDist.items()

    def randomSampleFor(self, context):
        """Gibt ein zufaelliges Wort fuer einen gegebnen Kontext zurueck. Die Wahrscheinlichkeit des Rueckgabewerts
        richtet sich nach der Wahrscheinlichkeitsverteilung fuer CONTEXT.
        context - Ein (n-1)-Tupel von strings (Kann bei n=2 als string statt als 1-Tupel von strings gegeben werden)
        return - Ein zufaelliger string, der mit context beobachtet wurde.
        """
        word_prob_list = list(self[context].items())
        
        rand = np.random.choice(range(len(word_prob_list)), p=[a[1] for a in word_prob_list])
        new_word = word_prob_list[rand][0]
        return new_word

    def generate_text(self):
        new_word = '<BOS>'
        context = tuple((self._n-2)*[""] + [new_word])
        generated_text_list = []
        generated_text_list.append(new_word)
        while new_word != '<EOS>':
            new_word = self.randomSampleFor(context)
            generated_text_list.append(new_word)
            context = tuple(context[1:]) + (new_word,)

        generated_text = " ".join(generated_text_list)
        return generated_text

    def combine(self, other, weight=0.5):
        new = self
        new_condProbDist = {}
        for context in [*self.keys()] + [*other.keys()]:
            if context in [*self._condProbDist.keys()] and not context in [*other._condProbDist.keys()]:
                new_condProbDist[context] = self[context]
            elif context in [*other._condProbDist.keys()] and not context in [*self._condProbDist.keys()]:
                new_condProbDist[context] = other[context]
            elif context in [*other._condProbDist.keys()] and context in [*self._condProbDist.keys()]:
                new_condProbDist[context] = {}
                for key in [*self[context].keys()] + [*other[context].keys()]:
                    new_condProbDist[context][key] = self[context].get(key, 0) * weight + self[context].get(key, 0) * (1-weight)
        new._condProbDist = new_condProbDist
        return new

    
    def _count(self, ws, n):
        cfd = {}
        for i in range((len(ws)-n)+1):
            ngram = tuple(ws[i:i+n])
            self._ngrams.add(ngram)
            nthWord = ngram[-1]
            context = ngram[:-1]

            if not (context in cfd):
                cfd[context] = Counter()

            cfd[context][nthWord] += 1

        return cfd


    def _mlEstimate(self, cfd, threshold):
        cpd = {}

        for context in cfd.keys():
            contextCount = sum(cfd[context].values())
            
        
            if contextCount >= threshold:
                cpd[context] = {}
                for nextWord in cfd[context].keys():
                    cpd[context][nextWord] = cfd[context][nextWord] / contextCount

        return cpd

    def __getitem__(self, context):
        """Index operator []. self[context] gibt die Wahrscheinlichkeitsverteilung fuer den gegebnen Kontext wieder.
        context - Bei n-grammen ein (n-1)-Tupel aus strings.
        return - Ein Dictionary als string -> float mapping."""

        return self._condProbDist[context]

    def __contains__(self, context):
        """Erlaubt Benutzung von 'in' keyword: ('the', 'good') in trigrams -> True
        Dies verhaelt sich so, dass wenn self[CONTEXT] einen Wert hat, CONTEXT in self = True ist.
        context - Ein (n-1)Tupel als vermeintlich beobachteter Kontext. (Bei n=2 auch als string akzeptierbar)
        return - Bool; je nach dem ob context beobachtet wurde."""        

        return context in self._condProbDist
