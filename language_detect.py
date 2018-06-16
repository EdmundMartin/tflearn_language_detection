import re
from collections import Counter
import string
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import tflearn
from tflearn.data_utils import pad_sequences


class LanguageDetection:

    def __init__(self, learning_rate=0.001, test_split=0.1, max_document_size=50, vocab_size=1000):
        self.model = None
        self._languages = list()
        self._sentences = {}
        self._word_index = {}
        self._index_length = None
        self._max_document_size = max_document_size
        self._vocab_size = vocab_size
        self._test_split = test_split
        self._learning_rate = learning_rate

    def load_words(self, file, language):
        self._sentences[language] = []
        self._languages.append(language)
        exclude = set(string.punctuation)
        with open(file, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                lower_stripped = ''.join(ch for ch in line.lower().strip() if ch not in exclude)
                tokenized = re.findall('\w+', lower_stripped)
                self._sentences[language].append(tokenized)

    def _labels_to_nums(self, label):
        zeros = [0 for i in range(len(self._languages))]
        index = self._languages.index(label.lower())
        zeros[index] = 1
        return zeros

    def _count_words(self):
        all_vocab = []
        for lang in self._languages:
            word_count = Counter()
            sentences = self._sentences[lang]
            for sentence in sentences:
                for token in sentence:
                    word_count[token] += 1
            common_vocab = word_count.most_common(self._vocab_size)
            for w in common_vocab:
                if w[0] not in all_vocab:
                    all_vocab.append(w[0])
        for i, word in enumerate(all_vocab, 1):
            self._word_index[word] = i
        self._index_length = len(all_vocab) + 1
        print(self._index_length)

    def create_training_data(self):
        X = []
        y = []
        for k, v in self._sentences.items():
            for sentence in v:
                word_ids = np.zeros(self._max_document_size, np.int64)
                for idx, token in enumerate(sentence):
                    if idx >= self._max_document_size:
                        break
                    word_id = self._word_index.get(token)
                    if word_id is None:
                        word_ids[idx] = 0
                    else:
                        word_ids[idx] = word_id
                X.append(word_ids)
                labels = self._labels_to_nums(k)
                y.append(labels)
        X = pad_sequences(X, maxlen=self._max_document_size, value=0.)
        y = [np.array(label) for label in y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_split, random_state=42)
        return X_train, X_test, y_train, y_test

    def _build_model(self):
        net = tflearn.input_data([None, self._max_document_size])
        net = tflearn.embedding(net, input_dim=self._index_length, output_dim=256)
        net = tflearn.lstm(net, 256, dropout=0.8)
        net = tflearn.fully_connected(net, len(self._languages), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=self._learning_rate,
                                 loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        return model

    def train_model(self, epochs=5, batch_size=32):
        if not self._word_index:
            self._count_words()
        X_train, X_test, y_train, y_test = self.create_training_data()
        model = self._build_model()
        model.fit(X_train, y_train, n_epoch=epochs, shuffle=True, validation_set=(X_test, y_test), show_metric=True,
                  batch_size=batch_size)
        self.model = model

    def save_model(self, model_name, word_index_format='pickle'):
        if not self.model:
            raise AttributeError('Cannot save non-existent model')
        self.model.save(model_name)
        if word_index_format == 'pickle':
            save_word_index = open('{}.pkl'.format(model_name), 'wb')
            pickle.dump(self._word_index, save_word_index)
            save_word_index.close()
            save_tags = open('{}-tags.pkl'.format(model_name), 'wb')
            pickle.dump(self._languages, save_tags)
            save_tags.close()

    def load_model(self, model_name, word_index_format='pickle'):
        if word_index_format == 'pickle':
            languages = open('{}-tags.pkl'.format(model_name), 'rb')
            word_index = open('{}.pkl'.format(model_name), 'rb')
            self._languages = pickle.load(languages)
            self._word_index = pickle.load(word_index)
            self._index_length = len(self._word_index.keys()) + 1
            model = self._build_model()
            model.load(model_name)
            self.model = model

    def _prepare_single_sentence(self, sentence):
        exclude = set(string.punctuation)
        normalized = ''.join(ch for ch in sentence.lower().strip() if ch not in exclude)
        tokens = re.findall('\w+', normalized)
        words_ids = np.zeros(self._max_document_size, np.int64)
        for idx, token in enumerate(tokens):
            if idx >= self._max_document_size:
                break
            word_id = self._word_index.get(token)
            if word_id is None:
                words_ids[idx] = 0
            else:
                words_ids[idx] = word_id
        return words_ids

    def predict(self, sentence):
        document_data = self._prepare_single_sentence(sentence)
        result = self.model.predict([document_data])[0]
        most_probable = max(result)
        results = list(result)
        most_probable_index = results.index(most_probable)
        class_name = self._languages[most_probable_index]
        return class_name, results


if __name__ == '__main__':
    w = LanguageDetection(vocab_size=5000)
    #w.load_words('ru.txt', 'ru')
    #w.load_words('bg.txt', 'bg')
    #w.load_words('ua.txt', 'ua')
    #w.load_words('mk.txt', 'mk')
    #w.train_model(epochs=1)
    #w.save_model('QuickExample')
    w.load_model('QuickExample')
    class_name, results = w.predict('Преспа ден пред потпишувањето на историскиот договор')
    print(class_name, results)