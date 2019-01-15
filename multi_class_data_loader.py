# -*- coding: utf-8 -*-
import numpy as np
import csv


class MultiClassDataLoader(object):
    """
    Handles multi-class training data.  It takes predefined sets of "train_data_file" and "dev_data_file"
    of the following record format.
        <text>\t<class label>
      ex. "what a masterpiece!	Positive"

    Class labels are given as "class_data_file", which is a list of class labels.
    """

    def __init__(self, flags, data_processor):
        self.__flags = flags
        self.__data_processor = data_processor
        self.__train_data_file = None
        self.__dev_data_file = None
        self.__class_data_file = None
        self.__classes_cache = None

    def define_flags(self):
        # self.__flags.DEFINE_string("train_data_file", "./data/kkk.train", "Data source for the training data.")
        # self.__flags.DEFINE_string("dev_data_file", "./data/kkk.dev", "Data source for the cross validation data.")
        # self.__flags.DEFINE_string("class_data_file", "./data/kkk.cls", "Data source for the class list.")

        #self.__flags.DEFINE_string("train_data_file", "./twitter/test200thou_emotion_train", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/raw_emotion_train_md60_mf100", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/raw_9_train_md60_mf1000", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/raw_9_test_train", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/raw_9_train_40m_md60_mf100", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/raw_sentiment_train_md60_mf1500", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/raw_3_train_md60_mf1000", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/edit_emotion_wo0_train", "Data source for the training data.")
        self.__flags.DEFINE_string("train_data_file", "./twitter/20_edit_emotion_train", "Data source for the training data.")

        #self.__flags.DEFINE_string("train_data_file", "./twitter/2m_emo3_train_md60_mf1000", "Data source for the training data.")
        #self.__flags.DEFINE_string("train_data_file", "./twitter/1m_dom_train", "Data source for the training data.")

        self.__flags.DEFINE_string("dev_data_file", "./twitter/edit_emotion_dev", "Data source for cross validation data.")
        #self.__flags.DEFINE_string("dev_data_file", "./twitter/raw_emotion_test_dev", "Data source for cross validation data.")
        #self.__flags.DEFINE_string("dev_data_file", "./twitter/test200thou_emotion_dev", "Data source for cross validation data.")
        #self.__flags.DEFINE_string("dev_data_file", "./twitter/raw_9_test_dev", "Data source for cross validation data.")
        #self.__flags.DEFINE_string("dev_data_file", "./twitter/raw_test_dev", "Data source for cross validation data.")
        #self.__flags.DEFINE_string("dev_data_file", "./twitter/raw_3_test_dev", "Data source for cross validation data.")


        #self.__flags.DEFINE_string("dev_data_file", "./twitter/2m_emo_dev", "Data source for cross validation data.")
        #self.__flags.DEFINE_string("dev_data_file", "./twitter/1m_dom_dev", "Data source for cross validation data.")

        #self.__flags.DEFINE_string("class_data_file", "./twitter/emotion3.cls", "Data source for the class list.")

        #self.__flags.DEFINE_string("class_data_file", "./twitter/emotion9.cls", "Data source for the class list.")
        self.__flags.DEFINE_string("class_data_file", "./twitter/emotion.cls", "Data source for the class list.")
        #self.__flags.DEFINE_string("class_data_file", "./twitter/sentiment..cls", "Data source for the class list.")


    def build_vocabulary(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        self.vocab_processor = self.__data_processor.vocab_processor(x_train, x_dev)
        return [x_train, y_train, x_dev, y_dev]


    def prepare_data_without_build_vocab(self, x_train, x_dev):
        x_train = np.array(list(self.vocab_processor.fit_transform(x_train)))
        x_dev = np.array(list(self.vocab_processor.fit_transform(x_dev)))
        return [x_train, x_dev]


    def prepare_data(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)

        # max_doc_len = max([len(doc) for doc in x_train])
        # max_doc_len_dev = max([len(doc) for doc in x_dev])
        # if max_doc_len_dev > max_doc_len:
        #     max_doc_len = max_doc_len_dev

        # Build vocabulary
        self.vocab_processor = self.__data_processor.vocab_processor(x_train, x_dev)
        # x_train = np.array(list(self.vocab_processor.fit_transform(x_train)))
        aa = self.vocab_processor.fit_transform(x_train)
        aa = list(aa)
        x_train = np.array(aa)

        # Build vocabulary
        # x_dev = np.array(list(self.vocab_processor.fit_transform(x_dev)))
        bb = self.vocab_processor.fit_transform(x_dev)
        bb = list(bb)
        x_dev = np.array(bb)
        return [x_train, y_train, x_dev, y_dev]

    def restore_vocab_processor(self, vocab_path):
        return self.__data_processor.restore_vocab_processor(vocab_path)

    def class_labels(self, class_indexes):
        return [self.__classes()[idx] for idx in class_indexes]

    def class_count(self):
        return self.__classes().__len__()

    def load_dev_data_and_labels(self):
        self.__resolve_params()
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        return [x_dev, y_dev]

    def load_train_data_and_labels(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        return [x_train, y_train]

    def load_data_and_labels(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        x_all = x_train + x_dev
        y_all = np.concatenate([y_train, y_dev], 0)
        return [x_all, y_all]

    def __load_data_and_labels(self, data_file):
        x_text = []
        y = []
        with open(data_file, 'r') as tsvin:
        #with open(data_file, 'r', encoding='utf-8') as tsvin:
            classes = self.__classes()
            one_hot_vectors = np.eye(len(classes), dtype=int)
            class_vectors = {}
            for i, cls in enumerate(classes):
                class_vectors[cls] = one_hot_vectors[i]
            tsvin = csv.reader(tsvin, delimiter=',')
            for row in tsvin:
                leng_row = len(row)
                text_col = leng_row-1
                if leng_row < 2:
                    continue
                data = self.__data_processor.clean_data(row[text_col])
                # data = self.__data_processor.clean_data(row[0])
                if data == '':
                    continue
                x_text.append(data)

                slct = ','.join([col for col in row[0:text_col]])
                y.append(class_vectors[slct])
                y_array = np.array(y)
        return [x_text, y_array]

    def __classes(self):
        self.__resolve_params()
        if self.__classes_cache is None:
            with open(self.__class_data_file, 'r') as catin:
                classes = list(catin.readlines())
                self.__classes_cache = [s.strip() for s in classes]
        return self.__classes_cache

    def __resolve_params(self):
        if self.__class_data_file is None:
            self.__train_data_file = self.__flags.FLAGS.train_data_file
            self.__dev_data_file = self.__flags.FLAGS.dev_data_file
            self.__class_data_file = self.__flags.FLAGS.class_data_file
