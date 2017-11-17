#! /usr/bin/env python
# -*- coding: utf-8 -*-
from langdetect import detect
from tensorflow.contrib import learn
from nlp.mecab import process


class WordDataProcessor(object):
    def vocab_processor(_, *texts):
        max_document_length = 0
        min_frequency = 200
        for text in texts:
            max_doc_len = max([len(line.split(" ")) for line in text])
            if max_doc_len > max_document_length:
                max_document_length = max_doc_len
        return learn.preprocessing.VocabularyProcessor(
            min_frequency=min_frequency
            ,max_document_length=max_document_length
                                                       )

    def restore_vocab_processor(_, vocab_path):
        return learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    def clean_data(_, string):
        try:
            if detect(string) != 'ko':
                pass
        except:
            pass
        string = process(string)
        # if ":" not in string:
        #     string = string.strip().lower()
        return string
