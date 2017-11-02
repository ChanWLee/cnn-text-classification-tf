from konlpy.tag import Komoran

from nlp.test_sentence import *


def process(sentence):
    ko = Komoran()
    pos = ko.pos(sentence)
    # print(pos)
    exceptions = ['J', 'E', 'JX', 'EC', 'JKS', 'EC', 'EF', 'NA']
    s = ''
    for x in pos:
        if x[1] not in exceptions:
            s += x[0]
            s += ' '
    return s


if __name__ == '__main__':
    for x in l:
        print(process(x))
