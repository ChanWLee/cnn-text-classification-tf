from konlpy.tag import Twitter

from nlp.test_sentence import *


def process(sentence):
    tw = Twitter()
    pos = tw.pos(sentence)
    # print(pos)
    exceptions = ['Josa', 'Eomi']
    s = ''
    for x in pos:
        if x[1] not in exceptions:
            s += x[0]
            s += ' '
    return s


if __name__ == '__main__':
    for x in l:
        print(process(x))
