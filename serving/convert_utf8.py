# -*- coding: utf-8 -*-
text = []

# hoo = unicode('한글', 'utf-8')

with open('./lex_ko_p2','rb') as f:
    for idx, line in enumerate(f.readlines()):
        if idx < 5 :
            print('raw:{}'.format(line))
        text.append(str(unicode(line)))
        if idx < 5:
            print('after:{}'.format(text))
