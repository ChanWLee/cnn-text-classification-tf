#-*-coding:utf-8-*-

def check_stopword(word):
    if word[0] == '@':
        return False
    if word[0] == '#':
        return False
    if word[:7] == 'http://':
        return False
    if word[:8] == 'https://':
        return False
    if len(word) == 1:
        if word in [u',', u'.', u'(', u')', u'[', u']', u'&', u'~', u'*', u'【', u'・', u',', u'】', u'、', u'。', u'け', u'の']:
            return False
    return True
'''
if __name__ == '__main__':
    print(check_stopword('@word'))
    print(check_stopword('word'))
    print(check_stopword('http://www.naver'))
    print(check_stopword('https://www.nvlaer'))
'''