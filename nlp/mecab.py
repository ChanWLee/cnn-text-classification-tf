# -*-coding:utf-8-*-
from konlpy.tag import Mecab
from nlp.stopword import check_stopword
from nlp.test_sentence import *


def process(sentence):
    mecab = Mecab()

    pos = mecab.pos(sentence)
    exceptions = ['JKO', 'JX', 'JKX', 'EC', 'EF', 'SSO', 'SSC', 'JKC', 'JKS', 'JKO', 'JKB', 'JKG', 'ETM', 'XSN']
    # s = list()
    s = ''
    for x in pos:
        if x[1] not in exceptions:
            if check_stopword(x[0]):
                s += x[0]
                s += ' '
                # s.append(x[0])

    return s


if __name__ == '__main__':
    r = '한국에서는 많은 사람들이 김치를 먹습니다 매일 매일 김치를 먹어야 한다고 말하는 사람도 있습니다 하지만 모든 사람이 다 그런 것은 아닙니다 저는 김치를 싫어하는 것은 아니지만 매일 먹지는 않습니다 또 식탁에 김치가 올라와 있지 않아도 별 상관 없습니다 '
    k = '【みをつくし料理帖】最終回をご覧くださった皆さま、ありがとうございました！とりあえず最後となりますが、公式ＨＰで料理レシピ動画、原作・高田郁さんの「ちょい足し料理帖」、スタッフブログを更新しています。'
    s = 'さい。彼はハッカーであり、あなたのFacebookアカウントにシステムを接続しています。あなたの連絡先の1人がそれを受け入れると、あなたもハッキングされるので、すべての友達にそれを知らせてください。受け取ったとおりに転送されます。'
    b = 'メッセージを指で押さえてください。真ん中の一番下に転送ボタンがあります。それをタッチし、あなたのリストにある人の名前をクリックすると、それがそれらに送信されますさい。彼はハッカーであり、あなたのFacebookアカウントにシステムを接続しています。あなたの連絡先の1人がそれを受け入れると、あなたもハッキングされるので、すべての友達にそれを知らせてください。受け取ったとおりに転送されます。'

    s2 = '나는 바보가 아닙니다 당신은 누구십니까? 이건 어떤건가 많은 수의 데이터가 필요하다 왜냐하면 20을 넘겨야 하기 때문에 나는 그래서 일부러 길게 쓴다'

    # print(process(r))
    for x in l:
        print(process(x))