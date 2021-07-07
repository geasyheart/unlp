# -*- coding: utf8 -*-

#


def bmes_of(sentence, segmented):
    if segmented:
        chars = []
        tags = []
        words = sentence.split()
        for w in words:
            chars.extend(list(w))
            if len(w) == 1:
                tags.append('S')
            else:
                tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    else:
        chars = list(sentence)
        tags = ['S'] * len(chars)
    return chars, tags


if __name__ == '__main__':
    sentence = '我 爱 中国 天安门'
    print(bmes_of(sentence, True))
