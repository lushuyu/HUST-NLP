class Tokenizer(object):
    def __init__(self, words, max_len):
        self.words = words
        self.max_len = max_len

    def fmm_split(self, text):
        '''
        正向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        index = 0
        while index < len(text):
            matched = False
            for size in range(self.max_len, 0, -1):
                if index + size > len(text):
                    continue
                word = text[index:index + size]
                if word in self.words:
                    result.append(word)
                    index += size
                    matched = True
                    break
            if not matched:
                result.append(text[index])
                index += 1
        return result

    def rmm_split(self, text):
        '''
        逆向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        index = len(text)
        while index > 0:
            matched = False
            for size in range(self.max_len, 0, -1):
                if index - size < 0:
                    continue
                word = text[index - size:index]
                if word in self.words:
                    result.insert(0, word)
                    index -= size
                    matched = True
                    break
            if not matched:
                result.insert(0, text[index - 1])
                index -= 1
        return result

    def bimm_split(self, text):
        '''
        双向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        fmm_result = self.fmm_split(text)
        rmm_result = self.rmm_split(text)

        if len(fmm_result) < len(rmm_result):
            return fmm_result
        elif len(fmm_result) > len(rmm_result):
            return rmm_result
        else:
            # If both methods produce the same number of words, choose the one with fewer single-character words
            fmm_single_count = sum(1 for word in fmm_result if len(word) == 1)
            rmm_single_count = sum(1 for word in rmm_result if len(word) == 1)
            if fmm_single_count <= rmm_single_count:
                return fmm_result
            else:
                return rmm_result

def load_dict(path):
    tmp = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            tmp.add(word)
    return tmp

if __name__ == '__main__':
    words = load_dict('dict.txt')
    max_len = max(map(len, [word for word in words]))

    # test
    tokenizer = Tokenizer(words, max_len)
    texts = [
        '研究生命的起源',
        '无线电法国别研究',
        '人要是行，干一行行一行，一行行行行行，行行行干哪行都行。'
    ]
    for text in texts:
        # 前向最大匹配
        print('前向最大匹配:', '/'.join(tokenizer.fmm_split(text)))
        # 后向最大匹配
        print('后向最大匹配:', '/'.join(tokenizer.rmm_split(text)))
        # 双向最大匹配
        print('双向最大匹配:', '/'.join(tokenizer.bimm_split(text)))
        print('')
