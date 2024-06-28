import torch
import pickle

if __name__ == '__main__':
    # 尝试使用 CUDA，如果可用的话
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('save/model.pkl', map_location=device)
    model.to(device)  # 将模型移到对应设备上
    output = open('cws_result.txt', 'w', encoding='utf-8')

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    with open('data/test.txt', 'r', encoding='utf-8') as f:
        for test in f:
            flag = False
            test = test.strip()

            x = torch.LongTensor(1, len(test)).to(device)  # 将张量移到对应设备上
            mask = torch.ones_like(x, dtype=torch.uint8).to(device)  # 将张量移到对应设备上
            length = [len(test)]
            for i in range(len(test)):
                if test[i] in word2id:
                    x[0, i] = word2id[test[i]]
                else:
                    x[0, i] = len(word2id)

            predict = model.infer(x, mask, length)[0]  # 预测结果是整数列表，无需移回 CPU
            for i in range(len(test)):
                print(test[i], end='', file=output)
                if id2tag[predict[i]] in ['E', 'S']:
                    print(' ', end='', file=output)
            print(file=output)
