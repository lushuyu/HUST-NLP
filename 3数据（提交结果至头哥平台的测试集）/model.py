import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, tag_to_ix, hidden_dim):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix), batch_first=True)

    def forward(self, sentence):
        bert_out = self.bert(sentence)[0]  # (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(bert_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def loss(self, feats, tags, mask):
        return -self.crf(feats, tags, mask=mask)

    def predict(self, feats, mask):
        return self.crf.decode(feats, mask=mask)
