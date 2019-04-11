
import torch
from torch import nn

class VQA_Model(nn.Module):
    def __init__(self, embedding_tokens, answer_num, label_length, question_encode_length = 1024, num_attention_map = 2):
        super().__init__()
        image_feature_length = 2048

        self.Text_encoder = Text_encoder(embedding_tokens, question_encode_length)
        self.Attention = Stacked_attention(image_feature_length, label_length, question_encode_length, num_attention_map)
        
        in_len = num_attention_map * image_feature_length + question_encode_length
        self.Classifer = Classifer(in_len, answer_num)

        self.Attention_Map = None

    def forward(self, v, l, q, q_len):
        encode_q = self.Text_encoder(q, q_len)

        v = nn.functional.normalize(v, dim=1)
        l = nn.functional.normalize(l, dim=1)
        feature, self.Attention_Map = self.Attention(encode_q, v, l)
        
        feature = torch.cat([feature, encode_q], dim=1)
        out = self.Classifer(feature)

        return out

class Classifer(nn.Module):
    def __init__(self, in_len, out_len):
        super().__init__()
        self.drop = nn.Dropout(inplace=False)
        self.lin1 = nn.Linear(in_len, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(1024, out_len)

    def forward(self, x):
        x = self.lin1(self.drop(x))
        x = self.relu(x)
        out = self.lin2(self.drop(x))
        return out

class Stacked_attention(nn.Module):
    def __init__(self, image_feature_length, label_length, question_encode_length, num_attention_map):
        super().__init__()
        self.conv = nn.Conv2d(label_length,
                                question_encode_length, 1, bias=True)
        self.lin = nn.Linear(question_encode_length, question_encode_length, bias=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(question_encode_length + image_feature_length, 512, 1)
        self.conv2 = nn.Conv2d(512, num_attention_map, 1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, q, v, l):
        q1 = self.tanh(self.lin(q))
        l1 = self.tanh(self.conv(l))
        q1 = self._vec_expand(q1, l1)
        f = q1 * l1
        fused = self.conv1(torch.cat([f, v], dim=1))
        attention_map = self.conv2(self.dropout(fused))
        
        n, c = v.size()[:2]
        attention_map_size = attention_map.size()

        attention_map = attention_map.view(n, attention_map_size[1], -1)
        softmaxed_attention = nn.functional.softmax(attention_map, dim=2)

        v = v.view(n, 1, c, -1).expand(n, attention_map_size[1], c, -1)
        expand_attention = softmaxed_attention.view(n, attention_map_size[1], 1, -1).expand(n, attention_map_size[1], c, -1)

        weighted_feature = expand_attention * v
        weighted_feature = weighted_feature.sum(dim=3)

        return weighted_feature.view(n, -1),  softmaxed_attention.view(*attention_map_size).detach()

    def _vec_expand(self, vec, map):
        n, c = vec.size()
        spatial_dim = map.dim() - 2
        expanded_vec = vec.view(n, c, *([1] * spatial_dim)).expand_as(map)
        return expanded_vec
         
class Text_encoder(nn.Module):
    def __init__(self, embedding_tokens, question_encode_length):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=embedding_tokens, 
                                    embedding_dim=300, padding_idx=0)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size = 300, 
                            hidden_size = question_encode_length,
                            num_layers = 1)

        self.question_encode_length = question_encode_length
        
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        torch.nn.init.zeros_(self.lstm.bias_ih_l0)
        torch.nn.init.zeros_(self.lstm.bias_hh_l0)
        
    def forward(self, q, q_len):
        embeded_q = self.embedding(q)
        x = self.tanh(self.dropout(embeded_q))
        packed_x = nn.utils.rnn.pack_padded_sequence(x, q_len, batch_first=True)
        
        _, (_, h) = self.lstm(packed_x)                      #cell state or hidden state?

        return h.squeeze(0)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            nn.init.xavier_uniform_(w)