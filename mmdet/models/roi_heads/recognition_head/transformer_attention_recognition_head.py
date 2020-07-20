import torch
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch import nn
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import torch.nn.functional as F
from mmdet.models.builder import HEADS


@HEADS.register_module()
class TransformerAttnRecognizer(nn.Module):

    def __init__(self,
                 in_channel,
                 encoder,
                 d_model,
                 n_head,
                 seq_len=25,
                 num_char=97,
                 dim_feadforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerAttnRecognizer, self).__init__()
        self.seq_len = seq_len

        if isinstance(encoder, str) and encoder == 'CnnEncoder':
            self.encoder = CnnEncoder(in_channel, d_model)
        else:
            self.encoder = None
        self.char_embedding = Embedding(d_model, num_char)
        self.decoderlayer = TransformerDecoderLayer(d_model,
                                                    n_head,
                                                    dim_feadforward,
                                                    dropout,
                                                    activation)
        self.decoder = TransformerDecoder(self.decoderlayer, 1)
        self.output_fc = nn.Linear(d_model, num_char)

        self.START_TOKEN = num_char - 3
        self.PAD_TOKEN = num_char - 2
        self_attn_mask = torch.ones((seq_len, seq_len))
        self.self_attn_mask = torch.tril(self_attn_mask, diagonal=0)\
            .masked_fill(self_attn_mask == 0, float('-inf')).masked_fill(self_attn_mask == 1, float(0.0))

    def forward_step(self, vis_seq, text_seq, text_mask=None):
        embedded_labels = self.char_embedding(text_seq) # N * T * d_model
        embedded_labels = embedded_labels.permute(1,0,2)
        output = self.decoder(embedded_labels, vis_seq, tgt_mask=self.self_attn_mask.to(embedded_labels.devices), tgt_key_padding_mask=text_mask)
        output = self.output_fc(output)
        return output

    def forward(self, roi_features, text_labels=None, text_masks=None):
        '''

        :param text_masks:
        :param roi_features: N * C * h * w
        :param text_labels: N * t
        :return:
        '''
        roi_features = self.encoder(roi_features) # N * d_model * h * w
        N, d_model, h, w = roi_features.shape
        roi_features = roi_features.permute(2,3,0,1).view(-1, N, d_model)
        if self.training:
            assert text_labels is not None
            text_labels[:, 1:] = text_labels[:, :-1]
            start_padding = torch.zeros(N, dtype=torch.long, device=text_labels.device).fill_(self.START_TOKEN)
            text_labels[:, 0] = start_padding
            text_masks[:, 1:] = text_masks[:, -1]

            logits = self.forward_step(roi_features, text_labels, text_masks) # T * N * num_char
            return logits.permute(1,0,2)
        else:
            text_seq = (torch.zeros((N, self.seq_len), dtype=torch.long, device=text_labels.device).fill_(self.PAD_TOKEN))
            pre_pred = torch.zeros(N, dtype=torch.long, device=text_labels.device).fill_(self.START_TOKEN)
            for t in range(self.seq_len):
                text_seq[:,t] = pre_pred
                logits = self.forward_step(roi_features, text_seq)
                current_logits = logits[t, :, :]
                pre_pred = torch.argmax(current_logits, dim=1)
            return logits.permute(1,0,2)

    def loss(self, pred_logits, text_labels, text_masks):
        '''
        calculate recognition loss
        :param pred_logits: N * T * num_char
        :param text_labels: N * T
        :param text_masks:  N * T
        :return: loss: dict
        '''
        N = pred_logits.shape[0]
        input_labels = text_labels.view(-1)
        pred_logits = pred_logits.view(-1, self.voc_len)
        input_mask = text_masks.view(-1)

        loss = dict()
        recog_loss = F.cross_entropy(pred_logits, input_labels, reduction='none')
        loss['recog loss'] = (input_mask.to(recog_loss.device) * loss).sum() / N
        return loss

    def init_weights(self):
        self.encoder.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


class CnnEncoder(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


class Embedding(nn.Module):

    def __init__(self,
                 d_model,
                 num_char=98,
                 dropout=0.1,):
        super(Embedding, self).__init__()
        self.letter_embedding = nn.Embedding(num_char, d_model)
        # self.pos_embed = nn.Embedding(cfg['pos_dim'], cfg['dim'])  # position embedding
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        E = d_model
        maxlen = 25
        position_enc = torch.tensor([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)], dtype=torch.float32)

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1
        self.pos_embed = nn.Embedding.from_pretrained(position_enc)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)  # (S,) -> (B, S)

        e = self.letter_embedding(x) + self.pos_embed(pos)
        return self.drop(self.norm(e))


