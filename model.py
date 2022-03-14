import torch
from torch import nn
import numpy as np
from torch.nn import init
from transformers import BertModel, RobertaModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertForSequenceClassification
from transformers.models.roberta import RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss

#remote_sever
def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)

    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (
                x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)


def nx_xent(x, x_adv, mask, cuda=True, t=0.5):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()
    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (
                    x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (
                    x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (
                    x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (
                    x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    return -loss.mean()


class SCLModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SCLModel, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.T = 0.3

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        labels = labels
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        P_mask = labels[:, None]==labels[None]
        N_mask = P_mask.eq(0)

        # S = torch.matmul(logits, logits.transpose(0, 1)).exp()
        S = torch.matmul(logits, logits.transpose(0, 1))/self.T
        S = S.exp()

        pos_scores = S*P_mask
        neg_scores = (S*N_mask).sum(dim=-1, keepdim=True)
        no_neg_mask = N_mask.sum(dim=-1, keepdim=True).eq(0)
        loss_con = (pos_scores.masked_fill(no_neg_mask, 0) / (neg_scores + 1e-12)).masked_fill(N_mask, 0).sum()/P_mask.sum()

        loss_cls = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss_cls = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss_cls = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        loss = loss_con * self.config.contrastive_rate_in_training + \
               loss_cls * (1 - self.config.contrastive_rate_in_training)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassification(nn.Module):

    def __init__(self, config):
        super(RobertaClassification, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class BertClassificationHead(nn.Module):
    def __init__(self, config):
        super(BertClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels - 1)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertClassification(nn.Module):
    def __init__(self, config):
        super(BertClassification, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.number_labels)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.out_proj(x)
        return x


class BertContrastiveHead(nn.Module):
    def __init__(self, config):
        super(BertContrastiveHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertContrastive(nn.Module):
    def __init__(self, config):
        super(BertContrastive, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.out_proj(x)
        return x


class RobertaContrastiveHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ContrastiveMoCoKnnBert(nn.Module):

    def __init__(self, config):
        super(ContrastiveMoCoKnnBert, self).__init__()
        self.config = config
        self.number_labels = config.num_labels
        if config.load_trained_model: # False
            self.encoder_q = BertModel(config)
            self.encoder_k = BertModel(config)
        else:
            self.encoder_q = BertModel.from_pretrained(config.model_name, config=config)
            self.encoder_k = BertModel.from_pretrained(config.model_name, config=config)

        self.classifier_liner = BertClassificationHead(config)

        self.contrastive_liner_q = BertContrastiveHead(config)
        self.contrastive_liner_k = BertContrastiveHead(config)

        self.m = 0.999
        self.T = config.T
        self.train_multi_head = config.train_multi_head #False
        self.multi_head_num = config.multi_head_num #32

        if not config.load_trained_model:
            self.init_weights() # Exec

        # create the label_queue and feature_queue
        self.K = config.queue_size # 7500

        self.register_buffer("label_queue", torch.randint(0, self.number_labels, [self.K])) # Tensor:(7500,)
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size)) # Tensor:(7500, 768)
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # Tensor(1,)
        self.top_k = config.knn_num # 25
        self.update_num = config.positive_num # 3


        # optional and delete can improve the performance indicated 
        # by some experiment
        params_to_train = ["layer." + str(i) for i in range(0,12)]
        for name, param in self.encoder_q.named_parameters():
            param.requires_grad_(False)
            for term in params_to_train:
                if term in name:
                    param.requires_grad_(True)


    def _dequeue_and_enqueue(self, keys, label):
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[: batch_size]
            label = label[: batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[ptr: ptr + batch_size, :] = keys
        self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_multi_head_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()  # K
        feature_queue = self.feature_queue.clone().detach()  # K * hidden_size

        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1])  # batch_size * K * hidden_size

        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        pos_sample_last = pos_sample[:, -1]
        pos_sample_last = pos_sample_last.view([-1, 1])

        pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k + 1])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1. expand label_queue and feature_queue to batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2.caluate sim
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3. get index of postive and neigative 
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4.another option
        #feature_value = cos_sim.masked_select(pos_mask_index)
        #pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        #pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5.topk
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)
        #pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k] # self.topk = 25
        #pos_sample_top_k = pos_sample[:, 0:self.top_k]
        #pos_sample_last = pos_sample[:, -1]
        ##pos_sample_last = pos_sample_last.view([-1, 1])
        pos_sample = pos_sample_top_k
        #pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.contiguous().view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self.top_k)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con


    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward_no_multi_v2(self,
                            query,
                            positive_sample=None,
                            negative_sample=None,
                            ):
        labels = query["labels"]
        labels = labels.view(-1)

        with torch.no_grad():
            self.update_encoder_k()
            update_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            update_keys = self.contrastive_liner_k(update_keys)
            update_keys = l2norm(update_keys)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.repeat([1, self.update_num])
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)
        query.pop('labels')
        query.pop('original_text')
        query.pop('sent_id')

        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        liner_q = l2norm(liner_q)
        logits_cls = self.classifier_liner(q)

        if self.number_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.number_labels - 1), labels)

        logits_con = self.select_pos_neg_sample(liner_q, labels)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
        else:
            loss = loss_cls

        return SequenceClassifierOutput(
            loss=loss,
        )

    def forward_by_multi_v2(self,
                            query,
                            positive_sample=None,
                            negative_sample=None,
                            ):
        labels = query["labels"]
        labels = labels.view(-1)
        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        logits_cls = self.classifier_liner(liner_q)

        if self.number_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.number_labels), labels)

        logits_con = self.select_pos_neg_sample(liner_q, labels)
        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
        else:
            loss = loss_cls

        with torch.no_grad():
            self.update_encoder_k()
            update_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            update_keys = self.contrastive_liner_k(update_keys)
            update_keys = l2norm(update_keys)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.repeat([1, self.update_num])
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)

        return SequenceClassifierOutput(
            loss=loss,
        )


    def forward(self,
                query,# batch_size * max_length
                mode,
                positive_sample=None,        # batch_size * max_length
                negative_sample=None,        # batch_size * sample_num * max_length
                ):
        if mode == 'train':
            if self.train_multi_head: # False
                return self.forward_by_multi_head(query=query, positive_sample=positive_sample, negative_sample=negative_sample)
            return self.forward_no_multi_v2(query=query, positive_sample=positive_sample, negative_sample=negative_sample)
        elif mode == 'validation':
            labels = query['labels']
            labels_one_hot = nn.functional.one_hot(labels, self.number_labels - 1).float()
            query.pop('labels')
            query.pop('original_text')
            query.pop('sent_id')
            seq_embed = self.encoder_q(**query)[1]

            #q = bert_output_q[1]
            #liner_q = self.contrastive_liner_q(seq_embed)
            #liner_q = l2norm(liner_q)
            logits_cls = self.classifier_liner(seq_embed)

            #seq_embed = self.dropout(seq_embed)
            #seq_embed = seq_embed.clone().detach().requires_grad_(True).float()

            #_, ht = self.rnn_pooler(seq_embed)
            #ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            #logits = self.classifier_liner_q(ht)
            #logits = self.fc(ht)
            probs = torch.softmax(logits_cls, dim=1)
            return torch.argmax(probs, dim=1).tolist(), labels.cpu().numpy().tolist()
            #return torch.argmax(labels_one_hot, dim=1).tolist(), torch.argmax(probs, dim=1).tolist(), ht
        elif mode == 'test':
            query.pop('labels')
            query.pop('original_text')
            query.pop('sent_id')
            seq_embed = self.encoder_q(**query)[1]
            #seq_embed = self.dropout(seq_embed)
            #seq_embed = seq_embed.clone().detach().requires_grad_(True).float()

            #_, ht = self.rnn_pooler(seq_embed)
            #ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            #logits = self.classifier_liner_q(ht)
            #logits = self.fc(ht)
            #liner_q = self.contrastive_liner_q(seq_embed)
            #liner_q = l2norm(liner_q)
            logits_cls = self.classifier_liner(seq_embed)
            #logits_cls = self.classifier_liner(seq_embed)

            probs = torch.softmax(logits_cls, dim=1)
            return probs, seq_embed
        else:
            raise ValueError("undefined mode")

    # eval
    def predict(self, query):
        with torch.no_grad():
            bert_output_q = self.encoder_q(**query)
            q = bert_output_q[1]
            logits_cls = self.classifier_liner(q)
            contrastive_output = self.contrastive_liner_q(q)
        return contrastive_output, logits_cls

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            contrastive_output = self.contrastive_liner_k(bert_output_k[1])
        return contrastive_output

# just trying something and has been abandoned.
class ContrastiveOrigin(nn.Module):
    def __init__(self, config):
        super(ContrastiveOrigin, self).__init__()
        self.config = config
        self.bert_model = BertModel.from_pretrained(config.model_name, config=config)
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_labels - 1)
        self.dropout = nn.Dropout(p=0.5)
        self.lmcl = config.lmcl
        self.norm_coef = config.norm_coef
        self.cl_mode = config.cl_mode
        self.number_labels = config.num_labels
        self.rnn_pooler = nn.GRU(input_size=768, hidden_size=config.hidden_dim, num_layers=config.rnn_number_layers,
                                 batch_first=True, bidirectional=True).to(config.device)
        for name, param in self.bert_model.named_parameters():
            if name.startswith('pooler'):
                continue
            else:
                param.requires_grad_(False)

    # label may be defined different from this codes
    def lmcl_loss(self, probs, label, margin=0.35, scale=30):
        probs = label * (probs - margin) + (1 - label) * probs
        probs = torch.softmax(probs, dim=1)

        return probs

    def forward(self, query, temperature=0.3, stage='sup', mode='ind_pre'):
        if mode == 'ind_pre' or mode == 'finetune':
            labels = query['labels']
            labels_one_hot = nn.functional.one_hot(labels, self.number_labels - 1).float()
            query.pop('labels')
            query.pop('original_text')
            query.pop('sent_id')
            seq_embed = self.bert_model(**query)[0]
            seq_embed = self.dropout(seq_embed)
            seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            _, ht = self.rnn_pooler(seq_embed)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            if self.lmcl:
                probs = self.lmcl_loss(logits, labels_one_hot)
            else:
                probs = torch.softmax(logits, dim=1)
            ce_loss = torch.sum(torch.mul(-torch.log(probs), labels_one_hot))
            if stage is not 'sup' or mode == 'finetune':
                return ce_loss
            else:
                seq_embed.retain_grad()
                ce_loss.backward(retain_graph=True)
                unnormalized_noise = seq_embed.grad.detach_()
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                norm = unnormalized_noise.norm(p=2, dim=-1)
                normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)
                noise_embedding = seq_embed + self.norm_coef * normalized_noise

                _, h_adv = self.rnn_pooler(noise_embedding)
                h_adv = torch.cat((h_adv[0].squeeze(0), h_adv[1].squeeze(0)), dim=1)
                label_mask = torch.mm(labels_one_hot, labels_one_hot.T).bool().long()
                sup_cont_loss = nx_xent(ht, h_adv, label_mask, cuda=True, t=temperature)
                return sup_cont_loss
        elif mode == 'validation':
            labels = query['labels']
            labels_one_hot = nn.functional.one_hot(labels, self.number_labels - 1).float()
            query.pop('labels')
            query.pop('original_text')
            query.pop('sent_id')
            seq_embed = self.bert_model(**query)[0]
            seq_embed = self.dropout(seq_embed)
            seq_embed = seq_embed.clone().detach().requires_grad_(True).float()

            _, ht = self.rnn_pooler(seq_embed)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            probs = torch.softmax(logits, dim=1)
            return torch.argmax(labels_one_hot, dim=1).tolist(), torch.argmax(probs, dim=1).tolist(), ht
        elif mode == 'test':
            query.pop('labels')
            query.pop('original_text')
            query.pop('sent_id')
            seq_embed = self.bert_model(**query)[0]
            seq_embed = self.dropout(seq_embed)
            seq_embed = seq_embed.clone().detach().requires_grad_(True).float()

            _, ht = self.rnn_pooler(seq_embed)
            ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.fc(ht)
            probs = torch.softmax(logits, dim=1)
            return probs, ht
        else:
            raise ValueError("undefined mode")

# just tring something and has been abandon.
class ContrastiveMoCoKnnInitByBert(nn.Module):

    def __init__(self, config):
        super(ContrastiveMoCoKnnInitByBert, self).__init__()
        self.config = config
        self.number_labels = config.num_labels
        if config.load_trained_model:
            self.encoder_q = BertModel(config)
            self.encoder_k = BertModel(config)
        else:
            self.encoder_q = BertModel.from_pretrained(config.model_name, config=config)
            self.encoder_k = BertModel.from_pretrained(config.model_name, config=config)

        self.classifier_liner = BertClassificationHead(config)

        self.contrastive_liner_q = BertContrastiveHead(config)
        self.contrastive_liner_k = BertContrastiveHead(config)

        self.m = 0.999
        self.T = 0.3
        self.train_multi_head = config.train_multi_head
        self.multi_head_num = config.multi_head_num

        if not config.load_trained_model:
            self.init_weights()

        # create the label_queue and feature_queue
        self.K = config.queue_size

        self.register_buffer("label_queue", torch.randint(0, self.number_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.top_k = config.knn_num
        self.update_num = config.positive_num

    def _dequeue_and_enqueue(self, keys, label):
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[: batch_size]
            label = label[: batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[ptr: ptr + batch_size, :] = keys
        self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_multi_head_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()  # K
        feature_queue = self.feature_queue.clone().detach()  # K * hidden_size

        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1])  # batch_size * K * hidden_size

        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        pos_sample_last = pos_sample[:, -1]
        pos_sample_last = pos_sample_last.view([-1, 1])

        pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k + 1])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        pos_sample_last = pos_sample[:, -1]
        pos_sample_last = pos_sample_last.view([-1, 1])

        pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k + 1])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def select_negative_sample(self, label): # batch * 1
        label_queue = self.label_queue.clone().detach() # train_size * 1
        feature_queue = self.feature_queue.clone().detach()
        
        batch_size = label.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])

        tmp_label = label.unsqueeze(1)
        tmp_label = tmp_label.repeat((1, self.K))

        mask_index = torch.ne(tmp_label_queue, tmp_label)

        tmp_feature = feature_queue.unsqueeze(0)
        tmp_feature = tmp_feature.repeat([batch_size, 1, 1]) # batch_size * K * hidden_zise
        tmp_index = mask_index.unsqueeze(-1)
        tmp_index = tmp_index.repeat([1, 1, self.config.hidden_size])
        feature_value = tmp_feature.masked_select(tmp_index)

        negative_sample = torch.zeros([batch_size, self.K, self.config.hidden_size]).to("cuda")
        negative_sample = negative_sample.masked_scatter(tmp_index, feature_value)

        tmp_index = ~ tmp_index

        feature_value = tmp_feature.masked_select(tmp_index)

        positive_sample = torch.zeros([batch_size, self.K, self.config.hidden_size]).to("cuda")
        positive_sample = positive_sample.masked_scatter(tmp_index, feature_value)

        return positive_sample, negative_sample

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def update_queue_by_bert(self,
                             inputs=None,        # batch_size * max_length
                             labels = None,
                             ):

        with torch.no_grad():
            update_sample = self.reshape_dict(inputs)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            update_keys = self.contrastive_liner_k(update_keys)
            update_keys = l2norm(update_keys)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward_by_multi_head(self,
                              query,
                              positive_sample=None,
                              negative_sample=None,
                              ):

        labels = query["labels"]
        labels = labels.view(-1)
        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        hidden_size = self.config.hidden_size
        multi_head_size = int(hidden_size/self.multi_head_num)
        liner_q = self.contrastive_liner_q(q)
        liner_q = liner_q.view([-1, multi_head_size])

        logits_cls = self.classifier_liner(q)

        loss_cls = None
        if labels is not None:
            if self.number_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss_cls = loss_fct(logits_cls.view(-1), labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss_cls = loss_fct(logits_cls.view(-1, self.number_labels), labels)

        with torch.no_grad():
            self.update_encoder_k()
            positive_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**positive_sample)
            p = bert_output_p[1]
            liner_p = self.contrastive_liner_k(p)
            liner_p_multi = liner_p.view([-1, multi_head_size])

            liner_n = self.select_negative_sample(labels)

            liner_n = liner_n.view([-1, self.K, self.multi_head_num, multi_head_size])
            liner_n = liner_n.transpose(1, 2)

            liner_n = liner_n.reshape([-1, self.K, multi_head_size])

        liner_p_multi = l2norm(liner_p_multi)
        liner_q = l2norm(liner_q)
        liner_n = l2norm(liner_n)

        liner_q_pos = liner_q.repeat([1, self.positive_num])
        liner_q_pos = liner_q_pos.view([-1, multi_head_size])

        l_pos = torch.einsum('nc,nc->n', [liner_q_pos, liner_p_multi]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum('nc,nkc->nk', [liner_q, liner_n])  # TODO 这里应该有问题需要修改

        l_neg = l_neg.repeat([1, self.positive_num])
        l_neg = l_neg.view([-1, self.K])
        # logits: Nx(1+K)   N = batch_size*24
        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        # labels: positive key indicators
        con_labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_fct = CrossEntropyLoss()
        loss_con = loss_fct(logits, con_labels)

        tmp_labels = labels.unsqueeze(-1)
        tmp_labels = tmp_labels.repeat([1, self.positive_num])
        tmp_labels = tmp_labels.view(-1)

        self._dequeue_and_enqueue(liner_p, tmp_labels)

        if loss_con is None:
            return loss_cls
        if loss_cls is None:
            return loss_con
        return loss_con * self.config.contrastive_rate_in_training + loss_cls * (1-self.config.contrastive_rate_in_training)

    def forward_no_multi_head(self,
                              query,
                              positive_sample=None,        # batch_size * max_length
                              negative_sample=None,        # batch_size * sample_num * max_length
                              ):
        labels = query["labels"]
        labels = labels.view(-1)
        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]  # batch * 768
        liner_q = self.contrastive_liner_q(q)
        logits_cls = self.classifier_liner(q)

        loss_cls = None
        if labels is not None:
            if self.number_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss_cls = loss_fct(logits_cls.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss_cls = loss_fct(logits_cls.view(-1, self.number_labels), labels.view(-1))

        with torch.no_grad():
            liner_p, liner_n = self.select_negative_sample(labels)
            self.update_encoder_k()

            if self.update_num == 1:
                bert_output_p = self.encoder_k(**query)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = l2norm(update_keys)
                self._dequeue_and_enqueue(update_keys, labels)
            else:
                update_sample = self.reshape_dict(positive_sample)
                bert_output_p = self.encoder_k(**update_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = l2norm(update_keys)
                tmp_labels = labels.unsqueeze(-1)
                tmp_labels = tmp_labels.repeat([1, self.update_num])
                tmp_labels = tmp_labels.view(-1)
                self._dequeue_and_enqueue(update_keys, tmp_labels)

        liner_q = l2norm(liner_q)

        l_pos = torch.einsum('nc,nkc->nk', [liner_q, liner_p])
        value_index = torch.ne(l_pos, 0.0)
        mask_index = ~ value_index
        value_index = value_index.int()
        value_number = value_index.sum(dim=-1)
        pos_min = int(value_number.min())
        # pos_min = min(pos_min, 1000)
        l_pos = l_pos.masked_fill(mask_index, -np.inf)
        l_pos, _ = l_pos.topk(pos_min, dim=-1)
        l_pos_top_100 = l_pos[:, 0:self.top_k]

        l_pos_last = l_pos[:, -1]
        l_pos_last = l_pos_last.view([-1, 1])

        l_pos = torch.cat([l_pos_top_100, l_pos_last], dim=-1)
        l_pos = l_pos.view([-1, 1])

        l_neg = torch.einsum('nc,nkc->nk', [liner_q, liner_n])
        value_index = torch.ne(l_neg, 0.0)
        mask_index = ~ value_index
        value_index = value_index.int()
        value_number = value_index.sum(dim=-1)
        neg_min = int(value_number.min())
        l_neg = l_neg.masked_fill(mask_index, -np.inf)
        l_neg, _ = l_neg.topk(neg_min, dim=-1)
        l_neg = l_neg.repeat([1, self.positive_num + 1])
        l_neg = l_neg.view([-1, neg_min])

        logits_con = torch.cat([l_pos, l_neg], dim=1)
        logits_con /= self.T

        labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()

        loss_fct = CrossEntropyLoss()
        loss_con = loss_fct(logits_con, labels_con)

        loss = loss_con * self.config.contrastive_rate_in_training + \
               loss_cls * (1 - self.config.contrastive_rate_in_training)

        # loss = loss_con

        return SequenceClassifierOutput(
            loss=loss,
        )

    def forward_no_multi_v2(self,
                            query,
                            positive_sample=None,
                            negative_sample=None,
                            ):
        labels = query["labels"]
        labels = labels.view(-1)
        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        # liner_q = self.contrastive_liner_q(q)
        liner_q = q
        logits_cls = self.classifier_liner(q)

        if self.number_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.number_labels), labels)

        logits_con = self.select_pos_neg_sample(liner_q, labels)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
        else:
            loss = loss_cls

        with torch.no_grad():
            self.update_encoder_k()
            update_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            # update_keys = self.contrastive_liner_k(update_keys)
            update_keys = l2norm(update_keys)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.repeat([1, self.update_num])
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)

        return SequenceClassifierOutput(
            loss=loss,
        )

    def forward_by_multi_v2(self,
                            query,
                            positive_sample=None,
                            negative_sample=None,
                            ):
        labels = query["labels"]
        labels = labels.view(-1)
        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        logits_cls = self.classifier_liner(q)

        if self.number_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.number_labels), labels)

        logits_con = self.select_pos_neg_sample(liner_q, labels)
        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * self.config.contrastive_rate_in_training + \
                   loss_cls * (1 - self.config.contrastive_rate_in_training)
        else:
            loss = loss_cls

        with torch.no_grad():
            self.update_encoder_k()
            update_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            update_keys = self.contrastive_liner_k(update_keys)
            update_keys = l2norm(update_keys)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.repeat([1, self.update_num])
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)

        return SequenceClassifierOutput(
            loss=loss,
        )


    def forward(self,
                query,                  # batch_size * max_length
                positive_sample=None,        # batch_size * max_length
                negative_sample=None,        # batch_size * sample_num * max_length
                ):
        if self.train_multi_head:
            return self.forward_by_multi_head(query=query, positive_sample=positive_sample, negative_sample=negative_sample)
        return self.forward_no_multi_v2(query=query, positive_sample=positive_sample, negative_sample=negative_sample)

    def predict(self, query):
        with torch.no_grad():
            bert_output_q = self.encoder_q(**query)
            q = bert_output_q[1]
            logits_cls = self.classifier_liner(q)
            contrastive_output = self.contrastive_liner_q(q)
        return contrastive_output, logits_cls

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            contrastive_output = self.contrastive_liner_k(bert_output_k[1])
        return contrastive_output


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


