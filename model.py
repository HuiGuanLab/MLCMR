import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformer_ori import BertAttention, TrainablePositionalEncoding, LinearLayer
from transformers import BertModel
# from ptflops import get_model_complexity_info
import numpy as np
from loss import TripletLoss,infoNCELoss
from basic.bigfile import BigFile
from collections import OrderedDict


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we)) 
    return np.array(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                print("fc_layers[0]:",fc_layers[0],"fc_layers[1]:",fc_layers[1])
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
          
            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch normalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features


class Video_multilevel_encoding(nn.Module):

    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        self.loss_fun = opt.loss_fun

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.visual_kernel_sizes
                ])

        
    def forward(self, videos):
        """Extract video feature vectors."""
        videos, videos_origin, lengths, videos_mask = videos
        
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        if self.gru_pool == 'mean':
            mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(gru_init_out):
                mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
            gru_out = mean_gru
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, videos_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        videos_mask = videos_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * videos_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full':
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)

class Text_bert_encoding(nn.Module):

    def __init__(self, opt):
        super(Text_bert_encoding, self).__init__()
        self.dropout = nn.Dropout(p=opt.dropout)
        self.txt_bert_params = {
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
        }

        txt_bert_config = 'bert-base-multilingual-cased'

        self.text_bert = BertModel.from_pretrained(txt_bert_config,return_dict=True, **self.txt_bert_params)


    def forward(self, text, *args):
        bert_caps, cap_mask = text
        batch_size, max_text_words = bert_caps.size()

        token_type_ids_list = []  # Modality id
        position_ids_list = []  # Position

        ids_size = (batch_size,)

        for pos_id in range(max_text_words):
            token_type_ids_list.append(torch.full(ids_size, 0, dtype=torch.long))
            position_ids_list.append(torch.full(ids_size, pos_id, dtype=torch.long))

        token_type_ids = torch.stack(token_type_ids_list, dim=1).cuda()
        position_ids = torch.stack(position_ids_list, dim=1).cuda()
        text_bert_output = self.text_bert(bert_caps,
                                        attention_mask=cap_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=None)

        del text
        torch.cuda.empty_cache()
        return text_bert_output

class Text_transformer_encoding_not_share(nn.Module):
    def __init__(self, opt):
        super(Text_transformer_encoding_not_share, self).__init__()
        self.max_ctx_l = 250
        self.bert_out_dim = 768
        self.input_drop = 0.1
        self.hidden_size = opt.text_hidden_size
        self.num_attention_heads = opt.text_num_attention
        self.layer = opt.text_layer

        self.text_bert = Text_bert_encoding(opt)

        self.pooling = opt.text_pooling

    def forward(self, texts,mode="multi"):
        if mode=="multi":
            text_en, text_zh = texts
            bert_caps_en,  cap_mask_en = text_en
            bert_caps_zh,  cap_mask_zh = text_zh

            # EN
            bert_out_en = self.text_bert((bert_caps_en, cap_mask_en))
            bert_seq_en = bert_out_en.last_hidden_state
            if self.pooling == 'mean':
                feat_en_pool = F.avg_pool1d(bert_seq_en.permute(0, 2, 1), bert_seq_en.size(1)).squeeze(2)
            # ZH
            bert_out_zh = self.text_bert((bert_caps_zh, cap_mask_zh))
            bert_seq_zh = bert_out_zh.last_hidden_state
            if self.pooling == 'mean':
                feat_zh_pool = F.avg_pool1d(bert_seq_zh.permute(0, 2, 1), bert_seq_zh.size(1)).squeeze(2)
            
            return feat_en_pool, feat_zh_pool
        elif mode=="single":
            bert_caps_en,  cap_mask_en = texts
            # EN
            bert_out_en = self.text_bert((bert_caps_en, cap_mask_en))
            bert_seq_en = bert_out_en.last_hidden_state
            if self.pooling == 'mean':
                feat_en_pool = F.avg_pool1d(bert_seq_en.permute(0, 2, 1), bert_seq_en.size(1)).squeeze(2)
            
            return feat_en_pool




class Latent_mapping_video(nn.Module):
    """
    Latent space mapping (Conference version)
    """
    def __init__(self, mapping_layers, dropout, l2norm=True):
        super(Latent_mapping_video, self).__init__()
        
        self.l2norm = l2norm
        # visual mapping
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)


    def forward(self, features):

        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)
        return latent_features
    
class Latent_mapping_text(nn.Module):
    """
    Latent space mapping (Conference version)
    """
    def __init__(self, mapping_layers, dropout, l2norm=True):
        super(Latent_mapping_text, self).__init__()
        self.l2norm = l2norm
        # visual mapping
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)
                                                         

    def forward(self, features,mode="multi"):

        # mapping to latent space
        if mode=="multi":
            features_en,features_zh=features
            latent_features_en = self.mapping(features_en)
            latent_features_zh = self.mapping(features_zh)
            if self.l2norm:
                latent_features_en = l2norm(latent_features_en)
                latent_features_zh = l2norm(latent_features_zh)
            return latent_features_en,latent_features_zh
        elif mode=="single":
            latent_features = self.mapping(features)
            if self.l2norm:
                latent_features = l2norm(latent_features)
            return latent_features




class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(),self.text_encoding_bert.state_dict(), self.vid_mapping.state_dict(), self.text_mapping.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding_bert.load_state_dict(state_dict[1])
        self.vid_mapping.load_state_dict(state_dict[2])
        self.text_mapping.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding_bert.train()
        self.vid_mapping.train()
        self.text_mapping.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding_bert.eval()
        self.vid_mapping.eval()
        self.text_mapping.eval()


    def init_info(self):

        # init gpu
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding_bert.cuda()
            self.vid_mapping.cuda()
            self.text_mapping.cuda()
            cudnn.benchmark = True

        text_param = []
        bert_name = 'text_bert.text_bert'
        layer_list = ['layer.11', 'layer.10','layer.9','layer.8','layer.7','layer.6']
        for name, param in self.text_encoding_bert.named_parameters():
            if bert_name in name and not any(layer in name for layer in layer_list):
                param.requires_grad = False
            else:
                text_param.append(param)


        # init params
        
        params = list(self.vid_encoding.parameters())
        params += text_param
        params += list(self.vid_mapping.parameters())
        params += list(self.text_mapping.parameters())
        self.params = params
        # param_count = 0
        # for param in params:
        #     param_count += param.view(-1).size()[0]

        # print structure
        print(self.vid_encoding)
        print(self.text_encoding_bert)
        print(self.vid_mapping)
        print(self.text_mapping)




class MLCMR(BaseModel):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.vid_encoding = Video_multilevel_encoding(opt)
        self.text_encoding_bert = Text_transformer_encoding_not_share(opt)

        self.vid_mapping = Latent_mapping_video(opt.visual_mapping_layers, opt.dropout)
        self.text_mapping = Latent_mapping_text(opt.text_mapping_layers, opt.dropout)
        self.init_info()
        self.opt=opt

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation,
                                            cost_style=opt.cost_style,
                                         direction=opt.direction)
            self.constractiveloss = infoNCELoss(margin=opt.margin,
                                            temp=0.07,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation,
                                            cost_style=opt.cost_style,
                                         direction=opt.direction)
        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.Eiters = 0

    def forward_emb(self, videos, targets,mode="multi", volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        videos,videos_mask=videos
        frames, mean_origin, video_lengths, vidoes_mask = videos
        frames_mask, mean_origin_mask, video_lengths_mask, vidoes_mask_mask = videos_mask


        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()

        frames_mask = Variable(frames_mask, volatile=volatile)
        if torch.cuda.is_available():
            frames_mask = frames_mask.cuda()

        mean_origin_mask = Variable(mean_origin_mask, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin_mask = mean_origin_mask.cuda()

        vidoes_mask_mask = Variable(vidoes_mask_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask_mask = vidoes_mask_mask.cuda()

        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)
        videos_mask_data = (frames_mask, mean_origin_mask, video_lengths_mask, vidoes_mask_mask)

        if mode=="multi":
            target_en,target_zh,target_en_mask,target_zh_mask = targets
            # text data EN
            bert_caps_en, cap_masks_en = target_en
            if bert_caps_en is not None:
                bert_caps_en = Variable(bert_caps_en, volatile=volatile)
                if torch.cuda.is_available():
                    bert_caps_en = bert_caps_en.cuda()

            if cap_masks_en is not None:
                cap_masks_en = Variable(cap_masks_en, volatile=volatile)
                if torch.cuda.is_available():
                    cap_masks_en = cap_masks_en.cuda()
            text_en_data = (bert_caps_en, cap_masks_en)
            # text data EN MASK
            bert_caps_en_mask, cap_masks_en_mask = target_en_mask
            if bert_caps_en_mask is not None:
                bert_caps_en_mask = Variable(bert_caps_en_mask, volatile=volatile)
                if torch.cuda.is_available():
                    bert_caps_en_mask = bert_caps_en_mask.cuda()

            if cap_masks_en_mask is not None:
                cap_masks_en_mask = Variable(cap_masks_en_mask, volatile=volatile)
                if torch.cuda.is_available():
                    cap_masks_en_mask = cap_masks_en_mask.cuda()
            text_en_data_mask = (bert_caps_en_mask, cap_masks_en_mask)

            # text data ZH
            bert_caps_zh, cap_masks_zh = target_zh
            if bert_caps_zh is not None:
                bert_caps_zh = Variable(bert_caps_zh, volatile=volatile)
                if torch.cuda.is_available():
                    bert_caps_zh = bert_caps_zh.cuda()

            if cap_masks_zh is not None:
                cap_masks_zh = Variable(cap_masks_zh, volatile=volatile)
                if torch.cuda.is_available():
                    cap_masks_zh = cap_masks_zh.cuda()
            text_zh_data = (bert_caps_zh, cap_masks_zh)
            # text data ZH MASK
            bert_caps_zh_mask, cap_masks_zh_mask = target_zh_mask
            if bert_caps_zh_mask is not None:
                bert_caps_zh_mask = Variable(bert_caps_zh_mask, volatile=volatile)
                if torch.cuda.is_available():
                    bert_caps_zh_mask = bert_caps_zh_mask.cuda()

            if cap_masks_zh_mask is not None:
                cap_masks_zh_mask = Variable(cap_masks_zh_mask, volatile=volatile)
                if torch.cuda.is_available():
                    cap_masks_zh_mask = cap_masks_zh_mask.cuda()
            text_zh_data_mask = (bert_caps_zh_mask, cap_masks_zh_mask)


            text_data = (text_en_data, text_zh_data)
            text_mask_data = (text_en_data_mask, text_zh_data_mask)


            vid_emb=self.vid_encoding(videos_data)
            vid_emb_mask=self.vid_encoding(videos_mask_data)

            vid_emb=self.vid_mapping(vid_emb)
            vid_emb_mask=self.vid_mapping(vid_emb_mask)

            en_feat,zh_feat = self.text_encoding_bert(text_data)
            en_mask_feat,zh_mask_feat = self.text_encoding_bert(text_mask_data)

            feat=(en_feat,zh_feat)
            cap_emb_en,cap_emb_zh = self.text_mapping(feat)
            
            mask_feat=(en_mask_feat,zh_mask_feat)
            cap_emb_en_mask,cap_emb_zh_mask = self.text_mapping(mask_feat)

            return vid_emb,vid_emb_mask,cap_emb_en,cap_emb_zh,cap_emb_en_mask,cap_emb_zh_mask

        elif mode=="single":
            target_en,target_en_mask = targets
            # text data EN
            bert_caps_en, cap_masks_en = target_en
            if bert_caps_en is not None:
                bert_caps_en = Variable(bert_caps_en, volatile=volatile)
                if torch.cuda.is_available():
                    bert_caps_en = bert_caps_en.cuda()

            if cap_masks_en is not None:
                cap_masks_en = Variable(cap_masks_en, volatile=volatile)
                if torch.cuda.is_available():
                    cap_masks_en = cap_masks_en.cuda()
            text_en_data = (bert_caps_en, cap_masks_en)
            # text data EN MASK
            bert_caps_en_mask, cap_masks_en_mask = target_en_mask
            if bert_caps_en_mask is not None:
                bert_caps_en_mask = Variable(bert_caps_en_mask, volatile=volatile)
                if torch.cuda.is_available():
                    bert_caps_en_mask = bert_caps_en_mask.cuda()

            if cap_masks_en_mask is not None:
                cap_masks_en_mask = Variable(cap_masks_en_mask, volatile=volatile)
                if torch.cuda.is_available():
                    cap_masks_en_mask = cap_masks_en_mask.cuda()
            text_en_data_mask = (bert_caps_en_mask, cap_masks_en_mask)


            text_data = text_en_data
            text_mask_data = text_en_data_mask


            vid_emb=self.vid_encoding(videos_data)
            vid_emb_mask=self.vid_encoding(videos_mask_data)

            vid_emb=self.vid_mapping(vid_emb)
            vid_emb_mask=self.vid_mapping(vid_emb_mask)

            en_feat = self.text_encoding_bert(text_data,mode="single")
            en_mask_feat = self.text_encoding_bert(text_mask_data,mode="single")

            cap_emb_en = self.text_mapping(en_feat,mode="single")
            cap_emb_en_mask = self.text_mapping(en_mask_feat,mode="single")

            return vid_emb,vid_emb_mask,cap_emb_en,cap_emb_en_mask

    def embed_vis(self, vis_data, volatile=True):
        """Compute the video embeddings
        """
        # video data
        vis_data,vis_data_mask=vis_data
        frames, mean_origin, video_lengths, vidoes_mask = vis_data
        frames_mask, mean_origin_mask, video_lengths_mask, vidoes_mask_mask = vis_data_mask
        with torch.no_grad():
            frames = Variable(frames)
        if torch.cuda.is_available():
            frames = frames.cuda()
        with torch.no_grad():
            mean_origin = Variable(mean_origin)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()
        with torch.no_grad():
            vidoes_mask = Variable(vidoes_mask)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()

        with torch.no_grad():
            frames_mask = Variable(frames_mask)
        if torch.cuda.is_available():
            frames_mask = frames_mask.cuda()
        with torch.no_grad():
            mean_origin_mask = Variable(mean_origin_mask)
        if torch.cuda.is_available():
            mean_origin_mask = mean_origin_mask.cuda()
        with torch.no_grad():
            vidoes_mask_mask = Variable(vidoes_mask_mask)
        if torch.cuda.is_available():
            vidoes_mask_mask = vidoes_mask_mask.cuda()

        vis_data = (frames, mean_origin, video_lengths, vidoes_mask)
        vis =self.vid_encoding(vis_data)

        return self.vid_mapping(vis)


    def embed_txt(self, txt_data, volatile=True):
        """Compute the caption embeddings
        """
        # text data
        target_en,target_zh,target_en_mask,target_zh_mask = txt_data
        # EN
        bert_caps_en,cap_masks_en=target_en
        if bert_caps_en is not None:
            with torch.no_grad():
                bert_caps_en = Variable(bert_caps_en)
            if torch.cuda.is_available():
                bert_caps_en = bert_caps_en.cuda()

        if cap_masks_en is not None:
            with torch.no_grad():
                cap_masks_en = Variable(cap_masks_en)
            if torch.cuda.is_available():
                cap_masks_en = cap_masks_en.cuda()
        text_en_data = (bert_caps_en, cap_masks_en)

        # ZH 
        bert_caps_zh,cap_masks_zh=target_zh
        if bert_caps_zh is not None:
            with torch.no_grad():
                bert_caps_zh = Variable(bert_caps_zh)
            if torch.cuda.is_available():
                bert_caps_zh = bert_caps_zh.cuda()
        if cap_masks_zh is not None:
            with torch.no_grad():
                cap_masks_zh = Variable(cap_masks_zh)
            if torch.cuda.is_available():
                cap_masks_zh = cap_masks_zh.cuda()
        text_zh_data = (bert_caps_zh, cap_masks_zh)
        text_data = (text_en_data, text_zh_data)
        en_feat,zh_feat = self.text_encoding_bert(text_data)
        feat=en_feat,zh_feat
        cap_emb_en,cap_emb_zh = self.text_mapping(feat)
        return cap_emb_en,cap_emb_zh



    def forward_loss(self, cap_emb_en,cap_emb_zh,cap_emb_en_mask,cap_emb_zh_mask, vid_emb,vid_emb_mask, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss_en_vid = self.criterion(cap_emb_en, vid_emb)
        loss_zh_vid = self.criterion(cap_emb_zh, vid_emb)
        # loss_en_zh = self.criterion(cap_emb_en, cap_emb_zh)
        loss_en_en_mask = self.constractiveloss(cap_emb_en, cap_emb_en_mask)
        loss_zh_zh_mask = self.constractiveloss(cap_emb_zh, cap_emb_zh_mask)
        loss_en_mask_en = self.constractiveloss(cap_emb_en_mask, cap_emb_en)
        loss_zh_mask_zh = self.constractiveloss(cap_emb_zh_mask, cap_emb_zh)
        loss_vid_mask_vid = self.constractiveloss(vid_emb, vid_emb_mask)
        loss_vid_vid_mask = self.constractiveloss(vid_emb_mask, vid_emb)
        loss = loss_zh_vid+loss_en_vid+(loss_en_en_mask+loss_en_mask_en)/2+(loss_zh_zh_mask+loss_zh_mask_zh)/2+(loss_vid_mask_vid+loss_vid_vid_mask)/2
        self.logger.update('Le', loss.item(), vid_emb.size(0))

        return loss

    def forward_loss_single(self, cap_emb_en,cap_emb_en_mask, vid_emb,vid_emb_mask, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss_en_vid = self.criterion(cap_emb_en, vid_emb)
        # loss_en_zh = self.criterion(cap_emb_en, cap_emb_zh)
        loss_en_en_mask = self.constractiveloss(cap_emb_en, cap_emb_en_mask)
        loss_en_mask_en = self.constractiveloss(cap_emb_en_mask, cap_emb_en)
        loss_vid_mask_vid = self.constractiveloss(vid_emb, vid_emb_mask)
        loss_vid_vid_mask = self.constractiveloss(vid_emb_mask, vid_emb)
        loss = loss_en_vid+(loss_en_en_mask+loss_en_mask_en)/2+(loss_vid_mask_vid+loss_vid_vid_mask)/2
        self.logger.update('Le', loss.item(), vid_emb.size(0))

        return loss

    def train_emb(self, videos, captions, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        # compute the embeddings
        if self.opt.train_mode == "parallel":
            vid_emb,vid_emb_mask, cap_emb_en,cap_emb_zh,cap_emb_en_mask,cap_emb_zh_mask  = self.forward_emb(videos, captions, volatile=False)
            self.optimizer.zero_grad()
            loss = self.forward_loss(cap_emb_en,cap_emb_zh,cap_emb_en_mask,cap_emb_zh_mask,vid_emb,vid_emb_mask)
        elif self.opt.train_mode == "unparallel":
            vid_emb,vid_emb_mask, cap_emb_en,cap_emb_en_mask= self.forward_emb(videos, captions,mode="single", volatile=False)
            self.optimizer.zero_grad()
            loss = self.forward_loss_single(cap_emb_en,cap_emb_en_mask,vid_emb,vid_emb_mask)
        loss_value = loss.item()
        # measure accuracy and record loss

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value
    


NAME_TO_MODELS = {'mlcmr': MLCMR}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
