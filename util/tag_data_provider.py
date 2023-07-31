import json
import torch
import torch.utils.data as data
import numpy as np
import copy
from basic.util import getVideoId
from util.vocab import clean_str

from transformers import BertTokenizer

VIDEO_MAX_LEN=64

def create_tokenizer():
    model_name_or_path = 'bert-base-multilingual-cased'
    do_lower_case = True
    cache_dir = 'data/cache_dir'
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir)
    return tokenizer
def tokenize_caption(tokenizer, raw_caption, cap_id, type='ZH'):

    if(type == 'EN'):
        word_list = clean_str(raw_caption)
        txt_caption = " ".join(word_list)
        # Remove whitespace at beginning and end of the sentence.
        txt_caption = txt_caption.strip()
        # Add period at the end of the sentence if not already there.
        try:
            if txt_caption[-1] not in [".", "?", "!"]:
                txt_caption += "."
        except:
            # print(cap_id)
            pass
        txt_caption = txt_caption.capitalize()

        ids = tokenizer.encode(txt_caption, add_special_tokens=True)

    else:
        ids = tokenizer.encode(raw_caption, add_special_tokens=True)

    return ids

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r',encoding='utf-8',errors='ignore') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos,videos_mask, bert_en_cap,bert_zh_cap, bert_en_mask_cap,bert_zh_mask_cap, idxs, cap_ids_en, video_ids = zip(*data)
  
    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths_mask = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos_mask]
    frame_vec_len_mask = len(videos_mask[0][0])
    vidoes_vmask = torch.zeros(len(videos_mask), max(video_lengths_mask), frame_vec_len_mask)
    videos_origin_mask = torch.zeros(len(videos_mask), frame_vec_len_mask)
    vidoes_mask_mask = torch.zeros(len(videos_mask), max(video_lengths_mask))
    for i, frames in enumerate(videos_mask):
            end = video_lengths_mask[i]
            vidoes_vmask[i, :end, :] = frames[:end,:]
            videos_origin_mask[i,:] = torch.mean(frames,0)
            vidoes_mask_mask[i,:end] = 1.0
    
    # BERT EN
    if bert_en_cap[0] is not None:
        lengths_en = [len(cap) for cap in bert_en_cap]
        bert_target_en = torch.zeros(len(bert_en_cap), max(lengths_en)).long()
        words_mask_en = torch.zeros(len(bert_en_cap), max(lengths_en))
        for i, cap in enumerate(bert_en_cap):
            end = lengths_en[i]
            bert_target_en[i, :end] = cap[:end]
            words_mask_en[i, :end] = 1.0
    else:
        bert_target_en = None
        words_mask_en = None
    # BERT EN MASK
    if bert_en_mask_cap[0] is not None:
        lengths_en_mask = [len(cap) for cap in bert_en_mask_cap]
        bert_target_en_mask = torch.zeros(len(bert_en_mask_cap), max(lengths_en_mask)).long()
        words_mask_en_mask = torch.zeros(len(bert_en_mask_cap), max(lengths_en_mask))
        for i, cap in enumerate(bert_en_mask_cap):
            end = lengths_en_mask[i]
            bert_target_en_mask[i, :end] = cap[:end]
            words_mask_en_mask[i, :end] = 1.0
    else:
        bert_target_en_mask = None
        words_mask_en_mask = None
    # BERT ZH
    if bert_zh_cap[0] is not None:
        lengths_zh = [len(cap) for cap in bert_zh_cap]
        bert_target_zh = torch.zeros(len(bert_zh_cap), max(lengths_zh)).long()
        words_mask_zh = torch.zeros(len(bert_zh_cap), max(lengths_zh))
        for i, cap in enumerate(bert_zh_cap):
            end = lengths_zh[i]
            bert_target_zh[i, :end] = cap[:end]
            words_mask_zh[i, :end] = 1.0
    else:
        bert_target_zh = None
        words_mask_zh = None
    # BERT ZH MASK
    if bert_zh_mask_cap[0] is not None:
        lengths_zh_mask = [len(cap) for cap in bert_zh_mask_cap]
        bert_target_zh_mask = torch.zeros(len(bert_zh_mask_cap), max(lengths_zh_mask)).long()
        words_mask_zh_mask = torch.zeros(len(bert_zh_mask_cap), max(lengths_zh_mask))
        for i, cap in enumerate(bert_zh_mask_cap):
            end = lengths_zh_mask[i]
            bert_target_zh_mask[i, :end] = cap[:end]
            words_mask_zh_mask[i, :end] = 1.0
    else:
        bert_target_zh_mask = None
        words_mask_zh_mask = None




    video_data = ((vidoes, videos_origin, video_lengths, vidoes_mask),(vidoes_vmask, videos_origin_mask, video_lengths_mask, vidoes_mask_mask))
    text_data = ((bert_target_en, words_mask_en),(bert_target_zh, words_mask_zh),(bert_target_en_mask, words_mask_en_mask),(bert_target_zh_mask, words_mask_zh_mask))

    return video_data, text_data,  idxs, cap_ids_en, video_ids

def collate_frame_gru_fn_single(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos,videos_mask, bert_en_cap, bert_en_mask_cap, idxs, cap_ids_en, video_ids = zip(*data)
  
    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths_mask = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos_mask]
    frame_vec_len_mask = len(videos_mask[0][0])
    vidoes_vmask = torch.zeros(len(videos_mask), max(video_lengths_mask), frame_vec_len_mask)
    videos_origin_mask = torch.zeros(len(videos_mask), frame_vec_len_mask)
    vidoes_mask_mask = torch.zeros(len(videos_mask), max(video_lengths_mask))
    for i, frames in enumerate(videos_mask):
            end = video_lengths_mask[i]
            vidoes_vmask[i, :end, :] = frames[:end,:]
            videos_origin_mask[i,:] = torch.mean(frames,0)
            vidoes_mask_mask[i,:end] = 1.0
    
    # BERT EN
    if bert_en_cap[0] is not None:
        lengths_en = [len(cap) for cap in bert_en_cap]
        bert_target_en = torch.zeros(len(bert_en_cap), max(lengths_en)).long()
        words_mask_en = torch.zeros(len(bert_en_cap), max(lengths_en))
        for i, cap in enumerate(bert_en_cap):
            end = lengths_en[i]
            bert_target_en[i, :end] = cap[:end]
            words_mask_en[i, :end] = 1.0
    else:
        bert_target_en = None
        words_mask_en = None
    # BERT EN MASK
    if bert_en_mask_cap[0] is not None:
        lengths_en_mask = [len(cap) for cap in bert_en_mask_cap]
        bert_target_en_mask = torch.zeros(len(bert_en_mask_cap), max(lengths_en_mask)).long()
        words_mask_en_mask = torch.zeros(len(bert_en_mask_cap), max(lengths_en_mask))
        for i, cap in enumerate(bert_en_mask_cap):
            end = lengths_en_mask[i]
            bert_target_en_mask[i, :end] = cap[:end]
            words_mask_en_mask[i, :end] = 1.0
    else:
        bert_target_en_mask = None
        words_mask_en_mask = None




    video_data = ((vidoes, videos_origin, video_lengths, vidoes_mask),(vidoes_vmask, videos_origin_mask, video_lengths_mask, vidoes_mask_mask))
    text_data = ((bert_target_en, words_mask_en),(bert_target_en_mask, words_mask_en_mask))

    return video_data, text_data,  idxs, cap_ids_en, video_ids


def collate_frame(data):

    videos,videos_mask, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths_mask = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos_mask]
    frame_vec_len_mask = len(videos_mask[0][0])
    vidoes_vmask = torch.zeros(len(videos_mask), max(video_lengths_mask), frame_vec_len_mask)
    videos_origin_mask = torch.zeros(len(videos_mask), frame_vec_len_mask)
    vidoes_mask_mask = torch.zeros(len(videos_mask), max(video_lengths_mask))
    for i, frames in enumerate(videos_mask):
            end = video_lengths_mask[i]
            vidoes_vmask[i, :end, :] = frames[:end,:]
            videos_origin_mask[i,:] = torch.mean(frames,0)
            vidoes_mask_mask[i,:end] = 1.0

    video_data = ((vidoes, videos_origin, video_lengths, vidoes_mask),(vidoes_vmask, videos_origin_mask, video_lengths_mask, vidoes_mask_mask))

    return video_data, idxs, video_ids


def collate_text(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    # captions, cap_bows, idxs, cap_ids = zip(*data)
    bert_cap_en,bert_cap_zh,bert_cap_en_mask,bert_cap_zh_mask, idxs, cap_ids = zip(*data)
    
    # EN
    if bert_cap_en[0] is not None:
        lengths_en = [len(cap) for cap in bert_cap_en]
        bert_target_en = torch.zeros(len(bert_cap_en), max(lengths_en)).long()
        words_mask_en = torch.zeros(len(bert_cap_en), max(lengths_en))
        for i, cap in enumerate(bert_cap_en):
            end = lengths_en[i]
            bert_target_en[i, :end] = cap[:end]
            words_mask_en[i, :end] = 1.0
    else:
        bert_target_en = None
        words_mask_en = None

    # EN MASK
    if bert_cap_en_mask[0] is not None:
        lengths_en_mask = [len(cap) for cap in bert_cap_en_mask]
        bert_target_en_mask = torch.zeros(len(bert_cap_en_mask), max(lengths_en_mask)).long()
        words_mask_en_mask = torch.zeros(len(bert_cap_en_mask), max(lengths_en_mask))
        for i, cap in enumerate(bert_cap_en_mask):
            end = lengths_en_mask[i]
            bert_target_en_mask[i, :end] = cap[:end]
            words_mask_en_mask[i, :end] = 1.0
    else:
        bert_target_en_mask = None
        words_mask_en_mask = None

    # ZH
    if bert_cap_zh[0] is not None:
        lengths_zh = [len(cap) for cap in bert_cap_zh]
        bert_target_zh = torch.zeros(len(bert_cap_zh), max(lengths_zh)).long()
        words_mask_zh = torch.zeros(len(bert_cap_zh), max(lengths_zh))
        for i, cap in enumerate(bert_cap_zh):
            end = lengths_zh[i]
            bert_target_zh[i, :end] = cap[:end]
            words_mask_zh[i, :end] = 1.0
    else:
        bert_target_zh = None
        words_mask_zh = None
    # ZH MASK
    if bert_cap_zh_mask[0] is not None:
        lengths_zh_mask = [len(cap) for cap in bert_cap_zh_mask]
        bert_target_zh_mask = torch.zeros(len(bert_cap_zh_mask), max(lengths_zh_mask)).long()
        words_mask_zh_mask = torch.zeros(len(bert_cap_zh_mask), max(lengths_zh_mask))
        for i, cap in enumerate(bert_cap_zh_mask):
            end = lengths_zh_mask[i]
            bert_target_zh_mask[i, :end] = cap[:end]
            words_mask_zh_mask[i, :end] = 1.0
    else:
        bert_target_zh_mask = None
        words_mask_zh_mask = None
    text_data =  ((bert_target_en, words_mask_en),(bert_target_zh, words_mask_zh),(bert_target_en_mask,  words_mask_en_mask),(bert_target_zh_mask,  words_mask_zh_mask))

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, opt, cap_en_file,cap_zh_file, visual_feat,video2frames=None):
        # Captions EN
        self.captions_en = {}
        self.cap_ids_en = []
        self.video_ids = set()
        self.video2frames = video2frames
        with open(cap_en_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions_en[cap_id] = caption
                self.cap_ids_en.append(cap_id)
                self.video_ids.add(video_id)
        self.visual_feat = visual_feat
        self.length = len(self.cap_ids_en)
        self.opt = opt
        if self.opt.train_mode=="parallel":
        # Captions ZH
            self.captions_zh = {}
            self.cap_ids_zh = []
            with open(cap_zh_file, 'r',encoding='utf-8') as trans_reader:
                for line in trans_reader.readlines():
                    cap_id_zh, caption_zh = line.strip().split(' ', 1)
                    self.captions_zh[cap_id_zh] = caption_zh
                    self.cap_ids_zh.append(cap_id_zh)
        elif self.opt.train_mode=="unparallel":
            pass

        self.tokenizer=create_tokenizer()


    def __getitem__(self, index):
        cap_en_id = self.cap_ids_en[index]
        str_ls = cap_en_id.split('#')
        assert self.opt.label_situation in ['human_label','translate']
        if self.opt.label_situation == "human_label":
            cap_zh_id = str_ls[0] + '#zh#' + str_ls[2]
        elif self.opt.label_situation == "translate":
            cap_zh_id = str_ls[0] + '#enc2zh#' + str_ls[2]

        video_id = getVideoId(cap_en_id)

        # video
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)


        # video_mask
        L = [x for x in range(len(frame_list))]
        random=int(len(frame_list)*0.8)
        matrix=np.random.choice(L,random,replace=False)
        matrix.sort()
        frame_list_mask=copy.deepcopy(frame_list)
        counter=0
        for j in range(len(matrix)):
            matrix[j]=matrix[j]-counter
            frame_list_mask.pop(matrix[j])
            counter+=1
        frame_vecs_mask = []
        for frame_id in frame_list_mask:
            frame_vecs_mask.append(self.visual_feat.read_one(frame_id))
        frames_mask_tensor = torch.Tensor(frame_vecs_mask)
        # text en
        caption_en = self.captions_en[cap_en_id]
        if self.opt.train_mode=="unparallel":
            bert_ids_en = tokenize_caption(self.tokenizer, caption_en, cap_en_id, type=str_ls[1][:2].upper())
        elif self.opt.train_mode=="parallel":
            bert_ids_en = tokenize_caption(self.tokenizer, caption_en, cap_en_id, type='EN')
        L = [x for x in range(1,len(bert_ids_en)-1)]
        random=int(len(bert_ids_en)*0.15)
        matrix=np.random.choice(L,random,replace=False)
        bert_ids_en_mask=copy.deepcopy(bert_ids_en)
        for j in range(len(matrix)):
            bert_ids_en_mask[matrix[j]]=103
        bert_tensor_en = torch.Tensor(bert_ids_en)
        bert_tensor_en_mask = torch.Tensor(bert_ids_en_mask)
        if self.opt.train_mode=="parallel":
        # text zh
            caption_zh = self.captions_zh[cap_zh_id]
            bert_ids_zh = tokenize_caption(self.tokenizer, caption_zh, cap_zh_id, type='ZH')
            L = [x for x in range(1,len(bert_ids_zh)-1)]
            random=int(len(bert_ids_zh)*0.15)
            matrix=np.random.choice(L,random,replace=False)
            bert_ids_zh_mask=copy.deepcopy(bert_ids_zh)
            for j in range(len(matrix)):
                bert_ids_zh_mask[matrix[j]]=103
            bert_tensor_zh = torch.Tensor(bert_ids_zh)
            bert_tensor_zh_mask = torch.Tensor(bert_ids_zh_mask)
            return frames_tensor,frames_mask_tensor, bert_tensor_en,bert_tensor_zh,bert_tensor_en_mask,bert_tensor_zh_mask,index, cap_en_id, video_id
        elif self.opt.train_mode=="unparallel":
            return frames_tensor,frames_mask_tensor, bert_tensor_en,bert_tensor_en_mask,index, cap_en_id, video_id

    def __len__(self):
        return self.length
     

class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, video2frames=None, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        # video_mask
        L = [x for x in range(len(frame_list))]
        random=int(len(frame_list)*0.8)
        matrix=np.random.choice(L,random,replace=False)
        matrix.sort()
        frame_list_mask=copy.deepcopy(frame_list)
        counter=0
        for j in range(len(matrix)):
            matrix[j]=matrix[j]-counter
            frame_list_mask.pop(matrix[j])
            counter+=1
        frame_vecs_mask = []
        for frame_id in frame_list_mask:
            frame_vecs_mask.append(self.visual_feat.read_one(frame_id))
        frames_mask_tensor = torch.Tensor(frame_vecs_mask)

        return frames_tensor,frames_mask_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """
    def __init__(self,opt, cap_en_file,cap_zh_file, lang_type):
        # Captions
        self.captions_en = {}
        self.cap_ids_en = []
        with open(cap_en_file, 'r',encoding='utf-8',errors='ignore') as cap_reader:
            for line in cap_reader.readlines():
                cap_id_en, caption_en = line.strip().split(' ', 1)
                self.captions_en[cap_id_en] = caption_en
                self.cap_ids_en.append(cap_id_en)
        self.length = len(self.cap_ids_en)
        # BERT
        self.tokenizer = create_tokenizer()
        self.opt=opt
        # trans
        self.captions_zh = {}
        self.cap_ids_zh = []
        with open(cap_zh_file, 'r',encoding='utf-8') as trans_reader:
            for line in trans_reader.readlines():
                cap_id_zh, caption_zh = line.strip().split(' ', 1)
                self.captions_zh[cap_id_zh] = caption_zh
                self.cap_ids_zh.append(cap_id_zh)

        self.type = lang_type

    def __getitem__(self, index):
        cap_en_id = self.cap_ids_en[index]
        str_ls = cap_en_id.split('#')
        if self.opt.train_test=='test':
            assert self.opt.target_language in ['en','zh']
            if self.opt.target_language == "en":
                cap_zh_id = str_ls[0] + '#enc2zh#' + str_ls[2]
            elif self.opt.target_language == "zh":
                cap_zh_id = str_ls[0] + '#zh#' + str_ls[2]
        else:
            cap_zh_id = str_ls[0] + '#zh#' + str_ls[2]

        # text zh
        caption_zh = self.captions_zh[cap_zh_id]
        bert_ids_zh = tokenize_caption(self.tokenizer, caption_zh, cap_zh_id, type='ZH')
        L = [x for x in range(1,len(bert_ids_zh)-1)]
        random=int(len(bert_ids_zh)*0.15)
        matrix=np.random.choice(L,random,replace=False)
        bert_ids_zh_mask=copy.deepcopy(bert_ids_zh)
        for j in range(len(matrix)):
            bert_ids_zh_mask[matrix[j]]=103
        bert_tensor_zh = torch.Tensor(bert_ids_zh)
        bert_tensor_zh_mask = torch.Tensor(bert_ids_zh_mask)
        # text en
        caption_en = self.captions_en[cap_en_id]
        bert_ids_en = tokenize_caption(self.tokenizer, caption_en, cap_en_id, type='EN')
        L = [x for x in range(1,len(bert_ids_en)-1)]
        random=int(len(bert_ids_en)*0.15)
        matrix=np.random.choice(L,random,replace=False)
        bert_ids_en_mask=copy.deepcopy(bert_ids_en)
        for j in range(len(matrix)):
            bert_ids_en_mask[matrix[j]]=103
        bert_tensor_en = torch.Tensor(bert_ids_en)
        bert_tensor_en_mask = torch.Tensor(bert_ids_en_mask)
       
        return bert_tensor_en,bert_tensor_zh,bert_tensor_en_mask,bert_tensor_zh_mask,index, cap_en_id

    def __len__(self):
        return self.length

def get_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, vocab, bow2vec, batch_size=100, num_workers=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], tag_path, tag_vocab_path, bow2vec, vocab, video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], None, tag_vocab_path, bow2vec, vocab, video2frames=video2frames['val']) }
    
    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_train_data_loaders(opt, cap_en_files,cap_zh_files, visual_feats, batch_size=100, num_workers=2,mode="multi", video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(opt,cap_en_files['train'],cap_zh_files['train'], visual_feats['train'],  video2frames=video2frames['train'])}
    if mode=="multi":
        data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                        batch_size=batch_size,
                                        shuffle=(x=='train'),
                                        pin_memory=True,
                                        num_workers=num_workers,
                                        collate_fn=collate_frame_gru_fn)
                            for x in cap_en_files  if x=='train' }
    elif mode=="single":
        data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                batch_size=batch_size,
                                shuffle=(x=='train'),
                                pin_memory=True,
                                num_workers=num_workers,
                                collate_fn=collate_frame_gru_fn_single)
                    for x in cap_en_files  if x=='train' }
    return data_loaders



def get_test_data_loaders(cap_files, visual_feats, batch_size=100, num_workers=2, video2frames = None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], video2frames = video2frames['test'])}


    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None, video_ids=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames, video_ids=video_ids)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader


def get_txt_data_loader(opt,cap_en_file,cap_zh_file, batch_size=100, num_workers=2,lang_type=None):
    dset = TxtDataSet4DualEncoding(opt,cap_en_file,cap_zh_file, lang_type=None)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader



if __name__ == '__main__':
    pass
