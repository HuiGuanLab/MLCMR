import os
import json
import torch
import numpy as np
from loss import jaccard_sim
from scipy.spatial import distance
from basic.generic_utils import Progbar
from basic.common import makedirsforfile

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

def norm_score(t2v_all_errors):
    t2v_all_score = -t2v_all_errors
    t2v_all_score = t2v_all_score - np.min(t2v_all_score)
    t2v_all_score = t2v_all_score / np.max(t2v_all_score)
    return -t2v_all_score

def cal_error(videos, captions_en,captions_zh, measure='cosine'):
    if measure == 'cosine':
        captions_en = l2norm(captions_en)
        captions_zh = l2norm(captions_zh)
        videos = l2norm(videos)
        errors_en = -1*np.dot(captions_en, videos.T)
        errors_zh = -1*np.dot(captions_zh, videos.T)
    return errors_en,errors_zh

def cal_error_test(opt, videos, captions_en,captions_zh,weight,measure='cosine'):
    if measure == 'cosine':
        captions_en = l2norm(captions_en)
        captions_zh = l2norm(captions_zh)
        videos = l2norm(videos)
        errors_en = -1*np.dot(captions_en, videos.T)
        errors_zh = -1*np.dot(captions_zh, videos.T)
    if opt.target_language=="zh":
        errors = (1-weight)*errors_en+weight*errors_zh
    elif opt.target_language=="en":
        errors = weight*errors_en+(1-weight)*errors_zh
    
    return errors

# ithe number of videos or captions are too large, the memory may be not enough for jaccard similarity computation.
# Hence, we split the sentence embedding matrix into a sequence of matrices with smaller size
def cal_error_batch(videos, captions, measure='cosine', batch_size=2000):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*np.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1':
        errors = distance.cdist(captions, videos, 'minkowski', p=1)
    elif measure == 'l2':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1_norm':
        errors = - distance.cdist(captions, videos, 'minkowski', p=1)/videos.shape[1]-1
    elif measure == 'l2_norm':
        errors = - distance.cdist(captions, videos, 'euclidean')/videos.shape[1]-1
    elif measure == 'jaccard':
        idx = 0
        errors = None
        while 1:
            # print(idx)
            sub_captions = captions[idx*batch_size:(idx+1)*batch_size,:]
            sub_captions = torch.Tensor(sub_captions)
            videos = torch.Tensor(videos)
            sub_errors = -1*jaccard_sim(sub_captions, videos)
            if errors is None:
                errors = sub_errors.numpy()
            else:
                errors = np.append(errors, sub_errors, axis=0)
            if (idx+1)*batch_size > captions.shape[0]:
                break
            idx=idx+1
    return errors


def cal_simi(captions, videos, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = np.dot(captions, videos.T)
    elif measure == 'jaccard':
        captions = torch.Tensor(captions)
        videos = torch.Tensor(videos)
        errors = jaccard_sim(captions, videos)
    return errors



# encode text or video
def encode_vid(encoder, data_loader, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embeddings = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        emb = encoder(datas)

        # initialize the numpy arrays given the size of the embeddings
        if embeddings is None:
            embeddings = np.zeros((len(data_loader.dataset), emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        embeddings[idxs] = emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if return_ids == True:
        return embeddings, ids,
    else:
        return embeddings

def encode_text(encoder, data_loader, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embeddings_en = None
    embeddings_zh = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):
        # compute the embeddings
        emb_en,emb_zh = encoder(datas)
        # iniwetialize the numpy arrays given the size of the embeddings
        if embeddings_en is None:
            embeddings_en = np.zeros((len(data_loader.dataset), emb_en.size(1)))
        if embeddings_zh is None:
            embeddings_zh = np.zeros((len(data_loader.dataset), emb_zh.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        embeddings_en[idxs] = emb_en.data.cpu().numpy().copy()
        embeddings_zh[idxs] = emb_zh.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if return_ids == True:
        return embeddings_en,embeddings_zh, ids
    else:
        return embeddings_en,embeddings_zh

