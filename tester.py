import os
import sys
import json
import torch
import pickle
import logging
import argparse

import evaluation
from model import get_model
from validate import norm_score, cal_perf

import util.tag_data_provider as data
from util.text2vec import get_text_encoder
import util.metrics as metrics
from util.vocab import Vocabulary

from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--split', default='test', type=str, help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=16, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    # mode
    parser.add_argument('--train_mode', type=str, default='parallel',help='training data properties (unparallel|parallel)')
    parser.add_argument('--label_situation', type=str, default='translate',help='training data properties (translate|human_label)')
    parser.add_argument('--target_language', type=str, default='zh',help='language to search (en|zh)')
    parser.add_argument('--train_test', type=str, default='test',help='in trainning or testing')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume,map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
        .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    
    # collection setting


    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname['test'] = opt.testCollection
    #trainCollection = options.trainCollection
    output_dir = resume.replace(opt.testCollection+"train", opt.testCollection)
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/%s/' % (options.cv_name, opt.testCollection))
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    log_config(output_dir)
    logging.info(json.dumps(vars(opt), indent=2))

    assert opt.target_language in ['en','zh']
    
    if opt.target_language=="en":
        test_cap_en = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s.caption.txt'%(testCollection, opt.split))
        test_cap_zh = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_google_enc2zh.caption.txt'%(testCollection, opt.split))
    elif opt.target_language=="zh":
        test_cap_en = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_google_zhc2en.caption.txt'%(testCollection, opt.split))
        test_cap_zh = os.path.join(rootpath, collections_pathname['test'], 'TextData', '%s%s_zh.caption.txt'%(testCollection, opt.split))
    

    caption_files_en = {'test': test_cap_en}
    caption_files_zh = {'test': test_cap_zh}
    img_feat_path = os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(rootpath, collections_pathname['test'], 'FeatureData', options.visual_feature, 'video2frames.txt'))}

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    # set data loader
    video_ids_list = data.read_video_ids(caption_files_en['test'])
    vid_data_loader = data.get_vis_data_loader(visual_feats['test'], opt.batch_size, opt.workers, video2frames['test'], video_ids=video_ids_list)
    text_data_loader = data.get_txt_data_loader(opt,caption_files_en['test'],caption_files_zh['test'], opt.batch_size, opt.workers)

    # mapping
    # if options.space == 'hybrid':
    #     video_embs, video_tag_probs, video_ids = evaluation.encode_text_or_vid_tag_hist_prob(model.embed_vis, vid_data_loader)
    #     cap_embs, cap_tag_probs, caption_ids = evaluation.encode_text_or_vid_tag_hist_prob(model.embed_txt, text_data_loader)
    # else:
    video_embs, video_ids = evaluation.encode_vid(model.embed_vis, vid_data_loader)
    cap_embs_en,cap_embs_zh, caption_ids = evaluation.encode_text(model.embed_txt, text_data_loader)


    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    logging.info("write into: %s" % output_dir)
    # if options.space != 'latent':
    #     tag_vocab_path = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'tags', 'video_label_th_1', 'tag_vocab_%d.json' % options.tag_vocab_size)
    #     evaluation.pred_tag(video_tag_probs, video_ids, tag_vocab_path, os.path.join(output_dir, 'video'))
    #     evaluation.pred_tag(cap_tag_probs, caption_ids, tag_vocab_path, os.path.join(output_dir, 'text'))
    if opt.train_mode=="parallel":
        if opt.label_situation == "human_label":
            weight=0.85
            t2v_all_errors_1 = evaluation.cal_error_test(opt,video_embs, cap_embs_en,cap_embs_zh,weight, options.measure)
        elif opt.label_situation == "translate":
            weight=0.55
            t2v_all_errors_1 = evaluation.cal_error_test(opt,video_embs, cap_embs_en,cap_embs_zh,weight, options.measure)
    elif opt.train_mode=="unparallel":
        weight=0.55
        t2v_all_errors_1 = evaluation.cal_error_test(opt,video_embs, cap_embs_en,cap_embs_zh,weight, options.measure)


    # if options.space in ['concept', 'hybrid']:
    #     # logging.info("=======Concept Space=======")
    #     t2v_all_errors_2 = evaluation.cal_error_batch(video_tag_probs, cap_tag_probs, options.measure_2)
    
    # if options.space in ['hybrid']:
    #     w = 0.6
    #     t2v_all_errors_1 = norm_score(t2v_all_errors_1_en)
    #     t2v_all_errors_2 = norm_score(t2v_all_errors_2_en)
    #     t2v_tag_all_errors = w*t2v_all_errors_1 + (1-w)*t2v_all_errors_2
    #     cal_perf(t2v_tag_all_errors, v2t_gt, t2v_gt)
    #     torch.save({'errors': t2v_tag_all_errors, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)
    #     logging.info("write into: %s" % pred_error_matrix_file)
    if opt.target_language == "en":
        t2v,v2t=cal_perf(t2v_all_errors_1, v2t_gt, t2v_gt,"EN")
    elif opt.target_language == "zh":
        t2v,v2t=cal_perf(t2v_all_errors_1, v2t_gt, t2v_gt,"ZH")
    t1,t2,t3,_,_,_=t2v
    v1,v2,v3,_,_,_=v2t
    logging.info(" * Recall sum :{}".format(round(v1+v2+v3+t1+t2+t3, 1)))
    torch.save({'errors': t2v_all_errors_1, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)
    logging.info("write into: %s" % pred_error_matrix_file)



if __name__ == '__main__':
    main()
