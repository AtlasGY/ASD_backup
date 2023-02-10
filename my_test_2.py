import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import numpy as np
from dataset_preprocess import *
from models.STgram import STgramMFN, ArcMarginProduct
from models.STgram_trainer import *
# 关闭warning
from adaloss import AdaCos
import warnings
warnings.filterwarnings("ignore")

from my_train_1 import load_config,preprocess

def my_eval(args):
    if args.losstype == 'arcface':
        cls_head = ArcMarginProduct(128, args.num_class, m=args.m, s=args.s)
    elif args.losstype == 'adacos':
        cls_head = AdaCos(128, args.num_class)
    else:
        cls_head = None
    model = STgramMFN(num_class=args.num_class,
                      c_dim=args.n_mels,
                      win_len=args.win_length,
                      hop_len=args.hop_length,
                      arcface=cls_head)
    if torch.cuda.is_available() and len(args.device_ids) > 0:
        args.device = torch.device(f'cuda:{args.device_ids[0]}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    with torch.cuda.device(args.device_ids[0]):
        model_path = os.path.join(args.model_dir, args.version, f'checkpoint_best.pth.tar')
        print("Loaded Model Path:")
        print(model_path)
        model.load_state_dict(torch.load(model_path)['clf_state_dict'])
        args.dp = False
        if len(args.device_ids) > 1:
            args.dp = True
            model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=args.ID_factor,
                                       classifier=model,
                                       arcface=cls_head,
                                       optimizer=None,
                                       scheduler=None,
                                       args=args)
        # trainer.abnormal_eval()
        #


        '''1用来构建用来测试和训练异常检测算法的模型的数据'''
        trainer.ASD_train_test_data_generation(args)


import pyod
"""异常检测部分"""
def post_anormal_detection():
    #加载train 和 test data_dict
    #3layer dict
    # data_dict-->machine type -->predict_logits normal_label
    f_read = open('./data/asd_train.pkl', 'rb')
    train_data_dict = pickle.load(f_read)
    f_read.close()

    f_read = open('./data/asd_test.pkl', 'rb')
    test_data_dict = pickle.load(f_read)
    f_read.close()



    return 0


if __name__ == "__main__":
    args = load_config()
    # 将训练数据的路径全部保存至 data/pre_data/test_path_list.txt 中。
    preprocess(args)

    '''1. 用来构建用来测试和训练异常检测算法的模型的数据'''
    my_eval(args)

    """2. 异常检测部分"""
    # post_anormal_detection()
