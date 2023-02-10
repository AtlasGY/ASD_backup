import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import numpy as np
from dataset_preprocess import *
from models.STgram import STgramMFN, ArcMarginProduct
from models.STgram_trainer import *
# 关闭warning
import warnings
from adaloss import AdaCos
warnings.filterwarnings("ignore")


def load_config():
    config_path = './configs/config.yaml'
    with open(config_path) as f:
        param = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description='STgram-MFN')

    # from config.yaml to parser class
    parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model saved dir')
    parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result saved dir')
    parser.add_argument('--data-dir', default=param['data_dir'], type=str, help='data dir')
    parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='processing data saved dir')
    parser.add_argument('--process-machines', default=param['process_machines'], type=list,
                        help='allowed processing machines')
    parser.add_argument('--ID-factor', default=param['ID_factor'], help='times factor for different machine types and ids to label')
    # extract feature
    parser.add_argument('--sr', default=param['sr'], type=int, help='sample rate of wav files')
    parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT param: n_fft')
    parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT param: n_mels')
    parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT param: hop_length')
    parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT param: win_length')
    parser.add_argument('--power', default=param['power'], type=float, help='STFT param: power')
    parser.add_argument('--frames', default=param['frames'], type=int, help='split frames')
    parser.add_argument('--skip-frames', default=param['skip_frames'], type=int, help='skip frames in spliting')
    # train
    parser.add_argument('--num-class', default=param['num_class'], help='number of labels')
    parser.add_argument('--epochs', default=param['epochs'], type=int)
    parser.add_argument('--workers', default=param['workers'], type=int, help='number of workers for dataloader')
    parser.add_argument('--batch-size', default=param['batch_size'], type=int)
    parser.add_argument('--lr', '--learning-rate', default=param['lr'], type=float)
    parser.add_argument('--log-every-n-steps', default=20, type=int)
    parser.add_argument('--save-every-n-epochs', default=param['save_every_n_epochs'], type=int, help='save encoder and decoder model every n epochs')
    parser.add_argument('--early-stop', default=param['early_stop'], type=int, help='number of epochs for early stopping')
    parser.add_argument('--device-ids', default=param['device_ids'])
    # arcface
    parser.add_argument('--losstype', default=param['losstype'], type=str, help='using arcface or adacos')#
    # parser.add_argument('--arcface', default=True, type=bool, help='using arcface or not')
    parser.add_argument('--m', type=float, default=param['margin'], help='margin for arcface')
    parser.add_argument('--s', type=float, default=param['scale'], help='scale for arcface')

    parser.add_argument('--version', default=param['version'], type=str,
                        help='trail version')

    args = parser.parse_args()
    return args

def preprocess(args):
    # root_folder = os.path.join(args.pre_data_dir, f'train_path_list.db')
    root_folder = args.pre_data_dir + '/train_path_list.txt'
    print(root_folder)
    if not os.path.exists(root_folder):
        print('Train_Path_File file exist?')
        print(os.path.exists(root_folder))
        utils.path_to_dict(process_machines=args.process_machines,
                           data_dir=args.data_dir,
                           root_folder=root_folder,
                           ID_factor=args.ID_factor)

    # root_folder = args.pre_data_dir + '/test_path_list.txt'
    # print(root_folder)
    # if not os.path.exists(root_folder):
    #     print('Test_Path_File file exist?')
    #     print(os.path.exists(root_folder))
    #     utils.path_to_dict(process_machines=args.process_machines,
    #                        data_dir=args.data_dir,
    #                        root_folder=root_folder,
    #                        ID_factor=args.ID_factor,
    #                        dir_name='test')


def train(args):
    """
    1. 准备训练数据
    """
    # 从预先处理好的test_path_list.TXT 和 train_path_list.TXT 文件中读取 训练数据和测试数据文档
    f = open("./data/pre_data/train_path_list.TXT")  # 返回一个文件对象
    train_path_dict = {}#分设备类别来记录样本的存储空间
    train_path_all_files = []# 将所有的样本地址都存储起来
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        # print(line, end='')  # 在 Python 3中使用
        line = line.replace('[','')
        line = line.replace(']', '')
        aa = line.split(': ')
        train_path_dict[aa[0]] = aa[1]
        line = f.readline()
    f.close()


    # 训练数据的每个音频文件对应的设备ID
    ID_factor_num = []
    ID_factor_train_y = np.zeros(0)
    train_file_dict = {}
    for key in train_path_dict.keys():
        id = args.ID_factor[key]
        id = int(id)
        # id = 2
        files = train_path_dict[key].split(',')
        train_path_all_files = train_path_all_files + files
        train_file_dict[key] = files
        ID_factor_num.append(len(files))
        ID_factor_train_y = np.hstack([ID_factor_train_y, np.ones(len(files))*id])
    assert ID_factor_train_y.shape[0] == np.array(ID_factor_num).sum()
    print(ID_factor_train_y)
    train_file_dict['ALL'] = train_path_all_files #
    #train_file_dict[machine type_sec_number] -->里面存储着一个list，这个list里包含了比如slider_sec_00 中所有训练数据的文件路径
    ##得到 train_file_dict 和 ID_factor_train_y
    # train_file_dict['ALL']中存储了所有设备的所有训练数据的位置

    #构造数据集
    clf_dataset = WavMelClassifierDataset(train_file_dict, args.sr, 'ALL', args.ID_factor)
    train_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
    #这里不用动，标准语句
    train_clf_loader = torch.utils.data.DataLoader(
        train_clf_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    print(train_clf_loader)
    print("1. WavMelClassifierDataset Have Initalied!")


    # # 构建test dataset
    # clf_dataset = WavMelClassifierDataset(train_file_dict, args.sr, 'ALL', args.ID_factor)
    # test_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
    #                                             n_fft=args.n_fft,
    #                                             hop_length=args.hop_length,
    #                                             win_length=args.win_length,
    #                                             power=args.power)
    # train_clf_loader = torch.utils.data.DataLoader(
    #     train_clf_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    # print(train_clf_loader)



    # 2. 准备模型
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


    print("2. Model Has Been Initalied")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_clf_loader), eta_min=0,
                                                           last_epoch=-1)


    # 3. 开始训练
    print("3. Training model...")
    with torch.cuda.device(args.device_ids[0]):
        args.dp = False
        if len(args.device_ids) > 1:
            args.dp = True
            model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=args.ID_factor,
                                       classifier=model,
                                       arcface=cls_head,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       args=args)
        trainer.train(train_clf_loader)
    print("Training Finished!!!")

if __name__ == "__main__":
    args = load_config()
    # 将训练数据的路径全部保存至 data/pre_data/train_path_list.txt 中。
    preprocess(args)


    train(args)