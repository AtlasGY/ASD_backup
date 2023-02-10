import torch
from torch.utils.data import Dataset
import torchaudio
import joblib
import librosa
import re
import speechpy
import numpy as np

class Generator(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


class Wav_Mel_ID_Dataset(Dataset):
    def __init__(self, train_file_dict, machine_type, machine_label_dict, sr,
                 win_length, hop_length, transform=None):
        self.file_path_list = train_file_dict[machine_type]
        self.transform = transform
        self.machine_type = machine_type
        self.machine_label_dict = machine_label_dict
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))
        print("Load machine type: "+machine_type)
        print("Number of samples: {}" .format(len(self.file_path_list)))

    def __getitem__(self, item):
        file_path = self.file_path_list[item]
        # 对file path 做预处理，把一些异常的符号去除掉
        file_path = file_path.replace('\'', '') # 去掉file path 开始和结束的引号
        file_path = file_path.replace(' ', '')
        file_path = file_path.replace('\n', '')
        # print(file_path)
        # print(file_path.split('\\'))
        splited_file_path = file_path.split('\\')
        while "" in splited_file_path:
            splited_file_path.remove("")
        # print(splited_file_path)
        machine = splited_file_path[-3]
        sec_num = splited_file_path[-1].split('_')[1]
        id = self.machine_label_dict[machine+'_sec_'+sec_num]
        label = int(id)

        (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

        x = x[:self.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)

        # #### STAERT: mel without cmvn
        # x_mel = self.transform(x_wav).unsqueeze(0)
        # print('x_mel')
        # print(x_mel.shape)
        # #### END

        #### STAERT: mel with cmvn
        ##抽取 MFCC特征
        x_mel = self.transform(x_wav)
        # print(x_mel.shape)
        x_mel = x_mel.T
        x_mel = np.array(x_mel)

        #使用CMVN对数据进行mean/STD的操作。本质上，如果后续的DL模型中有batch norm的操作，这里的这个标准化操作是等价于batch norm的。
        x_mel_cmvn = speechpy.processing.cmvn(x_mel, variance_normalization=False)
        x_mel_cmvn = x_mel_cmvn.T

        x_mel_cmvn = torch.from_numpy(x_mel_cmvn).unsqueeze(0)

        # print('x_mel_cmvn')
        # print(x_mel_cmvn.shape)
        #### END
        return x_wav, x_mel_cmvn, label

    def __len__(self):
        return len(self.file_path_list)


class WavMelClassifierDataset:
    def __init__(self, train_file_dict, sr, Selected_Machine_Type, machine_label_dict):
        self.train_file_dict = train_file_dict #一个字典，每个value存储的是不同设备的音频数据文件路径
        self.sr = sr # sample rate
        self.machine_type = Selected_Machine_Type # 选择构建数据集的设备类型，如果设为“ALL”，则表示构建所有的设备类型
        self.machine_label_dict = machine_label_dict # 设备名称和设备类型编号之间的关系字典


    def get_dataset(self,
                    n_fft=1024,
                    n_mels=128,
                    win_length=1024,
                    hop_length=512,
                    power=2.0):
        dataset = Wav_Mel_ID_Dataset(self.train_file_dict,
                                     self.machine_type,
                                     self.machine_label_dict,
                                     self.sr,
                                     win_length,
                                     hop_length,
                                     transform=Generator(
                                         self.sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         win_length=win_length,
                                         hop_length=hop_length,
                                         power=power,
                                     ))
        print("WavMelClassifierDataset Have Generated!")
        return dataset


if __name__ == '__main__':
    pass
