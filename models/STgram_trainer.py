import os
import sklearn
import numpy as np
import time
import re

import torch
import librosa
import matplotlib.pyplot as plt
from visdom import Visdom
from tqdm import tqdm
from utils import save_checkpoint, get_machine_id_list, create_test_file_list
from dataset_preprocess import Generator

import utils
import pickle


# torch.manual_seed(666)


class wave_Mel_MFN_trainer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.classifier = kwargs['classifier'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        visdom_folder = f'./log/'
        os.makedirs(visdom_folder, exist_ok=True)
        visdom_path = os.path.join(visdom_folder, f'{self.args.version}_visdom_ft.log')
        self.writer = Visdom(env=self.args.version, log_to_filename=visdom_path, use_incoming_socket=False)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.recon_criterion = torch.nn.L1Loss().to(self.args.device)

        self.csv_lines = []

    def train(self, train_loader):
        ##直接上来就先eval，来测试eval的性能对不对
        self.eval()
        n_iter = 0

        # create model dir for saving
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)

        print(f"Start classifier training for {self.args.epochs} epochs.")

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        for epoch_counter in range(self.args.epochs):
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for waveform, melspec, labels in pbar:
                waveform = waveform.float().unsqueeze(1).to(self.args.device)
                melspec = melspec.float().to(self.args.device)
                labels = labels.long().squeeze().to(self.args.device)
                self.classifier.train()
                predict_ids, _ = self.classifier(waveform, melspec, labels)
                loss = self.criterion(predict_ids, labels)
                pbar.set_description(f'Epoch:{epoch_counter}'
                                     f'\tLclf:{loss.item():.5f}\t')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.line([loss.item()], [n_iter],
                                     win='Classifier Loss',
                                     update='append',
                                     opts=dict(
                                         title='Classifier Loss',
                                         legend=['loss']
                                     ))
                    if self.scheduler is not None:
                        self.writer.line([self.scheduler.get_last_lr()[0]], [n_iter],
                                         win='Classifier LR',
                                         update='append',
                                         opts=dict(
                                             title='AE Learning Rate',
                                             legend=['lr']
                                         ))

                n_iter += 1

            if self.scheduler is not None and epoch_counter >= 20:
                self.scheduler.step()
            print(f"Epoch: {epoch_counter}\tLoss: {loss.item()}")
            if epoch_counter % 2 == 0:
                # save model checkpoints
                auc, pauc = self.eval()
                self.writer.line([[auc, pauc]], [epoch_counter], win=self.machine_type,
                                 update='append',
                                 opts=dict(
                                     title=self.machine_type,
                                     legend=['AUC_clf', 'pAUC_clf']
                                 ))
                print(f'{self.machine_type}\t[{epoch_counter}/{self.args.epochs}]\tAUC: {auc:3.3f}\tpAUC: {pauc:3.3f}')
                if (auc + pauc) > best_auc:
                    no_better = 0
                    best_auc = pauc + auc
                    p = pauc
                    a = auc
                    e = epoch_counter
                    checkpoint_name = 'checkpoint_best.pth.tar'
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'clf_state_dict': self.classifier.module.state_dict() if self.args.dp else self.classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False,
                        filename=os.path.join(self.args.model_dir, self.args.version, checkpoint_name))
            else:
                no_better += 1
            # if no_better > self.args.early_stop:
            #     break

            # if epoch_counter % 10 == 0:
            #     checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            #     save_checkpoint({
            #         'epoch': epoch_counter,
            #         'clf_state_dict': self.classifier.state_dict(),
            #         'optimizer': self.optimizer.state_dict(),
            #     }, is_best=False,
            #         filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))

        print(f'Traing {self.machine_type} completed!\tBest Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')

    def eval(self):
        print("="*20)
        print("Start Eval")
        sum_auc, sum_pauc, num, total_time = 0, 0, 0, 0
        #
        # sum_auc_r, sum_pauc_r = 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')#获得data_dir目录下所有的子目录
        # ['D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\bearing', 'D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\fan', 'D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\gearbox', 'D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\slider',
        # print('\n' + '=' * 20)
        # print(dirs)

        performance = []
        recore_dict = {}
        start = time.perf_counter()
        #通过for 循环，依次进入每个 dirs下的子文件夹
        for index, target_dir in enumerate(sorted(dirs)):
            # print(target_dir)

            machine_type = os.path.split(target_dir)[1]
            # print(machine_type)
            if machine_type not in self.args.process_machines:
                continue
            num += 1

            # 从target_dir_test 文件夹中，构建测试数据的样本和y_true
            #y_true ： 0--表示normal  1--表示abnormal
            test_files, y_true = create_test_file_list(target_dir, dir_name='test')
            y_pred = [0. for _ in test_files]
            # y_pred_recon = [0. for _ in test_files]
            # print(111, len(test_files), target_dir)
            #依次加载测试test文件夹下的wav文件
            for file_idx, file_path in enumerate(test_files):
                filename = os.path.split(file_path)[-1]
                sec_num = filename.split('_')[1]
                machine_type_sec_num = machine_type + '_sec_' + sec_num
                x_wav, x_mel, label = self.transform(file_path, machine_type_sec_num)
                with torch.no_grad():
                    self.classifier.eval()
                    net = self.classifier.module if self.args.dp else self.classifier
                    # out 是预测类别的输出，feature则是输入arcfacehead之前的特征
                    predict_ids, feature = net(x_wav, x_mel, label)
                probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                y_pred[file_idx] = probs[label]
            max_fpr = 0.1
            auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
            p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            recore_dict[machine_type] = {'auc':auc, 'pauc':p_auc}

        # calculate averages for AUCs and pAUCs
        averaged_performance = np.array(performance, dtype=float)
        averaged_performance = np.mean(averaged_performance, axis=0)
        # print(averaged_performance)
        mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
        # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
        # sum_auc += mean_auc
        # sum_pauc += mean_p_auc
        sum_auc = mean_auc
        sum_pauc = mean_p_auc

        time_nedded = time.perf_counter() - start
        print(f'Total test time: {time_nedded} secs!')
        print("Testing AUC for every Machine:")
        for key in recore_dict.keys():
            print(key + "-->")
            print(recore_dict[key])
        print("mean_AUC-->" + str(sum_auc))
        print("mean_pAUC-->" + str(sum_pauc))
        return sum_auc, sum_pauc


    def test(self, save=True):
        recore_dict = {}
        if not save:
            self.csv_lines = []

        sum_auc, sum_pauc, num = 0, 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # result csv
            self.csv_lines.append([machine_type])
            self.csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'{machine_type}_anomaly_score_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path, machine_type, id_str)
                    with torch.no_grad():
                        self.classifier.eval()
                        net = self.classifier.module if self.args.dp else self.classifier
                        predict_ids, feature = net(x_wav, x_mel, label)
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                #
                self.csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
            recore_dict[machine_type] = mean_auc + mean_p_auc
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            self.csv_lines.append(['Average'] + list(averaged_performance))
        self.csv_lines.append(['Total Average', sum_auc / num, sum_pauc / num])
        print('Total average:', sum_auc / num, sum_pauc / num)
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, self.csv_lines)
        return recore_dict

    def transform(self, file_path, machine_type):

        label = int(self.id_factor[machine_type])
        label = torch.from_numpy(np.array(label)).long().to(self.args.device)
        (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

        x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x_wav)
        x_wav = x_wav.float().to(self.args.device)

        x_mel = x[:self.args.sr * 10]  # (1, audio_length)
        x_mel = torch.from_numpy(x_mel)
        x_mel = Generator(self.args.sr,
                          n_fft=self.args.n_fft,
                          n_mels=self.args.n_mels,
                          win_length=self.args.win_length,
                          hop_length=self.args.hop_length,
                          power=self.args.power,
                          )(x_mel).unsqueeze(0).unsqueeze(0).to(self.args.device)
        return x_wav, x_mel, label


    def abnormal_eval(self):
        print("="*20)
        print("Start Eval")
        sum_auc, sum_pauc, num, total_time = 0, 0, 0, 0
        #
        # sum_auc_r, sum_pauc_r = 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')#获得data_dir目录下所有的子目录
        # ['D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\bearing', 'D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\fan', 'D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\gearbox', 'D:\\Coding_Work\\ASD_lgy\\data\\dev_dataset\\slider',
        # print('\n' + '=' * 20)
        # print(dirs)

        performance = []
        recore_dict = {}
        start = time.perf_counter()
        #通过for 循环，依次进入每个 dirs下的子文件夹
        for index, target_dir in enumerate(sorted(dirs)):
            # print(target_dir)

            machine_type = os.path.split(target_dir)[1]
            # print(machine_type)
            if machine_type not in self.args.process_machines:
                continue
            num += 1

            # 从target_dir_test 文件夹中，构建测试数据的样本和y_true
            #y_true ： 0--表示normal  1--表示abnormal
            test_files, y_true = create_test_file_list(target_dir, dir_name='test')
            y_pred = [0. for _ in test_files]
            # y_pred_recon = [0. for _ in test_files]
            # print(111, len(test_files), target_dir)
            #依次加载测试test文件夹下的wav文件
            for file_idx, file_path in enumerate(test_files):
                filename = os.path.split(file_path)[-1]
                sec_num = filename.split('_')[1]
                machine_type_sec_num = machine_type + '_sec_' + sec_num
                x_wav, x_mel, label = self.transform(file_path, machine_type_sec_num)
                with torch.no_grad():
                    self.classifier.eval()
                    net = self.classifier.module if self.args.dp else self.classifier
                    # out 是预测类别的输出，feature则是输入arcfacehead之前的特征
                    predict_ids, feature = net(x_wav, x_mel, label)

                """START Abnormal Detection"""
                #START Abnormal Detection##################
                # y_true ： 0--表示normal  1--表示abnormal
                probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                y_pred[file_idx] = probs[label]
                # print(probs)
                # if(probs[label]>0.5):
                #     y_pred[file_idx] = 0
                # else:
                #     y_pred[file_idx] = 1
                #END Abnormal Detection##################
                """END Abnormal Detection"""

            max_fpr = 0.1
            # print(y_true)
            # print(y_pred)
            auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
            p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            recore_dict[machine_type] = {'auc':auc, 'pauc':p_auc}

        # calculate averages for AUCs and pAUCs
        averaged_performance = np.array(performance, dtype=float)
        averaged_performance = np.mean(averaged_performance, axis=0)
        mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
        sum_auc = mean_auc
        sum_pauc = mean_p_auc

        time_nedded = time.perf_counter() - start
        print(f'Total test time: {time_nedded} secs!')
        print("Testing AUC for every Machine:")
        for key in recore_dict.keys():
            print(key + "-->" + str(recore_dict[key]))
            # print(recore_dict[key])
        print("mean_AUC-->" + str(sum_auc))
        print("mean_pAUC-->" + str(sum_pauc))
        return sum_auc, sum_pauc

    def ASD_train_test_data_generation(self, args):
        embed_size = 128
        print("="*20)
        """For Testing data"""
        print("Start Generate ASD Testing Data")
        dirs = utils.select_dirs(self.data_dir, data_type='')#获得data_dir目录下所有的子目录

        start = time.perf_counter()
        test_data_dict = {}
        for key in args.ID_factor.keys():
            test_data_dict[key] = {}
            test_data_dict[key]['predict_logits'] = np.zeros([1, args.num_class])
            test_data_dict[key]['normal_label'] = np.array([0])
            test_data_dict[key]['embeddings'] = np.zeros([1, embed_size])
        #通过for 循环，依次进入每个 dirs下的子文件夹
        for index, target_dir in enumerate(sorted(dirs)):
            # print(target_dir)

            machine_type = os.path.split(target_dir)[1]
            # print(machine_type)
            if machine_type not in self.args.process_machines:
                continue

            # 从target_dir_test 文件夹中，构建测试数据的样本和y_true
            #y_true ： 0--表示normal  1--表示abnormal
            test_files, y_true = create_test_file_list(target_dir, dir_name='test')
            #依次加载测试test文件夹下的wav文件
            for file_idx, file_path in enumerate(test_files):
                filename = os.path.split(file_path)[-1]
                sec_num = filename.split('_')[1]
                machine_type_sec_num = machine_type + '_sec_' + sec_num
                x_wav, x_mel, label = self.transform(file_path, machine_type_sec_num)
                with torch.no_grad():
                    self.classifier.eval()
                    net = self.classifier.module if self.args.dp else self.classifier
                    # out 是预测类别的输出，feature则是输入arcfacehead之前的特征
                    predict_ids, feature = net(x_wav, x_mel, label)
                    test_data_dict[machine_type_sec_num]['predict_logits'] = np.vstack([test_data_dict[machine_type_sec_num]['predict_logits'], predict_ids.mean(dim=0).squeeze().cpu().numpy()])
                    test_data_dict[machine_type_sec_num]['embeddings'] = np.vstack([test_data_dict[machine_type_sec_num]['embeddings'],feature.mean(dim=0).squeeze().cpu().numpy()])
                    test_data_dict[machine_type_sec_num]['normal_label'] = np.hstack([ test_data_dict[machine_type_sec_num]['normal_label'], y_true[file_idx]])
        #剔除第一行占位
        for key in args.ID_factor.keys():
            test_data_dict[key]['predict_logits'] = test_data_dict[key]['predict_logits'][1:,:]
            test_data_dict[key]['normal_label'] = test_data_dict[key]['normal_label'][1:]
            test_data_dict[key]['embeddings'] = test_data_dict[key]['embeddings'][1:,:]
            print(key)
            print(test_data_dict[key]['predict_logits'].shape)
            print(test_data_dict[key]['normal_label'].shape)
            print(test_data_dict[key]['embeddings'].shape)

        """For Training data"""
        print("Start Generate ASD Training Data")
        dirs = utils.select_dirs(self.data_dir, data_type='')#获得data_dir目录下所有的子目录

        start = time.perf_counter()
        train_data_dict = {}
        for key in args.ID_factor.keys():
            train_data_dict[key] = {}
            train_data_dict[key]['predict_logits'] = np.zeros([1, args.num_class])
            train_data_dict[key]['normal_label'] = np.array([0])
            train_data_dict[key]['embeddings'] = np.zeros([1, embed_size])
        #通过for 循环，依次进入每个 dirs下的子文件夹
        for index, target_dir in enumerate(sorted(dirs)):
            # print(target_dir)

            machine_type = os.path.split(target_dir)[1]
            # print(machine_type)
            if machine_type not in self.args.process_machines:
                continue

            # 从target_dir_test 文件夹中，构建测试数据的样本和y_true
            #y_true ： 0--表示normal  1--表示abnormal
            test_files, y_true = create_test_file_list(target_dir, dir_name='train')
            #依次加载测试test文件夹下的wav文件
            for file_idx, file_path in enumerate(test_files):
                filename = os.path.split(file_path)[-1]
                sec_num = filename.split('_')[1]
                machine_type_sec_num = machine_type + '_sec_' + sec_num
                x_wav, x_mel, label = self.transform(file_path, machine_type_sec_num)
                with torch.no_grad():
                    self.classifier.eval()
                    net = self.classifier.module if self.args.dp else self.classifier
                    # out 是预测类别的输出，feature则是输入arcfacehead之前的特征
                    predict_ids, feature = net(x_wav, x_mel, label)
                    train_data_dict[machine_type_sec_num]['predict_logits'] = np.vstack([train_data_dict[machine_type_sec_num]['predict_logits'], predict_ids.mean(dim=0).squeeze().cpu().numpy()])
                    train_data_dict[machine_type_sec_num]['embeddings'] = np.vstack([train_data_dict[machine_type_sec_num]['embeddings'],feature.mean(dim=0).squeeze().cpu().numpy()])
                    train_data_dict[machine_type_sec_num]['normal_label'] = np.hstack([ train_data_dict[machine_type_sec_num]['normal_label'], y_true[file_idx]])

        # 剔除第一行占位
        for key in args.ID_factor.keys():
            train_data_dict[key]['predict_logits'] = train_data_dict[key]['predict_logits'][1:, :]
            train_data_dict[key]['normal_label'] = train_data_dict[key]['normal_label'][1:]
            train_data_dict[key]['embeddings'] = train_data_dict[key]['embeddings'][1:, :]
            print(key)
            print(train_data_dict[key]['predict_logits'].shape)
            print(train_data_dict[key]['normal_label'].shape)
            print(train_data_dict[key]['embeddings'].shape)

        #将数据保存本地
        f_save = open('./data/asd_test.pkl', 'wb')
        pickle.dump(test_data_dict, f_save)
        f_save.close()

        f_save = open('./data/asd_train.pkl', 'wb')
        pickle.dump(train_data_dict, f_save)
        f_save.close()



        return test_data_dict, train_data_dict