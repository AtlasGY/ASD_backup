#%%
import numpy as np
import pyod
import pickle
import sklearn
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
import scipy
#%%
'''
读取已经处理好的数据(通过运行my_train训练Arcface网络。然后再运行my_test即可得到)
其中，train_data_dict和test_data_dict 按照设备类型进行划分，每个设备类型下包括：predict_logits，normal_label，embeddings
predict_logits: Arcface分类网络对设备声纹的分类结果（没有经过softmax处理的）
embeddings: Arcface网络对设备声纹的embedding（就是ArcHead 模块的Input部分）
normal_label: 设备声纹正常与否的标签，0--表示normal  1--表示abnormal
'''
f_read = open('./data/asd_train.pkl', 'rb')
train_data_dict = pickle.load(f_read)
f_read.close()

f_read = open('./data/asd_test.pkl', 'rb')
test_data_dict = pickle.load(f_read)
f_read.close()
print(train_data_dict.keys())
print(test_data_dict.keys())

#%%
"""
对选定的设备machine——type的数据进行加载
"""
machine_type = 'slider'

logit_train = train_data_dict[machine_type]['predict_logits']
embed_train = train_data_dict[machine_type]['embeddings']

logit_test = test_data_dict[machine_type]['predict_logits']
embed_test = test_data_dict[machine_type]['embeddings']
normal_label_test_gt = test_data_dict[machine_type]['normal_label']
#%%
'''
使用GMM模型进行异常检测
'''
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5, n_init=5, random_state=42)

gmm.fit(embed_train)

y_pred = gmm.score_samples(embed_test)

#%%
'''
使用
'''

reversed_Y_true = np.abs(normal_label_test_gt -1)
max_fpr = 0.1
auc = sklearn.metrics.roc_auc_score(reversed_Y_true, y_pred)
p_auc = sklearn.metrics.roc_auc_score(reversed_Y_true, y_pred, max_fpr=max_fpr)

print("AUC = {:.3f}".format(auc))
print("p_AUC = {:.3f}".format(p_auc))

#%%
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(reversed_Y_true, y_pred);
roc_auc = auc(fpr, tpr)


## 绘制roc曲线图
plt.subplots(figsize=(7,5.5));
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc);
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('ROC Curve');
plt.legend(loc="lower right");
plt.show()