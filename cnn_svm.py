#coding:utf-8
import argparse
import sys
import numpy as np

import chainer
from chainer import cuda
from chainer import serializers
import resnet_evalfine

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,recall_score, precision_score, f1_score

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size_x, crop_size_y):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        
        
    def __len__(self):
        return len(self.base)
    
    def get_example(self,i):
        image, label = self.base[i]
        crop_size_x = self.crop_size_x
        crop_size_y = self.crop_size_y
        _, h, w = image.shape
        
        top = (h - crop_size_y) // 2
        left = (w - crop_size_x) // 2
        bottom = top + crop_size_y
        right = left + crop_size_x
        #平均画像を減算、255で割る
        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image /= 255   
        return image, label

parser = argparse.ArgumentParser()
parser.add_argument('test', help='Path to test image-label list file')
parser.add_argument("feature")
parser.add_argument("labels",default="labels_svm.txt")
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--initmodel', '-init', 
                    help='Initialize the model from given file')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--arch', '-a', default='resnet_evalfine')
parser.add_argument('--test_batchsize', '-b', type=int, default=1,
                    help='test minibatch size, you should not change this!!')
parser.add_argument('--mean', '-m', default='mean_svm.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--width', '-wid', type=int, default=256)
parser.add_argument('--height', '-hei', type=int, default=384)                
args = parser.parse_args()

#訓練済みモデル読み込み
archs = {    
        'resnet_evalfine': resnet_evalfine.Encoder
        }
model = archs[args.arch]()
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
else:
    print('cannot evaluate model!!')
    
#データ読み込み
mean = np.load(args.mean)
val = PreprocessedDataset(args.test, args.root, mean, args.width, args.height)
val_iter = chainer.iterators.MultiprocessIterator(
            val, args.test_batchsize, repeat=False, n_processes=args.loaderjob)

#訓練済みCNNで特徴抽出
feature = []
label = []
dim=2048
k = 1
N = len(val)
for i in val:
    label.append(int(i[1]))
    x = i[0].reshape((1,) + i[0].shape)
    x = chainer.Variable(xp.asarray(x))
    with chainer.using_config('train',False):
        y = model(x)
        y = y[0].data.reshape(dim)
        feature.append(y)
        sys.stderr.write('{} / {}\r'.format(k, N))
        sys.stderr.flush()
        k += 1
        
#標準化
sc = StandardScaler()
sc.fit(feature)
feature_norm=sc.transform(feature)

#グリッドサーチによるパラメータチューニング
parameters = [
    #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1 ,10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]}
    #{'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
    #{'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.01,0.001, 0.0001,0.00001]}
    ]
score = 'accuracy'    
clf = GridSearchCV(SVC(C=1), parameters, cv=5, scoring=score, n_jobs=-1)
clf.fit(feature_norm, label)
print(score)    
print("\n ベストパラメータ:\n")
print(clf.best_estimator_)
print("\n クロスバリデーションした時の平均スコア:\n")
for params, mean_score, all_scores in clf.grid_scores_:
    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

#訓練:テスト9:1にランダム分割してSVM訓練・分類 ×100    
predict,true = [],[]
for i in range(100):
    feature_train,feature_val,label_train,label_val = train_test_split(feature_norm,label,test_size=0.1)
    svmclf = clf.best_estimator_
    svmclf.fit(feature_train, label_train)
    predict_label = svmclf.predict(feature_val)
    for pre in predict_label:
        predict.append(pre)
    for tru in label_val:
        true.append(tru)
    sys.stderr.write('{}\r'.format(i))
    sys.stderr.flush()

#分類結果計算
average = "macro"
print(confusion_matrix(true,predict))
print(accuracy_score(true,predict))
print(precision_score(true,predict,average=average))
print(recall_score(true,predict,average=average))
print(f1_score(true,predict,average=average))
