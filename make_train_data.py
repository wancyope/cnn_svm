import sys
import argparse
import os
import shutil
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--out","-o",type=str,default="svm")
args = parser.parse_args()
if not os.path.isdir("images_"+args.out):
        os.mkdir("images_"+args.out)
train = open("train_"+args.out+".txt",'w')
labelsTxt = open('labels_'+args.out+'.txt','w')
number_of_files=[]
classNo=0
cnt = 0
cntsum = 0
for label in sorted(glob(args.input+"/*")):
        dirname=os.path.splitext(os.path.basename(label))[0]
        N = len(glob(label+'/*.jpg'))
        print(label)
        labelsTxt.write(dirname+"\n")
        cnt_2=0
        
        for image in sorted(glob(label+"/*.jpg")):
                imagepath = "images_"+args.out+"/image%07d" %cnt +".jpg"
                shutil.copyfile(image,imagepath)
                train.write(imagepath+" %d\n" % classNo)
                sys.stderr.write('{} / {}\r'.format(cnt_2, N))
                sys.stderr.flush()
                cnt += 1
                cnt_2 += 1
        classNo += 1
        cntsum += cnt
print(cntsum)
train.close()
labelsTxt.close()
