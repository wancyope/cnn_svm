#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import sys
 
parser = argparse.ArgumentParser()
parser.add_argument("source_dir")
parser.add_argument("target_dir")
parser.add_argument("width",type=int, default=256)
parser.add_argument("height",type=int, default=256)
args = parser.parse_args()

if not os.path.isdir(args.target_dir):
  os.mkdir(args.target_dir)
i = 1
N = len(os.listdir(args.source_dir))
print(N)
  
for source_imgpath in os.listdir(args.source_dir):
  if ".jpg" in source_imgpath:  
      img = cv2.imread(args.source_dir+"/"+source_imgpath)        
      cropped_img = cv2.resize(img,(args.width,args.height))
      cv2.imwrite(args.target_dir + "/" + source_imgpath, cropped_img) 
      sys.stderr.write('{} / {}\r'.format(i, N))
      sys.stderr.flush()
      i += 1
