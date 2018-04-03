import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import matplotlib.animation as animation
import time as time
import sys
from sklearn.cluster import KMeans
import models
from flask import Flask, request, jsonify
app = Flask(__name__)


sess = tf.Session()

input_node = tf.placeholder(tf.float32, shape=(None, 228, 304, 3))
net = models.ResNet50UpProj({'data': input_node}, 1, 1, False)

saver = tf.train.Saver()     
saver.restore(sess, '/home/ally/IENC/NYU_FCRN.ckpt')

def predict(img):
	global sess, saver, net, input_node
	# Default input size
		
	img = cv2.resize(img, (304,228))
	img = np.array(img).astype('float32')
	img = np.expand_dims(np.asarray(img), axis = 0)

	pred = sess.run(net.get_output(), feed_dict={input_node: img})
	plt.show()
	ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')

	plt.savefig('/home/ally/IENC/fo.png')
	return detect()


def check(image):
    lower = [150, 0, 0]
    upper = [255, 250, 200]
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    #output = cv2.bitwise_and(image, image, mask = mask)

    # show the images 
    #cv2.imshow("images", output)
    #cv2.waitKey(1)
	
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower_blue = np.array([110,50,10])
    #upper_blue = np.array([240,255,250])
    #dst = cv2.inRange(image, lower_blue, upper_blue)
    
    blue_pix = cv2.countNonZero(mask)
    tot_pix = image.shape[0]*image.shape[1]
    #cv2.imshow('frame', dst)
    #cv2.waitKey(1)
    frac = float(blue_pix)/tot_pix
    return frac



def detect():
    img = cv2.imread('fo.png')

    crop_img = img[60:540, 110:711]
    #cv2.imwrite('f.png',crop_img)
    roi_l = img[60:540, 110:310]
    roi_c = img[60:540, 310:510]
    roi_r = img[60:540, 510:711]

    image = roi_c
    pe = check(image)

    #print pe
    if( pe<0.3 ):
        print("No Obstacle continue moving straight")  
    else:
        pl = check(roi_l)
        pr = check(roi_r)
        #print pl
        #print pr
        if(pl < pr):
            print("Obstacle Detected Turn Left")
        else:
            print("Obstacle Detected Turn Right")   


@app.route('/get_depth', methods=['POST'])
def get_depth():
	image = cv2.imread(request.form['path'])
	result = predict(image)
	return jsonify(result)


def centroid_histogram(clt):
  
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	hist = hist.astype("float")
	hist /= hist.sum()

 
	return hist

def plot_colors(hist, centroids):
  
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	pe =0
	t=0
	y=0
	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
		colo = color.astype("uint8").tolist()
		#print colo
		if(colo[2] > 200  and colo[1] < 50 and colo[0] < 50):
			pe = max(percent,pe)
		startX = endX
	
	return pe

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7980)
