import argparse
import os
import numpy as np
import tensorflow as t
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import matplotlib.animation as animation
import time as time
import sys
from sklearn.cluster import KMeans
import models

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        #print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
	plt.savefig('fo.png')
        detect()
	plt.pause(0.001)
        #fig.colorbar(ii)
        #plt.show()
	
	sys.stdout.flush()
        
	return pred

def check(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([70,50,50])
    upper_blue = np.array([130,255,255])
    dst = cv2.inRange(image, lower_blue, upper_blue)
    blue_pix = cv2.countNonZero(dst)
    tot_pix = image.shape[0]*image.shape[1]
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
    if( pe<0.03 ):
        print("No Obstacle. Continue moving straight.")  
    else:
    #   print("Obstacle detected")
        iml = roi_l
        pl = check(iml)
        imr = roi_r
        pr = check(imr)
        #print pl
        #print pr
        if(pl < pr):
            print("Obstacle Detected. Turn Left.")
        else:
            print("Obstacle Detected. Turn Right.")

    
       
                
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
   
    args = parser.parse_args()

    
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

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
        # plot the relative percentage of each cluster
        #per.append(percent)
        #print (color)
        # a = (np.array([1, 6, 2]))
        # f = (color[0])
        # g = round(f * 10000000000,2)
        # h = (color[1])
        # j = round(h,2)
        # k = (color[2])
        # l = round(k / 100,2)
        
        # print g
        # print j
        # print l
        # print l
        # print l
        # if(g!=1.4 and y!=1):
        #   t+=1
        # else:
        #   y=1
        # if( g ==1.4  and j ==6.16 and l == 2.3  ):
        #   print ("Yess" )
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        colo = color.astype("uint8").tolist()
        #print colo
        if(colo[2] > 200  and colo[1] < 50 and colo[0] < 50):
            pe = max(percent,pe)
        startX = endX
    
    return pe

if __name__ == '__main__':
    main()

        

