import cv2
import numpy as np
import os

def pad(img, size = 128):
 padded = np.zeros((size, size), np.uint8)
 height_start = size/2-img.shape[0]/2
 height_end = height_start + img.shape[0]
 width_start = size/2-img.shape[1]/2
 width_end = width_start + img.shape[1]
 padded[height_start:height_end,width_start:width_end] = img
 return padded

def mass(img, radius=60):
 mask = np.zeros(img.shape, np.uint8)
 cv2.circle(mask, (img.shape[0]/2, img.shape[1]/2), radius, 1, -1)
 mask = 1 - mask
 return cv2.moments(np.multiply(mask, img))['m00']

def rotate(img, step):
 imgs = [img]
 for amount in range(step, 360, step):
  r_mat = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[0]/2), amount, 1)
  imgs.append(cv2.warpAffine(img, r_mat, img.shape))
 return imgs

def find_radius(img):
 start_radius = img.shape[0]/2 - 7
 use_radius = start_radius
 for radius in range(start_radius, 0, -1):
  if mass(img, radius) > 1.0:
   break
  else:
   use_radius = radius
 return use_radius + 5

def transform(img_path):
 s_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
 s_img = 255 - s_img
 dim = int(max(s_img.shape)*(2**0.5)+0.9)
 l_img = pad(s_img, dim)

 r = find_radius(l_img)
 cropped_img = l_img[l_img.shape[0]/2-r:l_img.shape[0]/2+r,l_img.shape[0]/2-r:l_img.shape[0]/2+r]

 img_128 = cv2.resize(cropped_img, (128,128), interpolation = cv2.INTER_CUBIC)
 img_96 = pad(cv2.resize(cropped_img, (96,96), interpolation = cv2.INTER_CUBIC), 128)
 img_64 = pad(cv2.resize(cropped_img, (64,64), interpolation = cv2.INTER_CUBIC), 128)

 output = rotate(img_128, 30) + rotate(img_96, 30) + rotate(img_64, 30)

 return output

def consume(img_path):
 images = transform(img_path)
 for i in range(len(images)):
  new_path = img_path.rsplit('.',1)[0] + '_p_' + str(i) + '.' + img_path.split('.')[-1]
  cv2.imwrite(new_path, images[i])

def process(img_dir):
 for path, _, files in os.walk(img_dir):
  for file_name in files:
   extension = file_name.split('.')[-1]
   if extension.lower() not in ['jpg','png','bmp','gif','jpeg']:
    continue
   if '_p_' in file_name:
    continue
   img_path = os.path.join(img_dir, path, file_name)
   try:
    consume(img_path)
   except:
    print '(!) error with ' + img_path

if __name__ == '__main__':
 img_dir = str(raw_input('path: '))
 process(img_dir)
