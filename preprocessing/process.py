import cv2

def pad(img, size = (128, 128)):
 padded = np.zeros(size, np.uint8)
 padded[64 - img.shape[0]/2:64 + img.shape[1]/2, 64 - img.shape[1]/2:64 + img.shape[1]/2] = img
 return padded

def mass(img, radius=60):
 mask = np.zeros(img.shape, np.uint8)
 cv2.circle(mask, (img.shape[0]/2, img.shape[1]/2), radius, 1, -1)
 mask = 1 - mask
 return cv2.moments(np.multiply(mask, img))['m00']

def rotate(img, step):
  imgs = [img]
  for amount in range(step, 360, step):
   r_mat = cv2.getRotationMatrix2D((cropped_img.shape[0]/2, cropped_img.shape[0]/2), amount, 1)
   imgs.append(cv2.warpAffine(img, r_mat, img.shape))
  return imgs

def transform(img_path):
 s_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
 s_img = 255 - s_img
 dim = int(max(s_img.shape)*(2**0.5)+0.9)
 l_img = pad(s_img, (dim, dim))

 start_radius = l_img.shape[0]/2 - 7
 use_radius = start_radius
 for radius in range(start_radius, 0, -1):
  if mass(l_img, radius) > 1.0:
   break
  else:
   use_radius = radius

 r = use_radius + 5

 cropped_img = l_img[l_img.shape[0]/2-r:l_img.shape[0]/2+r,l_img.shape[0]/2-r:l_img.shape[0]/2+r]

 img_128 = cv2.resize(cropped_img, (128,128), cv2.INTER_CUBIC)
 img_96 = pad(cv2.resize(cropped_img, (96,96), cv2.INTER_CUBIC), (128,128))
 img_64 = pad(cv2.resize(cropped_img, (64,64), cv2.INTER_CUBIC), (128,128))

 output = rotate(img_128, 30) + rotate(img_96, 30) + rotate(img_64, 30)

 return output

def process(img_path):
 images = transform(img_path)
 for i in range(len(images)):
  new_path = img_path.rsplit('.',1)[0] + '_' + str(i) + '.' + img_path.split('.')[-1]
  cv2.imwrite(new_path, images[i])

if __name__ == '__main__':
 img_path = str(raw_input('path: '))
 process(img_path)
