import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

# Read in images
images = glob.glob('camera_cal/calibration*.jpg')

# plt.imshow(img)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object pints, like (0,0,0), (1,0,0), (2,0,0), ..., (9,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates
for image in images:
    # read in each image
    img = mpimg.imread(image)
    # convert to grayscale    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # if corners are detected, add object points and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        #plt.imshow(img)

image = images[0]
print(image)
img = mpimg.imread(image)
cv2.imwrite('output_images/1_original.jpg', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

start = time.time()
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
print(time.time() - start)
cv2.imwrite('output_images/1_undistorted.jpg', undistorted)
#plt.imshow(undist)
# function takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    #objpoints = []
    #imgpoints = []
    
    #objp = np.zeros((6*8,3), np.float32)
    #objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    #imgpoints.append(corners)
    #objpoints.append(objp)
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # undist = np.copy(img)  # Delete this line
    return undist

#undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
