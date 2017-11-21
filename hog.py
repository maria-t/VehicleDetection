import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from features_functions import *
from sliding_window import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


# Read in and make lists of car images and non-car images from vehicle and non-vehicle folders respectively
cars = glob.glob('vehicles/*/*/*.png')
notcars = glob.glob('non-vehicles/*/*/*.png')

# Read in and plot random images from cars and notcars lists
car_image = mpimg.imread(cars[np.random.randint(0, len(cars))])	
notcar_image = mpimg.imread(notcars[np.random.randint(0, len(notcars))])

f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
f.tight_layout()
ax1.imshow(car_image)
ax1.set_title('Car Image', fontsize=12)
ax2.imshow(notcar_image)
ax2.set_title('Non-car Image', fontsize=12)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()


# Define HOG parameters
orient = 13
pix_per_cell = 8
cell_per_block = 2
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
# Define sliding window search parameters
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [395, 620] # Min and max in y to search in slide_window()


gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
# Call get_hog_features function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

# Plot the examples
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
f.tight_layout()
ax1.imshow(car_image, cmap='gray')
ax1.set_title('Example Car Image', fontsize=12)
ax2.imshow(hog_image, cmap='gray')
ax2.set_title('HOG Visualization', fontsize=12)
plt.title('HOG Visualization')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

# Define a function to extract features from a list of images
# Have this function call bin_spatial(), color_hist() and get_hog_features contained in the features_functions.py
def extract_features(imgs, cspace=colorspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for i in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(i)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Function that extracts features from a single image window
def single_img_features(img, cspace=colorspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat):
    
	#1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)



# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, cspace=colorspace, 
                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                    hist_feat=hist_feat, hog_feat=hog_feat):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, cspace=colorspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


######################
t=time.time()
# Testing on a smaller sample of the datasets
#sample_size = 1000
#random_idxs = np.random.randint(0, len(cars), sample_size)
#test_cars = np.array(cars)[random_idxs]
#test_notcars = np.array(notcars)[random_idxs]

cars = glob.glob('vehicles/*/*/*.png')
notcars = glob.glob('non-vehicles/*/*/*.png')
car_features = extract_features(cars, cspace=colorspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

########################################################################

# Function that adds "heat" to a map for a list of bounding boxes
def add_heat(heatmap, hot_windows):
    # Iterate through list of bboxes
    for box in hot_windows:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
  
# Function that adds a threshold to the heatmap  
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


################################################################################################################3

test_images = glob.glob('test_images/*.jpg')
#Iterate over test images
for test_image in test_images:
	image = mpimg.imread(test_image)
	draw_image = np.copy(image)
	image = image.astype(np.float32)/255 # the test images are .jpg images
	

	windows = slide_window(img = image, x_start_stop=[180, 1280], y_start_stop=y_start_stop, 
		                xy_window=(64, 64), xy_overlap=(0.6, 0.6))
	windows += slide_window(img = image, x_start_stop=[180, 1280], y_start_stop=y_start_stop, 
		                xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#	windows += slide_window(img = image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#		                xy_window=(107, 107 ), xy_overlap=(0.5, 0.5))
	windows += slide_window(img = image, x_start_stop=[180, 1280], y_start_stop=y_start_stop, 
		                xy_window=(128,128), xy_overlap=(0.6, 0.6))	

	hot_windows = []
	hot_windows += (search_windows(image, windows, svc, X_scaler, cspace=colorspace, 
		                    spatial_size=spatial_size, hist_bins=hist_bins, 
		                    orient=orient, pix_per_cell=pix_per_cell, 
		                    cell_per_block=cell_per_block, 
		                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
		                    hist_feat=hist_feat, hog_feat=hog_feat))                       

	window_img = draw_boxes(draw_image, hot_windows, color=(255, 165, 0), thick=6)                    
#	plt.imshow(window_img)
#	plt.show()	

	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat,hot_windows)
		
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,2)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
#	plt.imshow(heatmap)
#	plt.show()
	
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(draw_image), labels)
#	plt.imshow(draw_img)
#	plt.show()

"""
	# Plot the examples
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(9,7))
	f.tight_layout()
	ax1.set_title('Test Image', fontsize=12)
	ax1.imshow(image)
	ax2.set_title('Identified Bounding Boxes', fontsize=12)
	ax2.imshow(window_img)	
	ax3.set_title('Heatmap', fontsize=12)
	ax3.imshow(heatmap, cmap='gray')
	ax4.set_title('After removing false positives', fontsize=12)
	ax4.imshow(draw_img)
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	plt.show()
"""

###############################################################################################################
def detect_cars(img):
	# image = mpimg.imread(img)
	draw_image = np.copy(img)
	img = img.astype(np.float32)/255 # the test images are .jpg images
	

	windows = slide_window(img, x_start_stop=[200, 1280], y_start_stop=y_start_stop, 
		                xy_window=(64, 64), xy_overlap=(0.6, 0.6))
	windows += slide_window(img, x_start_stop=[200, 1280], y_start_stop=y_start_stop, 
		                xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#	windows += slide_window(img = image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#		                xy_window=(107, 107 ), xy_overlap=(0.5, 0.5))
	windows += slide_window(img, x_start_stop=[200, 1280], y_start_stop=y_start_stop, 
		                xy_window=(128,128), xy_overlap=(0.6, 0.6))	

	hot_windows = []
	hot_windows += (search_windows(img, windows, svc, X_scaler, cspace=colorspace, 
		                    spatial_size=spatial_size, hist_bins=hist_bins, 
		                    orient=orient, pix_per_cell=pix_per_cell, 
		                    cell_per_block=cell_per_block, 
		                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
		                    hist_feat=hist_feat, hog_feat=hog_feat))                       

	window_img = draw_boxes(draw_image, hot_windows, color=(255, 165, 0), thick=6)                    
#	plt.imshow(window_img)
#	plt.show()	

	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat,hot_windows)
		
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,2)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
#	plt.imshow(heatmap)
#	plt.show()
	
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(draw_image), labels)

	return draw_img


output = 'project_video_output.mp4'

clip1 = VideoFileClip('project_video.mp4')

video_clip = clip1.fl_image(detect_cars) #NOTE: this function expects color images!!
video_clip.write_videofile(output, audio=False)







