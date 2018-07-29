from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mping
import collections

#Calibrating Camera Lecture

cal_images = glob.glob('camera_cal/*.jpg')

objpoints = []
imgpoints = []

# prepare object points
objp = np.zeros((9*6,3),np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

for fname in cal_images:
        # Read Image
        image = mping.imread(fname)
        # Convert to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find Chess Board Corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If Corners are found, add object points, and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        
def undistortImage(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist=cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_threshold(gray, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def Threshold(img, thresh=(0, 255)):
    # 2) Apply a threshold
    result = np.zeros_like(img)
    result[(img > thresh[0]) & (img <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return result

def HeavyFilter(img, ksize = 13, mag_thresh = (45, 175), dir_thresh = (0.7, 1.25)):
    img = np.copy(img)
    #Get red channel
    r_channel = img[:,:,0]
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    R = Threshold(r_channel,(205,255))

    S = Threshold(s_channel,(95,255))

    gradxs = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh = (15,120))
    gradxr = abs_sobel_thresh(r_channel, orient='x', sobel_kernel=ksize, thresh = (10,120))
        
    mag_binary_s = mag_threshold(s_channel, sobel_kernel=ksize, thresh=(65, 175))
    
    combined = np.zeros_like(R)
    combined[((S == 1) & (R == 1)) | (mag_binary_s == 1) | ((gradxs == 1) & (gradxr == 1))] = 1
    
    return combined
def LightFilter(img, ksize=13, sx_thresh=(20, 100), mag_thresh = (45, 175), dir_thresh = (0.7, 1.25)):
    img = np.copy(img)
    # Convert to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Sobel x on s
    sobelx_s = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=sx_thresh)
    
    # Magnitude and direction on gray
    mag_binary_gray = mag_threshold(gray, sobel_kernel=ksize, thresh=mag_thresh)
    dir_binary_gray = dir_threshold(gray, sobel_kernel=ksize, thresh=dir_thresh) 
    
    # Combine
    binary = np.zeros_like(sobelx_s)
    binary[((mag_binary_gray == 1) & (dir_binary_gray == 1)) | (sobelx_s == 1)] = 1
    
    return binary

def Pipeline(image):
    undist = undistortImage(image, objpoints, imgpoints)
    return LightFilter(undist)

def Warp_Test(image):
    img_size = (image.shape[1], image.shape[0])
    # Move points around more to get better transformation
    src = np.float32(
        [[img_size[0] * 0.45, img_size[1] * 0.63],
        [ img_size[0] * .1, img_size[1]],
        [ img_size[0] * .9, img_size[1]],
        [ img_size[0] * 0.55, img_size[1] * 0.63]])

    dst = np.float32(
        [[img_size[0] * 0.3, 0],
        [img_size[0] * 0.3, img_size[1]],
        [img_size[0] * 0.7, img_size[1]],
        [img_size[0] * 0.7, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, img_size)
    return(src, dst, warped, Minv)

def Warp(image):
    src, dest, warped, Minv = Warp_Test(image)
    return (warped, Minv)


def find_lane_pixels(binary_warped):  
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]*0.65):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #leftx_base = int(midpoint/2 +125)
    #rightx_base = int(midpoint + midpoint/2 -75)
    
    # Choose the number of sliding windows
    nwindows = 16
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    return (leftx, lefty, rightx, righty)
        
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return (leftx, lefty, rightx, righty)


def getCurvature(x, y, y_eval):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = ym_per_pix* y_eval
    
    fit = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    
    # Calculate radii of curvature
    curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return (curverad)

def getCenterOffset(left_fitx, right_fitx, center):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
    return((lane_center - center) * xm_per_pix)

#Create pipeline for video
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.minPoints = 10
        self.nFilterFrames = 5
        
        # Is first Frame
        self.FirstFrame = True
        # was the line detected in last iteration
        self.detected = False
        
        # is line detected for current iteration
        self.valid = False
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = collections.deque(maxlen=self.nFilterFrames)
       
        # running average of polynomials
        self.allPolys = collections.deque(maxlen=self.nFilterFrames)

        
    def getBestFit(self):
        print("Number of Lines = ",len(self.allPolys))
        return np.mean(self.allPolys, axis = 0)
    
    def addPoly(self, fit, curvature):
        self.valid = True
        self.allPolys.append(fit)
        self.radius_of_curvature.append(curvature)
    
    def getCurvature(self):
        return np.mean(self.radius_of_curvature)
        
def VideoPipeline(img):
    global LeftLine
    global RightLine
    
    binaryWarped, Minv = Warp(Pipeline(img))
    
    ploty = np.linspace(0, binaryWarped.shape[0]-1, binaryWarped.shape[0])
    
    if(LeftLine.detected & RightLine.detected):
        left_fitx = LeftLine.getBestFit()
        right_fitx = RightLine.getBestFit()
        leftx, lefty, rightx, righty = search_around_poly(binaryWarped, left_fitx, right_fitx)
    else:
        leftx, lefty, rightx, righty = find_lane_pixels(binaryWarped)
    
    # Check if we found enough points
    if (rightx.size > RightLine.minPoints):
        RightLine.detected = True
    else:
        RightLine.detected = False
    if (leftx.size > LeftLine.minPoints):
        LeftLine.detected = True
    else:
        LeftLine.detected = False
        
    if(RightLine.detected & LeftLine.detected):
        if(RightLine.FirstFrame | LeftLine.FirstFrame):
            RightLine.FirstFrame = False
            LeftLine.FirstFrame = False
            
            leftCurve = getCurvature(leftx, lefty, 719)
            rightCurve = getCurvature(rightx, righty, 719)
            
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            LeftLine.addPoly(left_fit, leftCurve)
            RightLine.addPoly(right_fit, rightCurve)
        else:
            leftCurve = getCurvature(leftx, lefty, 719)
            rightCurve = getCurvature(rightx, righty, 719)
            # make sure curvature is close
            if (abs((min(leftCurve, rightCurve) / max(leftCurve, rightCurve))) > 0.4):
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)
                # make sure polynomial is in same direction
                if((left_fit[0] * right_fit[0]) > 0):
                    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

                    LeftLine.addPoly(left_fit, leftCurve)
                    RightLine.addPoly(right_fit, rightCurve)
                else:
                    print("Skipping Frame Polynomial")
                    LeftLine.valid = False
                    RightLine.valid = False
            else:
                print("Skipping Frame Curvature")
                LeftLine.valid = False
                RightLine.valid = False
    
    if (RightLine.FirstFrame & (not RightLine.detected) | LeftLine.FirstFrame & (not LeftLine.detected)):
        # Skip
        return image
    elif(not LeftLine.valid):
        left_fit = LeftLine.getBestFit()
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = RightLine.getBestFit()
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binaryWarped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 1.0, 0)

    curv = (LeftLine.getCurvature() + RightLine.getCurvature())/2.0
    curvature = "Estimated lane curvature %.2fm" % (curv)

    distance_center = "Estimated offset from lane center %.2fm" % (getCenterOffset(left_fitx, right_fitx, result.shape[1]/2.0))

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, curvature, (30,60), font, 1.5, (255,0,0), 2)
    cv2.putText(result, distance_center, (30,120), font, 1, (255,0,0), 2)
        
    return result

LeftLine = Line()
RightLine = Line()
    
write_output = 'result.mp4'
#clip1 = VideoFileClip("project_video.mp4")
clip1 = VideoFileClip("project_video.mp4")
write_clip = clip1.fl_image(VideoPipeline)
write_clip.write_videofile(write_output, audio=False)