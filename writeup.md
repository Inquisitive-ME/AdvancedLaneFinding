## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./WriteupResources/undistort_checkerboard.png "Undistorted"
[image2]: ./WriteupResources/undistort_road.png "Road Undistorted"
[image3]: ./WriteupResources/binary_filter_light.png "Light Binary Example"
[image4]: ./WriteupResources/binary_filter_heavy.png "Heavy Binary Example"
[image5]: ./WriteupResources/warped_transform.png "Warp Example"
[image6]: ./WriteupResources/Finding_Lanes.png "Lane Find"
[image7]: ./WriteupResources/Finding_Lanes_2.png "Lane Find 2"

[image8]: ./WriteupResources/Result.png "Output"

[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./AdvancedLaneFinding.ipynb" and in lines 12-37 of the "./AdvancedLaneFinding.py" file

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied this distortion correction to one of the test images using the `cv2.undistort()` function with the parameters obtained from the camera calibration with the chessboard pictures and obtained this result: 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Cell 4 of the IPython notebook located in "./AdvancedLaneFinding.ipynb" shows the combination of color and gradient thresholds that were used to try to determine the best filter for extracting the lane lines from the image.

The HeavyFilter function in cell 5 and in lines 92-111 of the "./AdvancedLaneFinding.py" file contains the combination of thresholds on the saturation and red channels along with sobel x gradients on the saturation and red channels and a magnituded threshold on the saturation channel to create the final filter.

I originally had the LightFilter which is a combination of magnitude, and direction threshold holded combined with a sobel x gradient on the saturation channel of the image, but I was running into issues of getting too much noise on some of the poorer parts of the road so I moved to the HeavyFilter

Here's an example of my output for the LightFilter: 

![alt text][image3]

Here's an example of my output for the HeavyFilter:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `Warp()`, which appears in Cell 6 of the IPython notebook located in "./AdvancedLaneFinding.ipynb" and in lines 139-161 of the "./AdvancedLaneFinding.py" file. The `Warp()` function takes as inputs an image (`image`).  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 576, 453.6      | 384, 0        | 
| 128, 720      | 384, 720      |
| 1152, 720     | 896, 720      |
| 704, 453.6      | 896, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

in Cell 7 of the IPython notebook located in "./AdvancedLaneFinding.ipynb" and in lines 164-231 of the "./AdvancedLaneFinding.py" file. I performed a window searching method as described in the lectures to find the pixels that corresponds to the lanes then I fit a polynomial to those pixels for each lane. Below is a picture showing the windows and polynomial fit for each lane
![alt text][image6]

in Cell 8 of the IPython notebook located in "./AdvancedLaneFinding.ipynb" and in lines 245-272 of the "./AdvancedLaneFinding.py" file. I also implemented a function to search around the previous polynomial for the lines. Below is an image showing the region search based on the window search method and the line fit for the pixles found

![alt text][image7]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in Cell 9 and 10 of the IPython notebook located in "./AdvancedLaneFinding.ipynb" and in lines 275-291 of the "./AdvancedLaneFinding.py" file. Using the equations shown in lecture

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented a version of my lane finding algorithm on all the provided pictures in cell 11 of the IPython notebook located in "./AdvancedLaneFinding.ipynb".

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to use the functions provided in the lectures and try to handle any issues with bad lane lines by using information from the previous frames.

This approach did not work as well as I had hoped, and my pipeline ended up with problems whenever the lane lines are hard to distinguish from the road. I think the main issue I had was that the lecture functions I used were really geared around performing individual tasks, while the pipeline could use a combination of all the functions to obtain the result.

If I were to revist the  this project the main thing I would change is instead of developing the lane finding and filtering individual I would modify them together to try to get the best final result of the lanes. This way the lane finding could compensate for some of the areas that the filtering has issues with and vice virsa.

Another improvement would be to store the lane line points found in each frame instead of just the fitted polynomial coefficients. I think this would help a lot with making transitions between poor and good road conditions.

Overal, the performance is pretty good with the video for this project, however the algorithm fails on the challenge videos, and this is mainly because my binary filter is too restrictive to handle varying intensity of the lane lines. Whenever the lane lines are very light it is difficult to disquish them from the road.
