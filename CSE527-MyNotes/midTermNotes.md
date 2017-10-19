# Mid Term Notes
- Image Features
    - FAST
    - MOPS
    - SIFT
- Feature Matching
    - Strategies
    - ROC curve
- Image stitching
- Object detection
    - Viola-Jones
    - Bag of Visual Words
- Dimensionality reduction
- Tracking
    - Optical flow
    - Mean shift, CamShift
    - Kalman Filter
    - Particle Filter


# Image Features
## FAST (Features from Accelerated Segment Test)
- checks only few neighbours to detect corner
- applies machine learning to decide which samples to take first with a decision tree

## MOPS (Multi-scale Oriented Patches)
Use the detector’s orientation estimate (from the Harris matrix large eigenvector) and the scale (from a pyramid), to extract a patch which is then normalized w.r.t scale, orientation and intensity.

## SIFT (Scale Invariant Feature Transform)
Probably the most well-known feature descriptor
### Key idea
1. create local gradient histograms around a key point
2. Normalize patch with scale & orientation from detector
- Can handle change in viewpoint, intensity; may be
implemented to be very fast.

# Feature Matching
Given features of image 1 and of image 2, how do
we find the best matching of the two sets?
- Define a distance metric between a pair of
feature descriptors
- Find the best matching
  - efficient data structures e.g. k-d trees

**Naive** : Pick the nearest neighbor (NN), set a
threshold on distance.
## Strategies
- Ratio Test
- Reciprocacy

## ROC Receiver Operator Characteristic Curve
We vary the threshold and take the one
that has highest TP-rate, lowest FP-rate.
### Recall
Ratio of true positives w.r.t total positives
### Precision
1 - Ratio of false positives w.r.t total negatives

## Feature Matching (k-d tree)
- Preprocess the “training” dataset
- Accelerate the matching process

# Image Alignment and Stitching
Image Alignment as Fitting Problem
## Linear Fitting
- Least Squares
- RANSAC (RAndom SAmple Consensus)
  - RANSAC idea: use only a subset of the points, discard outliers from the model fitting
- Affine Transforms/ Alignment
- Homography based alignments
- Cylindrical Warping

# Object Detection
- Template Matching
    - Not useful - gives lot of false positives

- Detection by Classification
  - Nearest Neighbor Classifier
    - Pros: Simple, flexible decision boundaries
    - Cons: Slow (must check all examples), doesn't learn,
      has parameter k (difficult to choose)
  - Naive Bayes Classifier (Generalizing models)
  - Linear Classifier
  - Fast Object Detection (Viola-Jones)

# Viola-Jones Fast Object Detection
- Haar like features
  - Simple Operations
  - Pick the best features
  - Efficient Computation using Integral Image and Dynamic Programming

- Feature Selection
  - Treat Each feature as binary classifier
  - Pick threshold that minimizes the FP rate.
  - Re-weight samples: misclassified are stronger and correctly classified are weaker.
  - This process is called Adaptive Booting (AdaBoost)

[Face Detection using Haar Cascades](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) : OpenCV Tutorials

``` python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)
```
where faces are tuples in the form : (x,y,w,h)

- Problems in Haar Features
  - Features may be too generic, too low-level
  - May only work for some very simple objects like faces, (faces don't vary much)
  - May not translation invariant : We rely on expensive sliding window

- We model the class boundary in the parametric space of feature responses - it’s a discriminative model (not necessarily a bad thing!)

# Bag of Visual Words
Pipeline
- Extract features (e.g. SIFT)
- Cluster features from all images - learn Vocabulary
- Represent images as histogram of the vocabulary
- Train a classfier

# Dimensionality Reduction
To maintain the most information we’d like to maintain the most (i.e. maximize) variance.
What is the vector that maximizes the var (i.e. diagonalizes covariance matrix) ?
- Most Prominent Eigen Vectors
- SVD

# Probabilistic Framework for Tracking
## Discriminative
Find boundaries between classes
- Model p(y|x) - “probability of class y given
data x”
- Examples: SVM, Linear Classifiers
## Generative Models
Model the class probabilities directly
- Model p(x|y) - “how does data x look like
within class y”
- Examples: Naive Bayes
## Measurement Model
- Generative Model
- Model the data
## Temporal Model
- Markovianity: The state depends only on the previous state.

# Mean Shift
- start with know patch/ feature histogram
- convert the frame into image of confidence map using back-projection of histogram
``` python
# builds a confidance map using back projection of histogram
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
```
- iteratively move towards mean of probability density inside the window
- at convergence window will reach in the region where probability of finding the desired object is maximum.

# CAM-Shift
- Continuously Adaptive Mean-shift
- window is in the form of an ellipse
- recalculates the histogram of object after few frames
- resizes the window after each convergence step.

# Kalman Filter
- Linear model for state transition and measurement
- combines prior and measurement with assumed error distribution to calculate future state.
## Problems
- linear model
- difficult to choose parameters that govern error distribution

# Particle Filter
- overcomes the difficulty of predefined error distribution faced in Kalman filter
- vector field for propagation


# Optical Flow
Apparent motion of the object or surface in the scene. Apparent motion contains Camera motion, lighting other distortions.
## Lucas and Kanade algorithm
The close area around a pixel moves the
same way: Spatial Coherence.
- uses Harris operator to solve over-constrained system of equations
- requires A^(T)A to be invertible.
- Eigenvalues shouldn't be too small or too large.
## Shi and Tomasi
- good feature selection
  - which tackles the aperture problem
## Problems
  - no 'visual memory'
  - points tracked individually

