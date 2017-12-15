# Computer Vision - Part 2 (After Midterm)

## Syllabus
- Segmentation
  - Clustering
  - Graph cuts
- Multi-view Geometry (MVG)
  - Epipolar Geometry
  - Essential and Fundamental Matrices
  - Depth from Disparity
- Stereo
  - Dense Stereo
  - Structured Light
- Structure from Motion
- Machine Learning
  - Nearest Neighbor Classifiers
  - Linear Classifiers
  - Generalization
  - Optimization
  - Neural Networks
- Deep Learning
  - Concerns
  - Segmentation
  - Auto-encoders
  - Variational Auto-encoders
  - Generative Adversarial Networks


## Segmentation
Goal is to separate/ group image pixels to objects
Segmentation is an hard, subjective problem even for humans which may require
prior information, depth-motion cues.
### Gestalt Principals
The _whole_ is other than its _parts_.
__Key Idea__: The human visual system is always looking for patterns and ways to
connect things to _wholes_.
The idea was heavily criticized although being very intuitive but hard to
represent algorithmically:
  - Sparse
  - Relies on deep prior knowledge
  - Ambiguous, confusing
  - Subtle, low constrast
  - Impractical, just for show

### Segmentation Approaches
- Bottom up:
  - Start with pixel clusters (Super Pixels)
  - Gradually connect more regions

- Top Down:
  - Divide entire image to big area
  - Continue to divide up to the details

####  Semi-supervised Segmentation
  - Markings
  - Bounding box
  - Interactive boundary
  - Example based

### Segmentation methods
- Threshold
- Clustering
- Graph-cut

### Clustering
#### Super pixels by Clustering
- Group together similar looking pixels
- Used in bottom up segmentation
- Unsupervised (mostly)

#### K-means
  - Pick k cluster centers.
  - For each pixel assign the __closest__ cluster.
  _ Recompute the cluster center (mean) from assigned pixels.
  - Repeat until converged.

##### Distance Metric in clustering
   - Pixel color intensities
   - Edges
   when flooding if you hit a strong edge then stop.

#### SLIC (Simple Linear Iterative Clustering)
__Key Idea__:
  - Limit the search space to a region proportional to the super-pixel size.
  - A weighted distance measure to combine color and spatial proximity.
_(considered state of the art in speed and quality)._

### Graph Cut Approach
Consider an image as a graph, where edges have a weight proportional to image
similarity. (e.g. color, intensity, texture, etc.)

- Delete the weak links
  (Similarity Pixels should remain in the same cluster.)
_ __Minimum graph cut__ should give a __good segmentation.__

#### Normalized Cuts
Has different segmentation than minimum cuts.
Finds an equilibrium between cut parts. (No one is too small or too large)

#### Energy Based Graph Cut
Find a labeling that minimizes the energy (cost)
There are two parts of the cost
- Match cost
  Usually computed by creating a color model from user-labeled pixels
- Smoothness cost
  Neighboring pixels should generally have the same labels unless the pixels
  have a very different intensities.

__Solve by Min Cut - Max Flow__

- Multi-view Geometry (MVG)
  - Epipolar Geometry
    A point in image L conforms with a line in image R, where L and R are the
    images from a displaced and rotated camera.
      _Epipoles_: The perceived location of the camera in another image.

  - Essential Matrix
    This matrix gives spatial and orientational relations of two views taken
    from normalized cameras.
  - Fundamental Matrix
    This matrix gives spatial and orientational relations of two views taken
    from general cameras.
  - Depth from Disparity

## Stereo
Basic Stereo Matching Algorithm
  1. Scan Epilines
  2. Find the best match (for e.g. min change in intensity)
  3. Compute Disparity
  4. Calculate depth
  5. Repeat

Improvement: Window method
- In a window, take the best point and slid and find the same point in other
  image.
* Problems:
  - When not enough texture is found
  - Occlusions are present
  - Repetitions in image (confusing)
  - Very large motion
  - Non camera motion
  - Violations of brightness constancy
  - Camera Calibration Errors

### Stereo is Energy Minimization
Based on Assumption that,
  - Brightness constancy
  - Matching pixels will have a high matching score.
  - Neighboring pixels will have similar disparity.
We formulate the matching problem with a cost function.

###  - Dense Stereo
###  - Structured Light
###  - Camera Calibration

## Structure from Motion
Algorithm

  1. Extract corners/ interest points
  2. Get feature descriptors
  3. Match Features
  4. Prune features matching by applying a geometric model
    - Affine
    - Homography
    - Essential/ Fundamental matrix
  5. Find camera extrinsics
  6. Triangulate 3D from stereo
  7. Bundle Optimization

### Multiple View Stereo
Input: Calibrated images from several viewpoints
Output: 3D Object Model

* Pick a reference image, and slide the corresponding window along with epipolar
  lines of all other images using inverse depth relative to the first image as
  the _search_ parameters

__Optimal baseline?__:
  - Too small: Large depth error
  - Too large: Difficult _search_ problem

## Machine Learning
  - Nearest Neighbor Classifiers
  - Linear Classifiers
  - Generalization
  - Optimization
  - Neural Networks
    - Perceptrons
    - Multi-layer Perceptrons
    - Gradients - Backpropagation
    - Activation functions
    - Softmax again
- Deep Learning
  - Convolution layers
    - Padding, Stride
    - Pooling
    - Region-CNNs
  - ML problems
    - Overfitting
    - Regularization
    - Problems with data
  - Segmentation
  - Auto-encoders
  - Variational Auto-encoders
  - Generative Adversarial Networks
