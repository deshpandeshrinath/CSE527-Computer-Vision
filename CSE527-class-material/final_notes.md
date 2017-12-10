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

#### Clustering
##### Super pixels by Clustering
- Group together similar looking pixels
- Used in bottom up segmentation
- Unsupervised (mostly)

##### K-means
  - Pick k cluster centers.
  - For each pixel assign the __closest__ cluster.
  _ Recompute the cluster center (mean) from assigned pixels.
  - Repeat until converged.

###### Distance Metric in clustering
   - Pixel color intensities
   - Edges
   when flooding if you hit a strong edge then stop.

##### SLIC (Simple Linear Iterative Clustering)
__Key Idea__:
  - Limit the search space to a region proportional to the super-pixel size.
  - A weighted distance measure to combine color and spatial proximity.
_(considered state of the art in speed and quality)._

#### Graph Cut Approach
Consider an image as a graph, where edges have a weight proportional to image
similarity. (e.g. color, intensity, texture, etc.)

- Delete the weak links
  (Similarity Pixels should remain in the same cluster.)
_ __Minimum graph cut__ should give a __good segmentation.__

##### Normalized Cuts
Has different segmentation than minimum cuts.
Finds an equilibrium between cut parts. (No one is too small or too large)

##### Energy Based Graph Cut









  - Clustering
    - Super-pixels
  - Graph cuts
    - Binary
    - Energy based
- Multi-view Geometry (MVG)
  - Epipolar Geometry
  - Essential and Fundamental Matrices
  - Depth from Disparity
- Stereo
  - Dense Stereo
  - Structured Light
  - Camera Calibration
  - Projector-Camera Duality
  - Patterns and Laser-planes
- Structure from Motion
- Machine Learning
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
