# STEAM

## _Project Information_
- Performed by DXLAB at Gyeongsang National University
- Supported by National Research Foundation
- Duration: 2023.04.01. - 2027.12.31.

## _Contributors_
- Seonghyeon Moon(Project Director)
- <strong>Yutak Seo</strong>
- Haejun Seong
- Seokhye Lee

## _Collaboration_
- Seoul National University
- SmartInside AI
- Glotechsoft

## _Genetic Algorithm for Sensor Deployment Optimization_
**Genetic Algorithms** (GA) are based on an **evolutionary approach** to AI, in which methods of the evolution of a population is used to obtain an optimal solution for a given problem. They were proposed in 1975 by [John Henry Holland](https://wikipedia.org/wiki/John_Henry_Holland).

Genetic Algorithms are based on the following ideas:

* Valid solutions to the problem can be represented as **genes**
* **Crossover** allows us to combine two solutions together to obtain a new valid solution
* **Selection** is used to select more optimal solutions using some **fitness function**
* **Mutations** are introduced to destabilize optimization and get us out of the local minimum

If you want to implement a Genetic Algorithm, you need the following:

 * To find a method of coding our problem solutions using **genes** g&in;&Gamma;
 * On the set of genes &Gamma; we need to define **fitness function** fit: &Gamma;&rightarrow;**R**. Smaller function values correspond to better solutions.
 * To define **crossover** mechanism to combine two genes together to get a new valid solution crossover: &Gamma;<sup>2</sub>&rightarrow;&Gamma;.
 * To define **mutation** mechanism mutate: &Gamma;&rightarrow;&Gamma;.

In many cases, crossover and mutation are quite simple algorithms to manipulate genes as numeric sequences or bit vectors.

The specific implementation of a genetic algorithm can vary from case to case, but the overall structure is the following:

1. Select an initial population G&subset;&Gamma;
2. Randomly select one of the operations that will be performed at this step: crossover or mutation
3. **Crossover**:
  * Randomly select two genes g<sub>1</sub>, g<sub>2</sub> &in; G
  * Compute crossover g=crossover(g<sub>1</sub>,g<sub>2</sub>)
  * If fit(g)<fit(g<sub>1</sub>) or fit(g)<fit(g<sub>2</sub>) - replace corresponding gene in the population by g.
4. **Mutation** - select random gene g&in;G and replace it by mutate(g)
5. Repeat from step 2, until we get a sufficiently small value of fit, or until the limit on the number of steps is reached.

## _Typical Tasks_

Tasks typically solved by Genetic Algorithms include:

1. Schedule optimization
2. Optimal packing
3. Optimal cutting
4. Speeding up exhaustive search


source: [Dmitry Soshnikov, PhD](https://soshnikov.com/) (2024) "Genetic Algorithm", _AI For Beginners_, https://github.com/microsoft/AI-For-Beginners




## <strong>Usage</strong>
### 1. Import and Prepare Your Map


### 2. _HarrisCorner_ Class
The _HarrisCorner_ class provides a simple interface for detecting corner points in a given 2D map using the Harris corner detection algorithm in OpenCV.
You just need to provide your map(as a NumPy array) to initialize the detector.

```python
import numpy as np
from _MAPS_.<your_map_repo>.<your_map_file> import MAP
from _HarrisCorner.HCD_tools import HarrisCorner  #HCD:Harris Corner Detector

# When you create the HCD instance, simply pass your MAP as the argument:
HCD_instance = HarrisCorner(MAP)
```

The _HarrisCorner_ class provides 6 main methods. described below:

1. **`.gaussianBlur(self, map, ksize, sigX, sigY)`**
  Applies a Gaussian blur to the input map to reduce noise and improve corner detection robustness.

  - ***Parameters:***
    - `ksize`:  The kernel size for the Gaussian filter, given as a tuple `(height, width)`. For example, `(9, 9)` means the filter covers a 9×9 window.
    - `sigX`:  Standard deviation in the X direction. Controls the amount of smoothing horizontally. Default is `0`, which means it is automatically calculated from `ksize`.
    - `sigY`:  Standard deviation in the Y direction. Controls the amount of smoothing vertically. Default is `0`.

  - **Notes:**  
    Similarly to convolution in image processing, a larger kernel size or larger sigma values result in a smoother (more blurred) image.

---

2. **`.non_max_suppression(self, response, nmx_threshold, dilate_size=5)`**  
   Applies Non-Maximum Suppression (NMS) to the Harris corner response map, keeping only the local maxima above a specified threshold. This process removes redundant or closely located corner points, ensuring that only the most prominent corners are detected.

   - ***Parameters:***
     - `response`: The Harris corner response map (2D NumPy array) to be filtered.
     - `nmx_threshold`: The threshold value. Only peaks with a response greater than this value are kept as corners.
     - `dilate_size`: The size of the dilation window used to find local maxima. For example, `dilate_size=5` means that a 5×5 neighborhood is used when checking for peak values.

   - **Notes:**  
     - Increasing `dilate_size` will make NMS more selective, possibly resulting in fewer, more widely spaced corners.
     - Choosing an appropriate `nmx_threshold` helps filter out weak or insignificant corner responses.

---

3. **`.harrisCorner(self, map, block_size=3, ksize=3, k=0.05, threshold=0.01)`**  
   Computes the Harris corner response map using OpenCV's Harris corner detector.  
   Pixels with a response below the threshold (a ratio of the maximum response) are set to zero.

   - **Parameters:**
     - `map`:  
       Input map (2D NumPy array) to process. Should be a grayscale image (values 0~255).
     - `block_size`:  
       The size of the neighborhood considered for corner detection.  
       Larger values result in more regional analysis, smaller values are more local.
     - `ksize`:  
       Aperture parameter of the Sobel derivative used internally.  
       Typical values are 3, 5, or 7.
     - `k`:  
       Harris detector free parameter, usually in the range [0.04, 0.06].
     - `threshold`:  
       The threshold ratio used to filter out weak corner responses.  
       For example, `threshold=0.01` means only pixels with a response at least 1% of the max will be kept.

   - **Notes:**  
     - The output is a response map of the same size as the input map, with strong corners having higher values.
     - Thresholding helps remove weak or noisy detections, retaining only the most prominent corners.

---

4. **`.filter_close_corners(self, points, min_distance=5)`**  
   Filters out corner points that are too close to each other based on Euclidean distance.  
   This helps to remove redundant detections and ensures spatial diversity among detected corners.

   - **Parameters:**
     - `points`:  
       A list of corner point coordinates, typically as (x, y) tuples.
     - `min_distance`:  
       The minimum allowed distance between any two corner points.  
       If two points are closer than this value, only one is retained.

   - **Notes:**  
     - This function processes the list of detected corners and keeps only those that are sufficiently spaced apart.
     - Helps to prevent clusters of corner points in dense regions, improving the quality of the final detection results.

---

5. **`.extract(self, map)`**  
   Extracts the final corner coordinates from a processed map (typically the output of non-maximum suppression).  
   Converts detected pixel positions from (row, column) tuple format to (x, y) coordinate pairs.

   - **Parameters:**
     - `map`:  
       A 2D NumPy array where detected corner points are marked with the value `1` (all other pixels are `0`).

   - **Returns:**  
     - A list of `(x, y)` tuples representing the coordinates of detected corner points,  
       filtered to ensure they are not too close (via `filter_close_corners`).

   - **Notes:**  
     - This function converts indices from NumPy's default (row, column) to (x, y) for consistency with typical plotting and image processing conventions.
     - It is usually used after non-maximum suppression to retrieve the positions of valid corner points.

---

6. **`.run(self, map, block_size=3, ksize=3, k=0.05, dilate_size=5)`**  
   Runs the full Harris corner detection pipeline on the given map, including blurring, corner response, non-maximum suppression, and filtering.

   - **Processing Steps:**
     1. **Apply Gaussian Blur:**  
        Reduces noise in the input map to improve the reliability of corner detection.
     2. **Harris Corner Detection:**  
        Computes the corner response map using specified parameters.
     3. **Non-Maximum Suppression:**  
        Retains only local maxima in the response map that are above a threshold, eliminating redundant or weak corners.
     4. **Extract Corner Coordinates:**  
        Retrieves (x, y) locations of detected corners from the suppressed map.
     5. **Filter Close Corners:**  
        Further removes corner points that are too close to each other for better spatial distribution.

   - **Parameters:**
     - `map`:  
       The input map or grayscale image (2D NumPy array) to process.
     - `block_size`:  
       Neighborhood size for corner detection.
     - `ksize`:  
       Aperture size for the Sobel operator.
     - `k`:  
       Harris detector free parameter.
     - `dilate_size`:  
       Size of the dilation window for non-maximum suppression.

   - **Returns:**  
     - A list of `(x, y)` tuples representing the final detected corner coordinates.

   - **Notes:**  
     - This method provides an easy, single-call interface for full corner detection workflow.
     - All default parameters are chosen for general robustness, but can be tuned as needed.

