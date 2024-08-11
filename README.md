# CURVETOPIA
  "Regularizing and Beautifying 2D Curves"
  
  Welcome to Curvetopia!
  
  This project focuses on identifying, regularizing, and beautifying 2D curves, with an emphasis on transforming irregular hand-drawn shapes into their idealized forms.

## TABLE OF CONTENTS:
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Technical Details](#technical-details)
- [Summary](#summary)
- [Future Work](#future-work)

## PROJECT OVERVIEW:
This project is designed to regularize various shapes such as lines, circles, ellipses, rectangles, polygons, and star shapes. The project processes input data from CSV files, identifies the shape type, regularizes it, and then visualizes the output as both PNG images and SVG files.

## FEATURES:
- Shape Identification: Detects shapes including circles, ellipses, rectangles, polygons, and stars.
- Regularization: Smooths and regularizes shapes to ensure symmetry and precision.
- Symmetry Detection: Checks for symmetry in the shapes and adjusts accordingly.
- Curve Completion: Completes incomplete curves to form closed shapes.
- Visualization: Outputs regularized shapes as PNG images and SVG files.

## INSTALLATION:
To set up the project locally, follow these steps:
1. Clone the repository:
  ```
  git clone https://github.com/KavyaMitruka/ADOBE-GENSOLVE.git
  cd ADOBE-GENSOLVE
  ```
2. Install required dependencies:
  ```
  pip install -r requirements.txt
   ```
3. Ensure you have the following dependencies:
```
pip install numpy matplotlib svgwrite cairosvg scipy scikit-learn os
```
4. Run the project files.
  
5. Usage: To use ADOBE-GENSOLVE, place your input CSV file in the root directory, and run the scripts. The output will be saved as CSV file in `output-csv-files` folder and in PNG and SVG files in `output-images` folder
   
## TECHNICAL DETAILS:

  1. Importing Necessary Libraries
```
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
from scipy.spatial import distance
from scipy.optimize import minimize
from sklearn.cluster import KMeans
```
  3. Reading and Parsing CSV Data
      - `def read_csv(csv_path)`:: This function reads a CSV file from the specified path and processes it into a format suitable for further operations.
      - `np.genfromtxt(csv_path, delimiter=',')`: Reads the CSV file into a NumPy array, with each row representing a path and columns corresponding to coordinates.
      - Processing: The data is grouped by unique path identifiers, creating a list of lists (path_XYs), where each sublist represents a path's coordinates.
  4. Plotting Paths
      - `def plot(path_XYs)`:: This function plots the paths using Matplotlib.

  5. Resampling Points
      - `def determine_resample_spacing(points)`:: Determines the resampling spacing for the points based on the diagonal of the bounding box.
      - `def resample_points(points, S)`:: Resamples the points along a path to make them evenly spaced according to the spacing S.

  6. Feature Extraction
      - `def identify_shape(points)`:: Extracts geometric features from the points, such as the convex hull, perimeter, area, compactness, and angles between vertices.

  7. Shape Classification
      - `def classify_shape(features)`:: Classifies the shape based on the extracted features. It distinguishes between straight lines, rectangles, polygons, star shapes, 
    circles, ellipses, and unknown shapes.

  8. Regularizing Shapes
      - Line Regularization:
        - `def regularize_line(points)`:: Fits a straight line to the points using linear regression and returns the regularized line.
      - Circle Regularization:
        - `def fit_circle(points)`:: Fits a circle to the points by minimizing the variance of the radial distances from the center.
        - `def regularize_circle(points)`:: Generates a regular circle based on the fitted center and radius.
      - Ellipse Regularization:
        - `def fit_ellipse(points)`:: Fits an ellipse to the points using least squares.
        - `def regularize_ellipse(points)`:: Generates a regular ellipse based on the fitted parameters.
      - Rectangle Regularization:
        - `def regularize_rectangle(points)`:: Regularizes a set of points into a rectangle by adjusting the vertices to create straight, parallel sides.
      - Rounded Rectangle Regularization:
        - `def regularize_rounded_rectangle(points, radius=10)`:: Regularizes the points into a rounded rectangle, fitting quarter circles at the corners.
      - Polygon Regularization:
        - `def regularize_polygon(points, num_sides)`:: Regularizes the points into a polygon with a specified number of sides.
      - Star Shape Regularization:
        - `def regularize_star(points, num_points)`:: Regularizes a star shape by evenly distributing points around the center.
  9. Symmetry Detection and Regularization
      - `def detect_symmetry(points)`:: Detects the axis of symmetry for the points.
      - `def regularize_for_symmetry(points)`:: Adjusts the points to make them symmetric around the detected axis.
  10. Curve Completion
      - `def complete_curve(points)`:: Completes a curve by fitting a spline to the points and interpolating additional points to create a smooth, continuous curve.
  11. Douglas-Peucker Algorithm
      - `def douglas_peucker(points, epsilon)`:: Implements the Douglas-Peucker algorithm to simplify a polyline by reducing the number of points based on a given tolerance            (epsilon).
  12. Shape Identification
      - `def identify_shape(approx)`:: Identifies the shape type (e.g., triangle, rectangle) based on the approximate vertices after applying simplification or regularization.
  13. Generating SVG and PNG Files
    - `def polylines2svg(paths_XYs, svg_path)`::
      -  SVG Creation: Converts the regularized paths into SVG format using the svgwrite library.
      -  PNG Conversion: Converts the SVG file into a PNG image using cairosvg.
  14. Generaing CSV files
      - `def write_csv(path_XYs, output_csv_path)`:: Stores the shape polyline data in csv files in output_csv_folder.

## Summary:
This code is a comprehensive toolset for processing, regularizing, and analyzing 2D composite shapes represented by sets of points. It handles various geometric shapes, including straight lines, circles, ellipses, rectangles, polygons, and star shapes. 

The code supports shape classification, symmetry detection, and curve completion, and it visualizes the results using Matplotlib. Additionally, it generates output in both SVG and PNG formats, making it suitable for applications like image processing, pattern recognition, and computer vision, where accurate shape representation and regularization are essential.
  
## FUTURE WORK:
  - Expand Shape Support: Add more complex shapes such as curved polygons and irregular stars.
  - Enhance Accuracy: Improve regularization algorithms for better symmetry and precision.
  - Enhance Shape Completion: Devise Better Algorithms for contour completion of occluded shapes.

## Contributors:
- [Kavya Mitruka](https://github.com/KavyaMitruka/)
- [Deepanshu Singh](https://github.com/singh-deep-anshu/)

