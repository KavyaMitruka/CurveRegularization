## ADOBE-GENSOLVE
Curvetopia: Regularizing and Beautifying 2D Curves
Welcome to Curvetopia! This project focuses on identifying, regularizing, and beautifying 2D curves, with an emphasis on transforming irregular hand-drawn shapes into their idealized forms.

## TABLE OF CONTENTS:
Project Overview
Features
Installation
Usage
Technical Details
Future Work
Contributing

## PROJECT OVERVIEW:
This project is designed to regularize various shapes such as lines, circles, ellipses, rectangles, polygons, and star shapes. The project processes input data from CSV files, identifies the shape type, regularizes it, and then visualizes the output as both PNG images and SVG files.

## FEATURES:
Shape Identification: Detects shapes including circles, ellipses, rectangles, polygons, and stars.
Regularization: Smooths and regularizes shapes to ensure symmetry and precision.
Symmetry Detection: Checks for symmetry in the shapes and adjusts accordingly.
Curve Completion: Completes incomplete curves to form closed shapes.
Visualization: Outputs regularized shapes as PNG images and SVG files.

## INSTALLATION:
To set up the project locally, follow these steps:
1.Clone the repository:
  git clone https://github.com/KavyaMitruka/ADOBE-GENSOLVE.git
  cd ADOBE-GENSOLVE
2.Install required dependencies:
  pip install -r requirements.txt
3.Ensure you have the following dependencies:
  numpy
  matplotlib
  svgwrite
  cairosvg
  scipy
  scikit-learn
4.Run the project:
  python ADOBE-GENSOLVE.py
5.Usage
  To use ADOBE-GENSOLVE, place your input CSV file in the root directory, and run the script. The output will be saved as PNG and SVG files.
  Command Line Example:
  python curvetopia.py --input data1.csv --output output.svg

## TECHNICAL DETAILS:
## 1.Isolated_Regularization.ipynb

  1.Importing Necessary Libraries
     import numpy as np: Imports the NumPy library, which provides support for arrays and matrices, along with a collection of mathematical functions to operate on these         data structures.
     import matplotlib.pyplot as plt: Imports Matplotlib's pyplot module, which is used for creating static, animated, and interactive visualizations.
     from scipy.spatial import distance: Imports the distance module from SciPy, which provides various distance metrics and geometric functions.
  2.Reading and Parsing CSV Data
    def read_csv(csv_path):: This function reads a CSV file from the specified path and processes it into a format suitable for further operations.
    np.genfromtxt(csv_path, delimiter=','): Reads the CSV file into a NumPy array, with each row representing a path and columns corresponding to coordinates.
    Processing: The data is grouped by unique path identifiers, creating a list of lists (path_XYs), where each sublist represents a path's coordinates.
  3. Plotting Paths
    def plot(path_XYs):: This function plots the paths using Matplotlib.
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8)): Creates a plot with equal aspect ratio.
    ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2): Plots each path in a different color.
    plt.axis('off'): Hides the axis for cleaner visualization.
  4. Resampling Points
    def determine_resample_spacing(points):: Determines the resampling spacing for the points based on the diagonal of the bounding box.
    def resample_points(points, S):: Resamples the points along a path to make them evenly spaced according to the spacing S.
  5. Feature Extraction
    def extract_features(points):: Extracts geometric features from the points, such as the convex hull, perimeter, area, compactness, and angles between vertices.
  6. Shape Classification
    def classify_shape(features):: Classifies the shape based on the extracted features. It distinguishes between straight lines, rectangles, polygons, star shapes, 
    circles, ellipses, and unknown shapes.
  7. Regularizing Shapes
      Line Regularization:
      def regularize_line(points):: Fits a straight line to the points using linear regression and returns the regularized line.
      Circle Regularization:
      def fit_circle(points):: Fits a circle to the points by minimizing the variance of the radial distances from the center.
      def regularize_circle(points):: Generates a regular circle based on the fitted center and radius.
      Ellipse Regularization:
      def fit_ellipse(points):: Fits an ellipse to the points using least squares.
      def regularize_ellipse(points):: Generates a regular ellipse based on the fitted parameters.
      Rectangle Regularization:
      def regularize_rectangle(points):: Regularizes a set of points into a rectangle by adjusting the vertices to create straight, parallel sides.
      Rounded Rectangle Regularization:
      def regularize_rounded_rectangle(points, radius=10):: Regularizes the points into a rounded rectangle, fitting quarter circles at the corners.
      Polygon Regularization:
      def regularize_polygon(points, num_sides):: Regularizes the points into a polygon with a specified number of sides.
      Star Shape Regularization:
      def regularize_star(points, num_points):: Regularizes a star shape by evenly distributing points around the center.
  8. Symmetry Detection and Regularization
    def detect_symmetry(points):: Detects the axis of symmetry for the points.
    def regularize_for_symmetry(points):: Adjusts the points to make them symmetric around the detected axis.
  9. Curve Completion
    def complete_curve(points):: Completes a curve by fitting a spline to the points and interpolating additional points to create a smooth, continuous curve.
  10. Douglas-Peucker Algorithm
    def douglas_peucker(points, epsilon):: Implements the Douglas-Peucker algorithm to simplify a polyline by reducing the number of points based on a given tolerance            (epsilon).
  11. Shape Identification
    def identify_shape(approx):: Identifies the shape type (e.g., triangle, rectangle) based on the approximate vertices after applying simplification or regularization. 

# Summary:
The code is a comprehensive toolset for processing, regularizing, and analyzing 2D shapes represented by sets of points. It handles various geometric shapes, supports shape classification, symmetry detection, and curve completion, and visualizes the results using Matplotlib. This makes it suitable for applications like image processing, pattern recognition, and computer vision, where accurate shape representation and regularization are essential.

## 2. Fragmented_Regularization.ipynb

  1. Importing Necessary Libraries
    import numpy as np: Imports the NumPy library, which provides support for arrays and matrices, along with a collection of mathematical functions to operate on these         data structures.
    import matplotlib.pyplot as plt: Imports Matplotlib's pyplot module, which is used for creating static, animated, and interactive visualizations.
    import svgwrite: Imports the svgwrite module for creating SVG files, which is used to generate scalable vector graphics.
    import cairosvg: Imports the cairosvg module to convert SVG files into PNG images.
    from scipy.spatial import distance: Imports the distance module from SciPy, which provides various distance metrics and geometric functions.
    from scipy.optimize import minimize: Imports the minimize function from SciPy, which is used for minimizing functions, particularly for fitting shapes like circles.
    from sklearn.cluster import KMeans: Imports the KMeans clustering algorithm from Scikit-learn, used here for clustering points to regularize shapes such as rectangles.
  2. Reading and Parsing CSV Data
    def read_csv(csv_path): This function reads a CSV file from the specified path and processes it into a format suitable for further operations.
    np.genfromtxt(csv_path, delimiter=','): Reads the CSV file into a NumPy array, with each row representing a path and columns corresponding to coordinates.
    Processing: The data is grouped by unique path identifiers, creating a list of lists (path_XYs), where each sublist represents a path's coordinates.
  3. Plotting Paths
    def plot(path_XYs): This function plots the paths using Matplotlib.
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8)): Creates a plot with equal aspect ratio.
    ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2): Plots each path in a different color.
    plt.savefig('output_plot.png'): Saves the plotted figure as a PNG file.
  4.Regularizing Shapes
     Regularizing Straight Lines:
     def regularize_straight_lines(XYs): Regularizes a polyline into a straight line by using the first and last points.
     Regularizing Circles:
     def regularize_circle(XYs):
     calc_radius(center): Calculates the average radius from a given center to all points.
     minimize: Optimizes the center of the circle by minimizing the standard deviation of the radial distances.
     Regularization: Generates a regular circle based on the fitted center and radius.
     Regularizing Ellipses:
     def regularize_ellipse(XYs):
     Eigenvectors and eigenvalues: Computes the covariance matrix and performs eigen decomposition to determine the ellipse's orientation.
     Regularization: Generates a regular ellipse based on the calculated parameters.
     Regularizing Rectangles:
     def regularize_rectangle(XYs):
     KMeans Clustering: Uses KMeans clustering to identify the four corners of a rectangle.
     Ordering: Orders the corners to ensure correct alignment for regularization.
     Regularizing Polygons:
     def regularize_polygon(XYs, sides):
     Center Calculation: Computes the center of the polygon.
     Angle Distribution: Evenly distributes the points around the center based on the specified number of sides.
     Regularization: Generates a regular polygon by positioning points equidistant from the center.
     Regularizing Star Shapes:
     def regularize_star(XYs):
     Center Calculation: Computes the center of the star shape.
     Angle and Radius Calculation: Separately processes inner and outer points to create the star's structure.
     Regularization: Aligns the points to create a symmetric star shape.
  5. Detecting Symmetry
    def detect_symmetry(XYs): Checks if the shape is symmetric by comparing it with its reflection.
  6. Regularizing Curves
    def regularize_curves(path_XYs):
    Shape Classification: Based on the number of points and symmetry detection, classifies and regularizes shapes into straight lines, circles, ellipses, rectangles,            polygons, or star shapes.
  7. Generating SVG and PNG Files
    def polylines2svg(paths_XYs, svg_path):
    SVG Creation: Converts the regularized paths into SVG format using the svgwrite library.
    PNG Conversion: Converts the SVG file into a PNG image using cairosvg.
  8. Main Execution
    if name == "main":
    Reading Data: Loads the polyline data from a CSV file.
    Regularizing Shapes: Regularizes the shapes into standard geometric forms.
    Plotting and Exporting: Visualizes the regularized paths and exports the results as SVG and PNG files.

# Summary:
This code is a comprehensive toolset for processing, regularizing, and analyzing 2D composite shapes represented by sets of points. It handles various geometric shapes, including straight lines, circles, ellipses, rectangles, polygons, and star shapes. The code supports shape classification, symmetry detection, and curve completion, and it visualizes the results using Matplotlib. Additionally, it generates output in both SVG and PNG formats, making it suitable for applications like image processing, pattern recognition, and computer vision, where accurate shape representation and regularization are essential.
  
## FUTURE WORK:
Expand Shape Support: Add more complex shapes such as curved polygons and irregular stars.
Enhance Accuracy: Improve regularization algorithms for better symmetry and precision.

