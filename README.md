<h1 align="center">  Multi-Image Phasor Analysis for FLIM Data</h1>

### Description
To analyze FLIM data in phasor space without relying on representative images, we
use multi-image image analysis.The provided code can take multiple three 
dimensional FLIM tiff files, calibrate them using a tiff image from a 
calibration standard,represent the FLIM data in phasor space, and then use a Gaussian Mixture Model (GMM) to find
multiple components within the combined phasor data set. All of the analysis is controlled
through a GUI (GUI2.py). After determining the cluster positions using GMM, the GUI can then 
be used to apply the model to the data, resulting in color-by-cluster images, phasor plots
for each image with the clusters marked, and amalgamated phasor plots with data from all
inputted images.
