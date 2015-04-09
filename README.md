# Computer Vision MosaicPi
Code for a project in computer vision.

A very short project description: Stitch images for a mosaic panorama using servos and a raspberry pi.

### Progress

#### 2015-04-09
Now inliers and corresponding points are included in the plots. 
<p align="center">
<img src="images/ransac_books.jpg" height="500" alt="Screenshot"/>
</p>

#### 2015-04-08
First implementation of RANSAC. 
<p align="center">
<img src="images/ransac_test_inliers.png" height="500" alt="Screenshot"/>
</p>

#### 2015-04-07
Got the SURF working. Rather good matching, some outliers.
<p align="center">
<img src="images/matches_surf.png" height="500" alt="Screenshot"/>
</p>
Images used for matching are privately owned.

##### 2015-04-04
Extracted keypoints using the 'FAST' algorithm, implemented in opencv
<p align="center">
<img src="images/test010.jpg" height="500" alt="Screenshot"/>
</p>

##### 2015-03-27
Physical model completed. The downside in using servos is that it's hard to control position.

<p align="center">
<img src="images/model.jpg" height="500" alt="Screenshot"/>
</p>
