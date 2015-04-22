# Computer Vision MosaicPi
Code for a project in computer vision.

A very short project description: Stitch images for a mosaic panorama using servos and a raspberry pi.

### Progress

#### 2015-04-22
Nice results after only using translation fron the Ransac algorithm. 
<p align="center">
<img src="images/ny_city_stitch.jpg" height="250" alt="Screenshot"/>
</p>
"<a href="http://commons.wikimedia.org/wiki/File:NYC_Pan3.jpg#/media/File:NYC_Pan3.jpg">NYC Pan3</a>" by <a href="//commons.wikimedia.org/wiki/User:Jnn13" title="User:Jnn13">Jnn13</a> - <span class="int-own-work" lang="en">Own work</span>. Licensed under <a href="http://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a> via <a href="//commons.wikimedia.org/wiki/">Wikimedia Commons</a>.
"<a href="http://commons.wikimedia.org/wiki/File:NYC_Pan4.jpg#/media/File:NYC_Pan4.jpg">NYC Pan4</a>" by <a href="//commons.wikimedia.org/wiki/User:Jnn13" title="User:Jnn13">Jnn13</a> - <span class="int-own-work" lang="en">Own work</span>. Licensed under <a href="http://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a> via <a href="//commons.wikimedia.org/wiki/">Wikimedia Commons</a>.
"<a href="http://commons.wikimedia.org/wiki/File:NYC_Pan5.jpg#/media/File:NYC_Pan5.jpg">NYC Pan5</a>" by <a href="//commons.wikimedia.org/wiki/User:Jnn13" title="User:Jnn13">Jnn13</a> - <span class="int-own-work" lang="en">Own work</span>. Licensed under <a href="http://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a> via <a href="//commons.wikimedia.org/wiki/">Wikimedia Commons</a>.


#### 2015-04-19
Test stitching after seam was found.
<p align="center">
<img src="images/stitch_after_seam_found.jpg" height="450" alt="Screenshot"/>
</p>

First successful seam found. Key idea is to remove certain areas that cross the seam.
<p align="center">
<img src="images/nice_seam.png" height="450" alt="Screenshot"/>
</p>

#### 2015-04-13
Better stitching.
<p align="center">
<img src="images/result_books.jpg" height="450" alt="Screenshot"/>
</p>

#### 2015-04-11
Managed to stitch two images. Aligns good, but blending is not ok.
<p align="center">
<img src="images/first_align.png" height="450" alt="Screenshot"/>
</p>

#### 2015-04-09
Now inliers and corresponding points are included in the plots. 
<p align="center">
<img src="images/ransac_books.jpg" height="450" alt="Screenshot"/>
</p>

#### 2015-04-08
First implementation of RANSAC. 
<p align="center">
<img src="images/ransac_test_inliers.png" height="500" alt="Screenshot"/>
</p>

#### 2015-04-07
Got the SURF working. Rather good matching, some outliers.
<p align="center">
<img src="images/matches_surf.png" height="450" alt="Screenshot"/>
</p>
Images used for matching are privately owned.

##### 2015-04-04
Extracted keypoints using the 'FAST' algorithm, implemented in opencv
<p align="center">
<img src="images/test010.jpg" height="450" alt="Screenshot"/>
</p>

##### 2015-03-27
Physical model completed. The downside in using servos is that it's hard to control position.

<p align="center">
<img src="images/model.jpg" height="450" alt="Screenshot"/>
</p>
