# cvfind
Project cleanup tool and cpfind alternative for Hugin / panotools with a focus on tiling, chip photograpy, and source images with highly repeating fine patterns.

# Introduction
cvfind is intended to be a robust alternative to Hugin's cpfind for tile sets containing repeated patterns or nearly identical but distinct image tiles.  Its command line options are compatible with Hugin's calling conventions and can be used directly from Hugin as a generic control point detector.  Additional options can tune cvfind to significantly improve control point detection for tiled projects with disciplined shooting patterns.

![Example of repeated patterns](https://github.com/Bob-O-Rama/cvfind/assets/28986153/4b0a6fb3-dfce-4b8b-9e34-c4772f80729c)

( Most of the problematic image tiles in this README are available through the Image Tile Library: https://github.com/Bob-O-Rama/cvfind/blob/main/IMAGE-TILE-LIBRARY.md  If you have a challenging set of image tiles you want to add to the library, let "us" know. )

# Discinctions from Hugin cpfind
cvfind acts as a wrapper for many of opencv's detector and matcher algorithms, including methods that are more robust for matching blurry subject matter or containing line like features ignored by cpfind's SIFT.  In short, it can find many more high quality control points and offers a variety of methods to identify them.

![Example of control points detected by cvfind](https://github.com/Bob-O-Rama/cvfind/assets/28986153/ecd3eb7a-e9c7-4259-bb8b-3b03aa2b466f)

Detection of control points between blurry images is also improved.  ORB does a fairly good job finding control points between equally blurry image pairs.  ( See example below. )  When the overlapping areas differ significantly in terms of blur, AKAZE can be useful.

![Blurry Image A - Control Points](https://github.com/Bob-O-Rama/cvfind/assets/28986153/a7e681be-88c2-48f4-834a-940438731154)

![Blurry Image B - Control Points](https://github.com/Bob-O-Rama/cvfind/assets/28986153/f23506c8-f5fb-483c-8fb8-76f1d941daa8)

cvfind applies some guidance to the RANSAC process to limit the search space for identifying the overlap.  This allows cvfind to find a strong homography option describing the overlap where as cpfind would return none, few, or incorrect control points owing to multiple solutions blocking it from finding a consensus.

cvfind ensures that every image's overlap with its neighborhood of pairs are self consistent.   A bad pair may be properly aligned with one image but not with another.  By testing groups of nearby images, it is possible to root out the inconsistent pair and remove it.   This process is very robust and eliminates all "bad" pairs in most cases.

Finally, cvfind can use this process to identify "missing" pairs, and perform more intensive analysis to recoup them.  In most cases cvfind can identify about 85% of all possible pairs, and rejects all bad pairs.

![cvfind produced image pairing analysis](https://github.com/Bob-O-Rama/cvfind/assets/28986153/98206064-7990-4dfd-977a-fa9876882cd2)

# Installing from OBS
cvfind is available as pre-build RPMs via the Open Build Service, which is an automated cross-distribution packaging service that supports many major Linux distributions.  Currently cvfind is available for most SuSE and Fedora variants.  Debian and Ubuntu use an entirely different packaging system, but these too are supported by OBS once a build recipie is concocted.  The major obsticle so far is that many distributions have no release of OpenCV or are stuck at OpenCV 3.x.

Visit the cvfind page on OBS: https://build.opensuse.org/package/show/home:bob-o-rama/cvfind
Check if your distribution is building properly ( scroll down on the right ).  Then click the Download link.  This will try to do a one-click install using your package manager, that usually entails trusting the OBS repo, then fetching and installing the package.   Once that happens, usually your system will keep it up to date like any other software.

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/97014703-055d-4bd6-b439-40708a9978fd)
( PS: I'm just saying that this feels super creepy. )

# Building from Source - "Not as horrible as you may think!"
cvfind has no major build dependancies other than OpenCV 4.x.x opencv-devel and c++14.   cvfind uses a simple Makefile, and specifying the optional WITH_CONTRIB=TRUE make flag will assume opencv was made with the "non-free" opencv_contrib materials.  That is entirely optional, you can build cvfind against stock opencv.  The real impact is that cvfind will not offer the SURF, LINE, and LSD ( line segment detector ).  These are not critical, as ORB, AKAZE, SIFT, .... and their corresponding matchers will still work.  ( The source tarball for cvfind includes a "configure" script but it does nothing other than make OBS happy. )

To obatin the source tarball, obtain any of the source tarball versions for any distribution, they are the same.   The archive application on your system will let you treat the RPM liek a zip file, inside is the cvfind.tar.gz file used in the next sections.

Visit: https://software.opensuse.org//download.html?project=home%3Abob-o-rama&package=cvfind

And select Grab Binary Package Directly, pick any distribution, and download the cvfind src rpm

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/d21eb973-cd34-44e7-b51b-6fe9cf68416e)

Your archive manager will see the RPM like a zip file...

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/7d0a8b42-26a0-47c2-8f97-744f55cfa721)

You can then extract the cvfind.tar.gz used in the following sections.

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/70c8cefb-5076-45a5-a3ab-8030dc1301eb)

# Reference Build Environment #1 - Build with pre-build OpenCv

cvfind was developed on openSuSE Tumbleweed.  You can build cvfind from source by obtaining the source tarball ( see above ) and using the provided Makefile.  This brief recipe can be adapted for most any *nix.
```
#!/bin/bash
# Create a convenient writiable place to do the build 
mkdir ~/build
mkdir ~/build/cvfind
cd ~/build/cvfind
# Obtain cvfind tarball, it dumps files in the current directory
tar xvf cvfind.tar.gz
# See the "BuildRequires:" items in cvfind.spec for a list of dependencies.
# For managed distributions with older packages, you may not have
# opencv4 available.  cvfind required 4.x, so you may need to build
# both if your package manager cannot get you 4.x.   Also, the .spec
# file includes some other requirements 
# Use your package manager to fetch prerequisites:
sudo zypper in opencv opencv-devel cmake gcc gcc-c++ zlib-devel zlib unzip
# Once the dependancies are installed... make it:
make check
# This will compile, link, and then execute self tests.
# If the self test completes, you are done.  You can use cvfind in its
# current location /build/cvfind/cvfind or use
sudo make install
# to copy it to /usr/local/bin/cvfind
# easy peasy lemon squeezy
```

# Reference Build Environment #2 - Including OpenCV With opencv_contrib

cvfind was developed on openSuSE Tumbleweed.  This can be adapted for most any *nix.  The reason you would want to build opencv from source is to ensure you have the non-free contrib modules to utilize SURF, LINE, and LSD dettectors.    The script used to obtain and build opencv as well as build cvfind:

```#!/bin/bash
# Make a writable build directory and opencv directory
mkdir ~/build
cd ~/build
mkdir opencv
cd opencv
# Install a minimum tool set for the build
zypper in git cmake gcc gcc-c++ zlib-devel zlib
# obtain main opencv source 
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.7.0
cd ..
# opencv_extra contains reference data sets for regression and unit testing
git clone https://github.com/opencv/opencv_extra/
cd opencv_extra
git checkout 4.7.0
cd ..
# opencv_contrib contains additional modules for 2D feature analysis
git clone https://github.com/opencv/opencv_contrib/
cd opencv_contrib
git checkout 4.7.0
cd ..
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_ENABLE_NONFREE=ON -D BUILD_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON BUILD_PERF_TESTS=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D WITH_GTK=ON ..
make -j9
sudo make install
mkdir ~/build/opencv/cvfind
# ( Place cvfind.cpp and Makefile into /build/opencv/cvfind )
cd ~/build/opencv/cvfind
make WITH_CONTRIB=TRUE
# Optional: Run the self test to ensure cvfind can bind to opencv properly
./cvfind --test
# Optional: Display help screen
./cvfind -h
```
You will note that ./cvfind --test will show that the contrib materials are available and that additional detector types are available.  These additional detector types are hidden throughout cvfind ( help screens, etc. ) when not available:

```
./cvfind --test
 INFO: OpenCV version : 4.7.0
 TEST: Major version : 4 > 3, Passed.
 INFO: Minor version : 7
 INFO: Subminor version : 0
 INFO: Built With Contrib Materials.
 TEST: Binding AKAZE ... Passed.
 TEST: Binding ORB ... Passed.
 TEST: Binding BRISK ... Passed.
 TEST: Binding SIFT ... Passed.
 TEST: Binding SimpleBlobDetector ... Passed.
 TEST: Binding Contrib SURF ... Passed.
 TEST: Binding Contrib BinaryDescriptor ... Passed.
 TEST: Binding Contrib LSDDetector ... Passed.
 TEST: All Tests Passed
 INFO: Self Test Passed, returning error level 0.

```

# Hugin integration
Hugin can use cvfind as a control point detector.   Start Hugin... **File-->Preferences-->Control Point Detectors-->Add**   Fill in the detector details, for example:

![Example Hugin Detector Settings Dialog ](https://github.com/Bob-O-Rama/cvfind/assets/28986153/7f01d3fd-7c9e-428b-b281-df5a75074dff)

Now you can use cvfind as part of your Hugin workflow.  See USAGE.md for details on making this efficient by specifying shooting pattern hints.   cvfind can also be used for project by invoking from the command line.

# Large Projects
Large projects with > 2000 images have been tested.  It takes a lot of memory owing to pre-loading images into RAM. Initializing data structures may take a minute as that process is single threaded.  When shooting pattern hints are specified, the computational effort scales roughly linearly with the number of images.  When hints are effective, as few as N * 4 image matching jobs are performed. If no hints are provided this baloons to N * N / 2.   For 2000 images that is 8000 vs 2,000,000 or a 250x advantage.  Even if the shooting pattern is somewhat chaotic, increasing "--rowsizetolerance" still reduces the effort significantly without resorting the --linearsearchlen which itself only yields SQRT( N ) fold advantage, 44x for the 2000 image set.  During image processing cvfind will periodically output a progress status with an estimated completion time.

```tail -f `ls *.log` | grep -i progress```

# Note on Regression Testing
Testing for these types of projects can be difficult.  Given the same input images and input project file, cvfind should produce identical control point lists in the output pto from run to run.  Doing a diff between the control point lists should be sufficient to detect any significant changes in behaviour.   Common detectors and matchers are deterministic, the "random" culling of excess control points is also deterministic.  One simple way to compare the control point entries between runs is as follows:
```
cat run1.pto run2.pto run2.pto | grep -v '#' | grep 'c n' | sort | uniq -c | grep -v '      3 '
```
The output will list any non-matching control points and which run the control point came from, for example:
```
      2 c n101 N116 x2605.307861 y232.699982 X865.593628 Y1443.369629 t0
      2 c n101 N116 x2610.899658 y114.185249 X872.609070 Y1322.316650 t0
      2 c n101 N117 x1794.734253 y675.168152 X64.755440 Y659.549316 t0
      1 c n101 N117 x1797.955322 y935.134644 X67.925674 Y917.510193 t0
      2 c n101 N117 x1826.799927 y586.032471 X96.621895 Y570.753418 t0
      2 c n101 N117 x1827.290283 y615.585876 X97.085670 Y600.133240 t0
      2 c n101 N117 x1828.572998 y645.358276 X98.068466 Y629.617249 t0
      1 c n101 N117 x1832.078857 y453.598694 X102.543747 Y440.562622 t0
      2 c n101 N117 x1832.128540 y422.613007 X102.741425 Y409.105591 t0
      2 c n101 N117 x1839.401855 y726.145935 X108.538315 Y710.007751 t0
```
Ideally this list would be empty, demonstrating the control point lists are the same between both PTO files.   The snipped above adds run2.pto twice.  If a control point is shared, it will be present 3 times, if on run2 only, 2 times.   Its an easy way to compare lines between unordered files.   A test case would compare the before and after results, it should return no lines so a simple test using wc -l in a script can confirm cvfind's behaviour was not altered.   A regression test suite is being developed.

# Test Data
The Image Tile Library can provide a source of test immagery that is comparable between testers.   If you have a large stitch image, it can be dissected into tiles for testing.  This is useful to determine the effects of things like image overlap, the aspect ratio of tiles, etc.   Image Magick can be used to perform these sorts of dissections, though it stores the intermediate output in memory - and so had a huge memory footprint.   Here is trivial script to create a 600 tile ( 20 x 30 ) series from an existing source TIFF:
```
#!/bin/bash
convert $1 -verbose -background green -virtual-pixel background \
   -set option:distort:viewport "%[fx:ceil(w/20)*20]x%[fx:ceil(h/30)*30]" \
   -distort SRT 0 +repage -crop "20x30@+400+300" +repage "$2_%04d.jpg"
```
e.g. **./dissect.sh some_huge.tiff tiles** to create a 20 x 30 grid of overlapping tiles, tiles_0000.jpg thru tiles_0599.jpg.  The overlap is specified as 400 horizontally, 300 vertically in this example.  

# How To Help
Check out the Issues tab: https://github.com/Bob-O-Rama/cvfind/issues and see if there is an Issue you can help with. 

# Known Issues
- ORB and AKAZE detectors have been tested the most, others are in various stages of "working" while others just might not be very effective.
- cvfind pre-loads all images into memory and fairly efficiently uses swap, 32GB of memory ( between RAM and swap ) should be sufficient for a 250 16MP tile project.
- Out of memory conditions can be reduced by reducing the number of threads for image processing.
- Processing scales fairly linearly with more cores on systems that support multiple memory channels, such as server class equipment.
- By default cvfind can produce many more control points than cpfind.  For stitching images this seems not to cause any significant issues and projects with 250,000 control points optimize and stitch without issue.  However there appears to be a bug in Hugin UI that cause it to crash when re-painting the numeric labels in the pairwise control point view.  There is generally no need to edit control points, but be aware that Hugin may unexpectedly exit if you do.   
- Some longer processing steps may not adequately inform the user of their progress.
- cvfind is 10-20 times more efficient when provided with shooting patterns hints.  
- Stiching large numbers of images with highly regular overlap may result in triggering bugs in *enblend,* that results in grey or black shaded areas in place of a particular image tile.   This is not a cvfind issue, but owing to the fact that cvfind makes larger projects more feasible, you may see this.   Adding some or all of the following options to the enblend command line usually resolve the issue:  **--preassemble --primary-seam-generator=nft --blend-colorspace=identity** with **--primary-seam-generator=nft** generally being the most useful.
