# cvfind
Project cleanup and cpfind alternative for Hugin

# Introduction
cvfind is intended to be a robust alternative to Hugin's cpfind for tile sets containing repeated patterns or nearly identical but distinct image tiles.  Its command line options are compatible with Hugin's calling conventions and can be used directly from Hugin as a generic control point detector.  Additional options can tune cvfind to significantly improve control point detection for tiled projects with disciplined shooting patterns.

![Example of repeated patterns](https://github.com/Bob-O-Rama/cvfind/assets/28986153/4b0a6fb3-dfce-4b8b-9e34-c4772f80729c)

# Discinctions from Hugin cpfind
cvfind acts as a wrapper for many of opencv's detector and matcher algorithms, including methods that are more robust for matching blurry subject matter or containing line like features ignored by cpfind's SIFT.  In short, it can find many more high quality control points and offers a variety of methods to identify them.

![Example of control points detected by cvfind](https://github.com/Bob-O-Rama/cvfind/assets/28986153/ecd3eb7a-e9c7-4259-bb8b-3b03aa2b466f)

cvfind applies some guidance to the RANSAC process to limit the search space for identifying the overlap.  This allows cvfind to find a strong homography option describing the overlap where as cpfind would return none, few, or incorrect control points owing to multiple solutions blocking it from finding a consensus.

cvfind ensures that every image's overlap with its neighborhood of pairs are self consistent.   A bad pair may be properly aligned with one image but not with another.  By testing groups of nearby images, it is possible to root out the inconsistent pair and remove it.   This process is very robust and eliminates all "bad" pairs in most cases.

Finally, cvfind can use this process to identify "missing" pairs, and perform more intensive analysis to recoup them.  In most cases cvfind can identify about 85% of all possible pairs, and rejects all bad pairs.

![cvfind produced image pairing analysis](https://github.com/Bob-O-Rama/cvfind/assets/28986153/98206064-7990-4dfd-977a-fa9876882cd2)

# Building
cvfind has no major dependancies other than opencv built with extra and contrib options to ensure optional feature2d components are available.  Some distributions do not include these optional feature in their factory opencv libraries.  On such platforms it may be necessary to build opencv from source or find a packager who has done so.   An example Makefile is provided, it may need to be modified to reflect the actual locations of opencv headers and libraries on your system.

# Reference Build Environment

cvfind was developed on openSuSE Tumbleweed.  The script used to obtain and build opencv as well as build cvfind:

```#!/bin/bash
# Make build directory and opencv directory
mkdir /build
cd /build
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
mkdir /build/opencv/cvfind
# ( Place cvfind.cpp and Makefile into /build/opencv/cvfind )
cd /build/opencv/cvfind
make
./cvfind -h
```
# Hugin integration
Hugin can use cvfind as a control point detector.   Start Hugin... **File-->Preferences-->Control Point Detectors-->Add**   Fill in the detector details, for example:

![Example Hugin Detector Settings Dialog ](https://github.com/Bob-O-Rama/cvfind/assets/28986153/7f01d3fd-7c9e-428b-b281-df5a75074dff)

Now you can use cvfind as part of your Hugin workflow.  See USAGE.md for details on making this efficient by specifying shooting pattern hints.   cvfind can also be used for project by invoking from the command line.

# Large Projects
Large projects with > 2000 images have been tested.  It takes a lot of memory owing to pre-loading images into RAM. Initializing data structures may take a minute as that process is single threaded.  When shooting pattern hints are specified, the computational effort scales roughly linearly with the number of images.  When hints are effective, as few as N * 4 image matching jobs are performed. If no hints are provided this baloons to N * N / 2.   For 2000 images that is 8000 vs 2,000,000 or a 250x advantage.  Even if the shooting pattern is somewhat chaotic, increasing "--rowsizetolerance" still reduces the effort significantly without resorting the --linearsearchlen which itself only yields SQRT( N ) fold advantage, 44x for the 2000 image set.

# Known Issues
- ORB and AKAZE detectors have been tested the most, others are in various stages of "working" while others just might not be very effective.
- cvfind pre-loads all images into memory and fairly efficiently uses swap, 32GB of memory ( between RAM and swap ) should be sufficient for a 250 16MP tile project.
- Out of memory conditions can be reduced by reducing the number of threads for image processing.
- Processing scales fairly linearly with more cores on systems that support multiple memory channels, such as server class equipment.
- Some longer processing steps may not adequately inform the user of their progress.
- cvfind is 10-20 times more efficient when provided with shooting patterns hints.  
- If cvfind immediately crashes during image processing it may be trying to use a different copy of opencv libraries, use **ldd ./cvfind** to confirm it is linking with the correct version of opencv libraries.
