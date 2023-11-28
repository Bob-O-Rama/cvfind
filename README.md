# cvfind
Project cleanup and cpfind alternative for Hugin
# Introduction
cvfind is intended to be a robust alternative to Hugin's cpfind for tile sets containing repeated patterns or nearly identical but distinct image tiles.  Its command line options are compatible with Hugin's calling convenmtions and can be used directly from Hugin as a generic control point detector.  Additional options can tune cvfind to significantly improve control point detection for tiled projects with disciplined shooting patterns.

cvfind acts as a wrapper for many of opencv's detector and matcher algorithms, including methods that are more robust for matching blurry subject matter or containing line like features ignored by cpfind's SIFT.  In short, it can find many more high quality control points and offers a variety of methods to identify them.

cvfind applies some guidance to the RANSAC process to limit the search space for identifying the overlap.  This allows cvfind to find a strong homography option describing the overlap where as cpfind would return none, few, or incorrect control points owing to multiple solutions blocking it from finding a consensus.

cvfind ensures that every image's overlap with its neighborhood of pairs are self consistent.   A bad pair may be properly aligned with one image but not with another.  By testing groups of nearby images, it is possible to root out the inconsistent pair and remove it.   This process is very robust and eliminates all "bad" pairs in most cases.

Finally, cvfind can use this process to identify "missing" pairs, and perform more intensive analysis to recoup them.  In most cases cvfind can identify about 85% of all possible pairs, and rejects all bad pairs.
# Building
cvfind has no major dependancies other than opencv built with extra and contrib options to ensure optional feature2d components are available.  Some distributions do not include these optional feature in their factory opencv libraries.  On such platforms it may be necessary to build opencv from source or find a packager who has done so.   An example Makefile is provided, it may need to be modified to reflect the actual locations of opencv headers and libraries on your system.
# Known Issues
ORB and AKAZE detectors have been tested the most, others are in various stages of "working" while others just might not be very effective.  cvfind pre-loads all images into memory and fairly efficiently uses swap.  32GB of memory ( between RAM and swap ) should be sufficient for a 250 16MP tile project.    Out of memory conditions can be reduced by reducing the number of threads for image processing.  Processing scales fairly linearly with more cores on systems that support multiple memory channels, such as server class equipment.  Some longer processing steps may not adequately inform the user of their status.  cvfind is 10-20 times more efficient when provided with shooting patterns hints. 
