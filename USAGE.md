# Usage:
`./cvfind [-d] [-h] [--option ...] [ -o output.pto] input.pto`

# Examples:

`./cvfind --detect pto --pattern rltb --rowsize 7 --overlap 30 --hugin -o output.pto input.pto`

Example usage of cvfind for project cleanup when some shooting pattern info, in this case overlap, pattern, and row size are known.  This instructs cvfind to import and analyze existing control points from input.pto, save the cleaned up project to output.pto and overwrite it if it already exists.  cvfind performs loop homography filtering by default.

`./cvfind --detect orb --pattern tblr --rowsize 16 --overlap 30 --hugin -o output.pto input.pto`

Example usage for instructing cvfind to read input.pto, detect control points for all image pairs satisfying the shooting pattern hints, merge those control points into the output.pto and overwrite it if it already exists.  This is appropriate also for when you have a project shell without any control points.   

`./cvfind --detect orb --filter original --pattern lrbt --rowsize 11 --overlap 25 --hugin -o output.pto input.pto`

Similar, but instructs cvfind to comment out all previously existing control points. 

`./cvfind --rowsize 7 --rowsizetolerance 2 --nofilter all --filter distance,original --linearmatchlen 14 --nochippy --nodetect all --detect orb,line,blob --cpperpairmin 25 --detect pto --log --alignlog --verbose --pattern rltb --looprepair --noimage 70-80 --savediag neighbors -noimage 20-30,90-100 --overlap 33 -o PMax_deluxe.pto PMax.pto`

An example of going a little overboard with the options.  The takeaway is that when using the same option multiple times, as with the --detect option, the results are additive.  For range valued options, like --noimage, the effects are also additive.  Where single valued options are specified multiple times, the last one take precedence.

# Hugin Integration

In Hugin --> Settings --> Control Point Detectors add a new detector and specify the location of the cpfind binary.  Then specify the command line options template, which is a combination of cvfind options and Hugin macros representing the input and output project files.   Hugin creates a temporary project for both input and output, so access to existing control points or project file may not be possible at run time.

A basic parameter string might be `--hugin --detect orb -o %o %s` which specifies no shooting pattern.  This is will be very inefficient, but will work.  If a shooting pattern is known, create a separate detector config with those shooting pattern hints and any other custom parameters.  Future versions will include a means to store custom options in the project file or with the source images so that a more generic integration can be used.

Alternatively, Hugin can be used to create an initial project file, with the source images specifified, and the control points can be added by cvfind by generating a new project from the command line.

# Caveats, Advice And Limitations

- cvfind can handle up to 10,000 images in a project, however all image files are loaded into RAM, and pairwise data structures ( millions! ) are also resident.  Projects with 250 16MP images can be handled with 32GB of physical memory or a reasonable mix of physical and swap memory.  Its a memory pig.
- the code is fairly stable and is not expected to crash, any cases resulting in a crash should be reported, preferably with a backtrace indicating cvfind source
- image processing occurs in a multi-threaded pool of lambda functions.  It is possible that memory requirements for certain combinations of detectors may temporarily cause an out of memory condition.  If OOM occurs, try reducing the number of concurrent threads by half or temporarily add a swap file.  Since the bulk of the memory usage is image data, it is generally very swappable.
- there are a lot of detectors, most don't work as well as ORB, SIFT, and AKAZE.  Try ORB first, then add --looprepair to engage AKAZE for bad pairs, or use AKAZE if compute time is not an issue.
- Specifying shooting pattern hints is key to efficiency, as it reduces most of the work by sidestepping impossible pairs, and that leads to less false positives to remove in later steps.  At a minimum use the **--linearmatchlen** or **--pattern** and **--rowsize** options.
- The **--savediag neighbors** option is very useful if you still have spurious pairs or cannot identify an easy cause for optimization failure.   Its also a quick tool for inspecting pairs for all images by paging through them in an image viewer.

# Options
  **-d**                   Enables startrup debugging.  Must be first if used.

Enables the debugging flag during startup for troubleshooting command line parsing and other issues. 
  
  **-h**                   Display (this) help page
  
  **-o** *output.pto*        Saves modified copy of .pto to specified file

This preserves compatability with cpfind.  If no output file is specified, cvfind performs all the work, but does not produce an edited PTO.  PTO files specirfied with -o will not be overwritten and cvfind throws an error and terminates.  This is to protect prior work which can require significant computational effort.  the --hugin option may be used to allow overwriting of existing output files.

  **--benchmarking**       Enable reporting of benchmarking info

Benchmarking is currently in a state of flux.  Avoid.  
  
  **--exitdelay N**        Sleep for N seconds prior to exiting.  Currently: 0

This option will hold open the control point detection dialog by forcing cvfind to sleep N seconds.  
  
  **--hugin**              Make cvfind vaguely cpfind compatible by not protecting output .pto files.

  
  **--pattern** *pattern* Specify the shooting pattern.  Currently:

The shooting pattern may be specified as a 4 character string using the letter l, r, t, b: **l**eft, **r**ight, **t**op, **b**ottom.  lr = left to right, rl = right to left; tb = top to bottom, bt = bottom to top.  The first axis referenced is assumed to tbe sequential in terms of input series.   Example *rltb* assumes images were shot right to left, then top to bottom, sequential images are rows.  Example *btlr* assumes images were shot bottom to top, then left to right, sequential images are columns.   When omitted, no pattern is assumed and cvfind will not enforce an overlap based filter.    
  
  **--linearmatchlen** *N*   Set maximum image index difference for pairs to N.  Currently: 100000

Similar to the cpfind option, this limits how far cvfind will look for pairs from the current image in the series.  The default will try to match all images all other images in batches by sequence distance.

  **--rowsize** *N*          Set estimated row size to N.  Currently: 0

Can be used to specify the row ( or column, based on shooting pattern ) size.  cvfind assumes this many images are overlapped in sequence.  When specified, it is combined with **--rowsizetolerance** and **--pattern** options to construct a range of possible image pairs.  For example, for image 37, and **--rowsize 13** and **--rowsizetolerance 2**, valid image pairs for image 37 are { 48, 49, 50, 51, 52 } all other pairs are ignored, not processed, or filtered.   This value should be set when known for a significant improvement in speed, as well as eliminating spurious pairs.

  **--rowsizetolerance** *N* Effective row size is --rowsize +/- tol.  Currently: 1

See **--rowsize** details. 
  
  **--overlap** *N*          Set estimated overlap percentage.  Currently: 66%

Can be used to specify the expected overlap between images in an image pair.  For example --overlap 25 tells cvfind to ignore image data in the interior 50% of the image and assume all valid control points are in the outer 25% of the image.   Currently this overlap applies to both horizontal and vertical overlaps.  Generally specifying a larger number than actual is OK, especially if there is considerable variability from pair to pair.  --overlaptolerance is added to this value to produce the effective value.

  **--overlaptolerance** *N* Effective overlap is --overlap +/- tol.  Currently: 0%
  
When specified, this is added to --overlap and used as the effective overlap value at runtime.  It should not be used and instead the desired value specified via --overlap.  ( This option may be used at some point for weight based filtering. )

  **--noimages** *range*     Exclude pairs with images specified, e.g. all 0-39,44,66,92-

The **--noimage** option accepts a comma separated list of image numbers and ranges which are cvfind is forbidden from processing, both for loading image data, and for participating in image pairs.  By default this list is empty, with no image being forbidden.  If you wanted, for example, to have cvfind ignore images 3,5,7, and any after image 41 you could use **--noimage 3,5,7,42-**  The output PTO will reference all images from the input pto, however no control points for any of the forbidden images will be added.  If used with the **--detector pto** option and related options, it is possible to selectively add control points for specified images.  See **--images** for additional details.
  
  **--images** *range*       Include pairs with images specified, e.g. all 0- 27-50,60-70,-
  
Similar to the **--noimages** option, **--images** constructs a separate mandatory image list.  By default this list is empty, when specified, overrules the **--noimages** list.  When cvfind is deciding to load an image or do image processing on an image pair, it checks **--image** list first, if the image is present processing continues.  If the image is on the **--noimages** list, the task is ignored.   If present on either list, processing continues normally.  Since the **--images** and **--noimages** options are used to compute which images to load as well as which pairs to examine.

  **--centerfilter** *N*     Allow more than N centers ( of two possible ).

( Undocumented )

  **--peakwidth** *N*        Window width for detecting spurious pairs in adjacent images.

Legacy "Chippy" era option used to detect bad pairs.  Use **--loopfilter** instead.
  
  **--nocpfind**           Disables loading and examining images ( dry run )

Legacy "Chippy" era option which bypasses most of the homography logic.  Don't use.
  
  **--nothreads**          Disable threading.  Processes each image one at a time
  
  **--ncores** **-n** *N*        Changes the maximum threads from 1 to N

  **--savediag** *type*      Create diagnostic markup images.  Specify multiples types if needed.
        where *type* is: { masks | trials | pairs | aligned | neighbors }  

cvfind can generate various diagnostic images representing the internal state of the processing.  Working options include:  *masks* - when specified saves an images representing the masks generated based on shooting pattern.  Can be useful to determine why cvfind fails to produce control points where you expect.  *neighbors* - when specified produce a collage for each image, with the image centered, and those of any images it is paired with showing the image number and the number of control points for the pair.
  
  **--dohomography**       Computes homography between images in pairs.  ( Default: On )

By default cvfind uses opencv to find homography solutions between image pairs.  It is the default and this option should not be specified.   cvfind takes all matches ( usually thousands ) and computes the homography between the images in the pair.  Use of **--trialhomography** performs this step repeatedly on subsets of matches and then filters the results prior to the final homography calculation.  
  
  **--trialhomography**    Picks best of several overlaps based on CP distance.  ( Default: On )

Uses a window based distance filter on matches and then computes the homography based on that subset of points.   For tiling of images which were shot with the same relative rotation of sensor to subject, the images can be registered by sliding them relative to one another.   Control points for valid pairs will have nearly identical distance between the   
  
  **--trialhomsize** *N*     Match distance filter width for homography trials.  Currently: 100
  **--trialhomstep** *N*     Match distance filter steps of N for homography trials.  Currently: 33

Specified the width of the distance filter --trialhomography uses for matches, as well as how the window is racheted forward for the next trial.  The value is in RMS pixel distance units.  Image pairs where the images are slightly rotated relative to each other will have a wider range of distances.  The step size should be approximately a third that of window size to ensure multiple trials capture the same valid homography.  A smaller --trialhomsize and / or smaller --trialhomstep increases the  number of trial homograph runs and can result in diminishing returns.  For images with many repeated patterns, these values should be chosen to be approximately the size of the pattern or smaller.  This ensures at least one trial has its distance window centered over that of the "correct" overlap.  For image series with few small scale repeating patterns, these values can be increased to improve speed.  
  
  **--loopanalysis**       Enables loop homography analysis to identify bad pairs.
  **--loopfilter**         ... Plus enables loop homography filter to remove bad pairs. 
  **--looprepair**         ... Plus enabled loop homography repair using trials.

These options enable loop homography processing in cvfind, each subsequent option implies the one above.  **--loopfilter** is the default to remove all spurious pairs.  **--looprepair** attempts to use the trial homography data collected to find a trial that is mutually consistent with neighboring images.  this option runs AKAZE agains the pair to enhance the number of matches.  Generally this option results in > 85% of possible pairs, even corner to corner pairs, being detected while eliminating all invalid pairs.
  
  **--detect** *type*        Enable a detector.  Specify multiple types if needed.
  
  **--nodetect** *type*      Disable a detector.  Specify multiple type if needed.
  
  where *type* is:
  - all     - Generally used with --nodetect to clear defaults.
  - mask    - Selected detectors mask off non-overlapping central 100% of image.
  - pto     - Treats control points from input PTO as newly detected ( default ) 
  - orb     - best mix of performance and robustness.
  - sift    - slower, slightly more robust than ORB.  Generally finds
  - akaze   - slower, implements a auto tuning to return ~20K key points.
  - line    - detects line via binary descriptor, matched keylines.
  - surf    - slower, better for fine patterns?
  - brisk   - slow, generally not as robust as ORB.
  - gftt    - Good Features To Track, good, I guess.
  - dsift   - dense SIFT, force feature detection in a sliding window. 
  - blob    - detects blobs, then extracts SIFT features.
  - segment - detects line segment via LSD, matches keylines.
  - corner  - detects corners.
                       
  **--list detectors**     Provides additional info for each detector.

  **--cellsize** *N*         Specified the cell size for decimation.  Currently: 100px

When a high density of control points is detected, cvfind attempts to select the highest quality points over the entire overlap area.  It accomplishes this by dividing the image overlap into *N* x *N* pixel regions and sorting the list of control points in each region.  The filter then selects the best point from each, then the next, and so on untill the quota of control points is met.

  **--cellmincp**          Minimum control points per cell.  Currently: 0
  
  **--cellmaxcp**          Maximum control points per cell.  Currently: 5

The minimum and maximum number of control points to harvest from each cell during the round robin selection process.  The requirement indicated will cause more or less control points to survive decimation depending on the depth of each cell and the number of cells.

  **--cpperpairmin**       Minimum control points per image pair.  Currently: 15

  **--cpperpairmax**       Maximum control points per image pair.  Currently: 200

The minimum and maximum control points per pair.  When a pair has below the minimum it is considered to be bad, and all of the pairs control points are discarded.  When the maximum is exceeded cpfind will more aggressively apply the cell based decimation filter.

  **--neighborsnin**       Minimum number of pairs an image should have.  Currently: 15

  **--neighborsmax**       Maximum number of pairs an image should have.  Currently: 200

The minimum and maximum number of pairs an image should participate in.  These value result in warning messages, but otherwise do not impact the control points saved to the output PTO.  A corner image should be connected to 3 images, an edge image should be connected to at least 4, an interior image at least 6.  No image should be connected to more than 8 other images.  These values can be modified to alter the reporting thresholds to match the shooting pattern. 

  
  **--benchmarking**       Enable reporting of benchmarking info
  
  **--exitdelay** *N*        Sleep for N seconds prior to exiting.  Currently: 0
  
  **--prescale** *N*         (Broken) Downsamples images loaded to 1/N scale.  Currently: 100px

  **--mindist** *N*         minimum RANSAC error distance used for filtering.  Currently: 3
  
  **--maxdist** *N*         maximum RANSAC error distance used for filtering.  Currently: 9

The minimum and maximum RANSAC error distance values modify the way cvfind grades the fitness of the control points identified via homography.  The control point is projected from one image to the other in the image pair via the homography matrix discovered during homography calculations.  The difference between the original and final point represents the same error used by RANSAC.  Outliers in the homography calculation exceed 9.  Technically any value below 9 is an inlier.  However most of the best control points have an error below 3.  The mean error is calculated for all points.  When below **--mindist** the **--mindist** value is used instead - preserving more control points.  When the number passing the filter is below **--cpperpairmin**, the filter cutoff is increased to **--maxdist**   
  
  **--prescale** *N*         (Broken) Downsamples images loaded to 1/N scale.  Currently: 100px
  
  **--huginflipin**        When reading CPs from a .pto, mirrors their x and y coordinates
  
  **--huginflipout**       When adding new CPs to a .pto, mirror their x and y coordinates

For some reason Hugin may produce projects with the coordinates mirrored, where the origin is the lower right corner of the image.  It is not understood why or which config element in the PTO causes this behaviour.   The **--huginflipin** and **--huginflipout** options mirror the coordinates on input and output.  These can be used separately or together to re-write the coordinates.   Its hope to eventually identify the cause of the problem and eliminate the need for these options altogether.  These options are generally not used.
  
  **--debug**              Enable all debug messages, same as --loglevel 7
  
  **--verbose**            Enable some debug messages, same as --loglevel 4
  
  **--loglevel** *N*         0-2: Terse; 3-5: Verbose; 6-8: Debug; 9+: Trace.  Currently: 1
  
  **--log**                Logs messages to a file, a mess without --nothreads
  
  **--alignlog**           Keeps a separate log for each image pair

These options control how cvfind reports and logs its operations.  The main log enabled by **--log** does not include the image alignment logs.  Alignment logs record the details of the homography and filtering process, writing out a separate log for each image pair.  Ideally no logs need to be written and logging options should be used for troubleshooting, debugging, and development purposes.
  
  **--** **--ignore_rest**     ignores all further options
