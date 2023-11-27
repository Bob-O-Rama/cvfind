# Usage:
./cvfind [-d] [-h] [--option ...] [ -o output.pto] input.pto

# Options
  -d                   Enables startrup debugging.  Must be first if used.

Enables the debugging flag suring startup for troubleshooting command line parsing and other issues. 
  
  -h                   Display (this) help page
  
  -o output.pto        Saves modified copy of .pto to specified file

This preserves compatability with cpfind.  If no output file is specified, cvfind performs all the work, but does not produce an edited PTO.  PTO files specirfied with -o will not be overwritten and cvfind throws an error and terminates.  This is to protect prior work which can require significant computational effort.  the --hugin option may be used to allow overwriting of existing output files.

  --benchmarking       Enable reporting of benchmarking info

Benchmarking is currently in a state of flux.  Avoid.  
  
  --exitdelay N        Sleep for N seconds prior to exiting.  Currently: 0

This option will hold open the control point detection dialog by forcing cvfind to sleep N seconds.  
  
  --hugin              Make cvfind vaguely cpfind compatible by not protecting output .pto files.
  
  --linearmatchlen N   Set maximum image index difference for pairs to N.  Currently: 100000

Similar to the cpfind option, this limits how far cvfind will look for pairs from the current image in the series.  The default will try to match all images all other images in batches by sequence distance.

  --rowsize N          Set estimated row size to N.  Currently: 0

Can be used to specify the row ( or column, based on shooting pattern ) size.  cvfind assumes this many images are overlapped in sequence.  When specified, it is combined with --rowsizetolerance and --pattern options to construct a range of possible image pairs.  For example, for image 37, and --rowsize 13 and --rowsizetolerance 2, valid image pairs for image 37 are { 48, 49, 50, 51, 52 } all other pairs are ignored, not processed, or filtered.   This value should be set when known for a significant improvement in speed, as well as eliminating spurious pairs.

  --rowsizetolerance N Effective row size is --rowsize +/- tol.  Currently: 1

See --rowsize details. 
  
  --overlap N          Set estimated overlap percentage.  Currently: 66%
Can be used to specify the expected overlap between images in an image pair.  For example --overlap 25 tells cvfind to ignore image data in the interior 50% of the image and assume all valid control points are in the outer 25% of the image.   Currently this overlap applies to both horizontal and vertical overlaps.  Generally specifying a larger number than actual is OK, especially if there is considerable variability from pair to pair.  --overlaptolerance is added to this value to produce the effective value.
  
  --overlaptolerance N Effective overlap is --overlap +/- tol.  Currently: 0%
  
When specified, this is added to --overlap and used as the effective overlap value at runtime.  It should not be used and instead the desired value specified via --overlap.  ( This option may be used at some point for weight based filtering. )

  --noimages range     Exclude pairs with images specified, e.g. all 0-39,44,66,92-
The --noimage option accepts a comma separated list of image numbers and ranges which are cvfind is forbidden from processing, both for loading image data, and for participating in image pairs.  By default this list is empty, with no image being forbidden.  If you wanted, for example, to have cvfind ignore images 3,5,7, and any after image 41 you could use --noimage 3,5,7,42-  The output PTO will reference all images from the input pto, however no contrrol points for any of the forbidden images will be added.  If used with the --detector pto option and related options, it is possible to selectively add control points for specified images.  See --images for additional details.
  
  --images range       Include pairs with images specified, e.g. all 0- 27-50,60-70,-
  
Similar to the --noimages option, --images constructs a separate mandatory image list.  By default this list is empty, when specified, overrules the --noimages list.  When cvfind is deciding to load an image or do image processing on an image pair, it checks --image list first, if the image is present processing continues.  If the image is on the --noimages list, the task is ignored.   If present on either list, processing continues normally.  Since the --images and --noimages options are used to compute which images to load as well as which pairs to examine.  
  --centerfilter N     Allow more than N centers ( of two possible ).

( Undocumented )

  --peakwidth N        Window width for detecting spurious pairs in adjacent images.

Legacy "Chippy" era option used to detect bad pairs.  Use --loopfilter instead.
  
  --nocpfind           Disables loading and examining images ( dry run )

Legacy "Chippy" era option which bypasses most of the homography logic.  Don't use.
  
  --nothreads          Disable threading.  Processes each image one at a time
  
  --ncores -n N        Changes the maximum threads from 1 to N

  --savediag type      Create diagnostic markup images.  Specify multiples types if needed.
        where type is: { masks | trials | pairs | aligned | neighbors }  

cvfind can generate various diagnostic images representing the internal state of the processing.  Working options include:  masks - when specified saves an images representing the masks generated based on shooting pattern.  Can be useful to determine why cvfind fails to produce control points where you expect.  neighbors - when specified produce a collage for each image, with the image centered, and those of any images it is paired with showing the image number and the number of control points for the pair.
  
  --dohomography       Computes homography between images in pairs.  ( Default: On )

By default cvfind uses opencv to find homography solutions between image pairs.  It is the default and this option should not be specified.   cvfind takes all matches ( usually thousands ) and computes the homography between the images in the pair.  Use of --trialhomography performs this step repeatedly on subsets of matches and then filters the results prior to the final homography calculation.  
  
  --trialhomography    Picks best of several overlaps based on CP distance.  ( Default: On )

Uses a window based distance filter on matches and then computes the homography based on that subset of points.   For tiling of images which were shot with the same relative rotation of sensor to subject, the images can be registered by sliding them relative to one another.   Control points for valid pairs will have nearly identical distance between the   
  
  --trialhomsize N     Match distance filter width for homography trials.  Currently: 100
  --trialhomstep N     Match distance filter steps of N for homography trials.  Currently: 33
  --loopanalysis       Enables loop homography analysis to identify bad pairs.
  --loopfilter         ... Plus enables loop homography filter to remove bad pairs. 
  --looprepair         ... Plus enabled loop homography repair using trials.
  --detect type        Enable a detector.  Specify multiple types if needed.
  --nodetect type      Disable a detector.  Specify multiple type if needed.
        where type is: { akaze | brisk | blob | corner | line | orb | surf | sift ... 
                       ... segment | rsift | mask | all }.  'all' can be used to disable all.
        detector type: all     - Generally used with --nodetect to clear defaults.
                       mask    - Selected detectors mask off non-overlapping central 100% of image.

                       orb     - best mix of performance and robustness. ( Default )
                                 generally works without futzing.
                       sift    - slower, slightly more robust than ORB.  Generally finds
  (generally usable)             more usable features.  Old reliable.
                       akaze   - slower, implements a auto tuning to return ~20K key points.
                       line    - detects line via binary descriptor, matched keylines.
                       surf    - slower, better for fine patterns?
                       brisk   - slow, generally not as robust as ORB.
                       gftt    - Good Features To Track, good, I guess.

                       dsift   - dense SIFT, force feature detection in a sliding
                                 window.  Useful in traces usually ignored by above. 
    (experimental)     blob    - detects blobs, then extracts SIFT features.
                       segment - detects line segment via LSD, matches keylines.
                       corner  - detects corners.
  --list detectors     Provides additional info for each detector.
  --cellsize N         Specified the cell size for decimation.  Currently: 100px
  --cellmincp          Minimum control points per cell.  Currently: 0
  --cellmaxcp          Maximum control points per cell.  Currently: 5
  --cpperpairmin       Minimum control points per image pair.  Currently: 15
  --cpperpairmax       Maximum control points per image pair.  Currently: 200
  --neighborsnin       Minimum number of pairs an image should have.  Currently: 15
  --neighborsmax       Maximum number of pairs an image should have.  Currently: 200
  --benchmarking       Enable reporting of benchmarking info
  --exitdelay N        Sleep for N seconds prior to exiting.  Currently: 0
  --prescale N         (Broken) Downsamples images loaded to 1/N scale.  Currently: 100px
  --mindist  N         minimum RANSAC error distance used for filtering.  Currently: 3
  --maxdist  N         maximum RANSAC error distance used for filtering.  Currently: 9
  --prescale N         (Broken) Downsamples images loaded to 1/N scale.  Currently: 100px
  --huginflipin        When reading CPs from a .pto, mirrors their x and y coordinates
  --huginflipout       When adding new CPs to a .pto, mirror their x and y coordinates
  --debug              Enable all debug messages, same as --loglevel 7
  --verbose            Enable some debug messages, same as --loglevel 4
  --loglevel N         0-2: Terse; 3-5: Verbose; 6-8: Debug; 9+: Trace.  Currently: 1
  --log                Logs messages to a file, a mess without --nothreads
  --alignlog           Keeps a separate log for each image pair
  -- --ignore_rest     ignores all further options
