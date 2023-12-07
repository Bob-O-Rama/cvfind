/*

   Copyright (c) 2023 Robert Charles Mahar
                      bob@muhlenberg.edu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.         

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <ctime>

#include <thread>
#include <mutex>
#include <numeric>
#include <future>
#include <format>

#include <filesystem>

#include <unordered_set>

#if defined(__linux__)
  // For non portable use of prctl(PR_SET_NAME, "name_here", 0, 0, 0);
  #include <sys/prctl.h>
  #include <unistd.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/line_descriptor.hpp"
// opencv placed all non open IP into opencv_contrib which may not be present.
#if defined(WITH_OPENCV_CONTRIB)
  #include "opencv2/xfeatures2d.hpp"
#endif

// My only regrets with this codebase is that I cannot use emoji in comments.  Lots of emoji.

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
// opencv placed all non open IP into opencv_contrib which may not be present.
#if defined(__OPENCV_XFEATURES2D_HPP__)
  using namespace cv::xfeatures2d;
#endif

#define MAX_IMAGES 10000
#define MAX_CPS_PER_PAIR 10000

bool dotrace = false;            // enable trace output
bool debug = false;              // enable debug output
int loglevel = 1;		 // 0 - 2: terse, 3 - 5: verbose, 6 - 8: debug, > 8: trace

enum detectorType {
    AKAZE = 0,  // An entry for each detector algorithm
    ORB,
    BRISK,
    CORNER,
    GFTT,
    LINE,
    SEGMENT,
    BLOB,
    SURF,
    SIFT,
    DSIFT,      // Dense SIFT, keypoints on a grid
    PTO,        // CPs imported from the input PTO 
    ALL,        // A merged version of all, aka number of detectors
    COUNT       // Ordinal count of the number of detectors
  };

vector<std::string> detectorName { 
    "AKAZE", 
    "ORB", 
    "BRISK", 
    "CORNER", 
    "GFTT", 
    "LINE",
    "SEGMENT", 
    "BLOB", 
    "SURF", 
    "SIFT",
    "DSIFT",
    "PTO", 
    "ALL",
    "COUNT" };

// Used to sort vector< Point2f >
struct p2fcomp_st{
    bool operator() ( Point2f a, Point2f b ){
        if ( a.y != b.y ) 
            return a.y < b.y;
        return a.x <= b.x ;
    }
} p2fcomp;

// Used to sort vector< vector<int>  > descending by the 1st element in the child vector
struct vvintcompd_st{
    bool operator() ( vector<int> a, vector<int> b ){
            return a[0] > b[0];
    }
} vvintcompd;

struct ptoPoint {
  double x;
  double y;
};

/* is struct candidatePair
class ptoIndexPair {
  public:
  int idx1;
  int idx2;
  ptoIndexPair( int i_idx1, int i_idx2 ) { idx1 = i_idx1, idx2 = i_idx2; }
};
*/

struct ptoControlPoint {
  int idx1;      // source image index aka "n"
  int idx2;      // destination image index aka "N"
  ptoPoint src;    //
  ptoPoint dst;    //
  ptoPoint vector; //
  int line;        // the line number in the .pto image, if any
  double scalar;   //
  double slope;    //
  int detector;    // when known
};

/* subsumed by ptoControlPoint
struct ptoCPEntry {
  int idx1; // source image index aka "n"
  int idx2; // destination image index aka "N"
  ptoPoint src;
  ptoPoint dst;
  int line;
};
*/

int  preScaleFactor = 1;	 // scale ( down sample ) input images and unscale on output.
std::mutex imageMutex[MAX_IMAGES]; // a mutex for each ptoImage in images[]

class ptoImage {
  public:
  std::mutex* mutex;		// Pointer to a mutex
  bool needed;			// marked true when a pair is identified for image analysis
  int rowStartHint;		// Number of upvotes for this being the start of a row ( per shooting pattern )
  int rowEndHint;		// Number of upvotes for this being the end of a row ( per shooting pattern ) 
  int w;                        // width aka xSize from the PTO file       
  int h;                        // height aka ySize from the PTO file
  int idx;			// Image index
  std::string filename;         // Filename of the image as found in the PTO file.
  std::filesystem::path path;	// ( this is a path object, not just the parent directory ) 
  int groupRoot;                // The image's root group
  std::vector<int> candidates;  // Images this images may be paired with, possibly   
  std::vector<int> neighbors;   // Image neighbors passing all validation steps 
  Mat img;                      // the original image, cached in memory
  Mat imgGray;			// grey scale version of img
  //
  // Each ptoImage maintains its own cache of features detected for each detector type.  The detector deposits
  // features into the respective structure ( here ).  The matchers for an individual image pair deposits matches into
  // the ptoImagePair structure.   This allows for all of this data to be available during analysis.
  //
  std::vector<std::vector<KeyLine>> keylines = std::vector<std::vector<KeyLine>>(detectorType::ALL);
  std::vector<std::vector<KeyPoint>> keypoints = std::vector<std::vector<KeyPoint>>(detectorType::ALL);
  std::vector<Mat> descriptors = std::vector<Mat>(detectorType::ALL);
  
  // Constructor used during importing referenced images from a PTO
  ptoImage( std::string i_filename, int i_idx )
  { 
     ptoImage();
     filename = i_filename;
     idx = i_idx;
     needed = false;
  }
  
  // Constructor used for an uninitiallized ptoImage
  ptoImage() { w = h = idx = groupRoot = rowStartHint = rowEndHint = 0; needed = false; mutex = &imageMutex[idx]; };
  
  // loads the image and makes various useful copies of it
  void load()
  {
     lock();
     img = imread(filename);
     if( preScaleFactor == 1 )
     {
        cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
     }
     else
     {
        Mat scaled;
                // specify fx and fy and let the function compute the destination image size.
        resize(img, scaled, Size(), (float)1 / (float)preScaleFactor, (float)1 / (float)preScaleFactor, INTER_LINEAR );
        cvtColor(scaled, imgGray, cv::COLOR_BGR2GRAY);
     }
     // img = imgGray;
     unlock();
  }
  
  
  // Caller manages concurrency, as when loading an image or populating features / keypoints 
  void lock()   { mutex->lock(); }
  void unlock() { mutex->unlock(); }


};   // ptoImage 


// Get the center of mass ( COM ) in terms of image coordinates
Point2f COM( std::vector<Point2f> pv )
{
   Point2f com = Point2f(0,0);
   for( int i = 0; i < pv.size(); i++ )
   {
      com.x += pv[i].x;
      com.y += pv[i].y;
   }
   if( pv.size() > 0 )
   {
      com.x = com.x / pv.size();
      com.y = com.y / pv.size();
   }
   return com;
}

// Return a comma separated string of ranges, e.g. {1,3,5,7,8,9,10,...18,100,106} --> "1,3,5,7-18,100,106"
std::string vi2csv( std::vector<int> vi )
{
   // handle edge cases
   if( vi.size() == 0 ) return "";
   if( vi.size() == 1 ) return to_string( vi[0] );
   if( vi.size() == 2 ) return to_string( vi[0] ) + "," + to_string( vi[1] );
   std::string retval = "";
   int idx = 0;
   int cur = 0;
   while( idx < vi.size() )
   {
      retval += to_string(vi[idx] ); // add vi[idx] 
      cur = idx;
      // while vi[idx+1] INCREMENTS by one
      while( idx + 1 < vi.size() && vi[idx] == vi[idx+1] - 1 )
      {
         idx++;
      }
      if( idx > cur )
      {
         retval += "-" + to_string( vi[idx] );
      }
      idx++;
      if( idx < vi.size() ) retval += ",";
   }
   return retval;
}

// Global to hold all ptoImage objects referenced in the PTO
std::vector<ptoImage> images{};              // Holds a class instance for each image


// A point pad serves as a scratch pad for holding control point clouds for a pair of images.
// The image pair object maintains lists of point pads, generally the most recent is used 
// others are kept for troubleshooting purposes.
class ptoPointPad {
  public:
     std::string padName;
     std::vector<Point2f> points1;  // image1 point array
     std::vector<Point2f> points2;  // image2 point array
     std::vector<int> pointType;    // detectorType ENUM 
     std::vector<int> lineIdx;      // If in-memory PTO line number if imported from PTO
     std::vector<float> distance;   // "Distance" between corresponding control points cionfirmed by homography     
     cv::Rect rect1;                // rect containing points1
     cv::Rect rect2;                // rect containing points2
     cv::RotatedRect minAreaRect1;  // minAreaRect contating points1
     cv::RotatedRect minAreaRect2;  // minAreaRect contating points2
     cv::Point2f com1;              // center of mass for points1
     cv::Point2f com2;              // center of mass for points2
     float comscalar;		    // scalar distance between centers of mass
     float comslope;		    // slope of line passing through centers of mass
     int idx1 = 0;                  // source image index aka "n"
     int idx2 = 0;                  // destination image index aka "N"
     int deltaX, deltaY;            // the X and Y difference between com1 and com2
     Mat h;			    // current homography matrix, if any
      
     void refresh()
     {     
        if( idx1 == idx2 ) cout << "FATAL: Uninitiallized pointPad structure." << std::endl;
        // update bounding rectangles
        if( points1.size() > 0 ) rect1 = boundingRect( points1 );
        if( points2.size() > 0 ) rect2 = boundingRect( points2 );
        if( points1.size() > 0 ) minAreaRect1 = minAreaRect( points1 );
        if( points2.size() > 0 ) minAreaRect2 = minAreaRect( points2 );
        com1 = COM( points1 );
        com2 = COM( points2 );
        deltaX = com2.x - com1.x;
        deltaY = com2.y - com1.y;
        comscalar=(float)sqrt( ( deltaX * deltaX ) + ( deltaY * deltaY ) );
        if( deltaX == 0 )
        {
           comslope=(float)deltaY * (float)1000000; // instead of divide by zero, multiply by 1000000
        }
        else
        {	
           comslope=(float)( (float)deltaY / (float)deltaX );
        }

     }
     
     int size() { return pointType.size(); }

     // copies bare minimum data to prime the structure to revieve filtered data
     void prime( ptoPointPad *pp ) 
     {
         idx1  = pp->idx1;
         idx2  = pp->idx2;
         padName = pp->padName;
         return;
     }

     std::vector<std::string> dump()
     {
        std::vector<std::string> retval;
        std::string s;
        
        if( idx1 == idx2 ) cout << "FATAL: Uninitiallized pointPad structure." << std::endl;

        // Loop through each match, apply a quadrant based filter
        for( size_t i = 0; i < size(); i++ )
        {
           int xSize = images[idx1].img.cols;
           int ySize = images[idx1].img.rows;
           float x1, y1, x2, y2, dx, dy, scalar, slope;
           std::string notes = "";

           x1 = points1[i].x;
           x2 = points2[i].x;
           y1 = points1[i].y;
           y2 = points2[i].y;

           dx = x2 - x1;
           dy = y2 - y1;
    
           scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

           if( dx == 0 )
           {
               slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
           }
           else
           {	
               slope=dy / dx;
           }
    
           s = "Raw Matches: [" + to_string(idx1)  + "] --> [" + to_string(idx2) + "] " \
               + detectorName[ pointType[i] ] + " ( " + to_string( x1 ) + ", " + to_string( y1 ) \
               + " ) <---> ( " + to_string( x2 ) + ", " + to_string( y2 ) + " ) Scalar: " \
               + to_string( scalar ) + " Slope: " + to_string( slope ) + " PTO_Line: " + to_string( lineIdx[i] ) ;

           retval.push_back( s );
        }
        return retval;
     }
};

//
// Cheesy class to keep benchmarks
//
class benchmarkEpoc {
  public:
  std::string name;
  std::chrono::time_point<std::chrono::system_clock> time;
  
  benchmarkEpoc( std::string i_name, std::chrono::time_point<std::chrono::system_clock> i_time )
  { name = i_name; time = i_time; }
};

//
// The "idea" of an image pair.   Includes all feature, match, control point, and homography info for a pair
// only idx1 < idx2 is valid.  This reflects the way PTO files maintain their control points.
// If homography projected in the reverse order is needed, that can be accomplished by inversion of the H matrix. 
//
class ptoImgPair {
  public:
     int idx1 = 0 ; // source image index aka "n"
     int idx2 = 0 ; // destination image index aka "N"
     std::vector<benchmarkEpoc> benchmarks;
     std::vector<ptoControlPoint> points{};
     ptoControlPoint cpAve;
     int groupRoot; // image number of the first image in the group set to itself by default.
     std::vector<std::vector<int>> loops;  // Loops tried ( all pairs get a copy of all loop results )
     std::vector<float> loopError; // Error for loop[n]
     int votesL2, votesL1, votesL0, votesLB;
     int repairScalar = 0; // For rework, this scalar overrides any detected scalar and is used for final homography
            
     ptoImgPair() { idx1 = idx2 = groupRoot = votesL2 = votesL1 = votesL0 = votesLB = 0; };  // An invalid unitialized state
  
     // matches [ detector type ] [ scratch_pad ] [ match_idx ]
     // we probabaly don't need a locking mechanism as access will be pair wise, single threaded for write
     // detectorType::COUNT is the ordinal number of detector slots we need to have, including the synthetic
     // "ALL" detector.

     std::vector<std::vector<std::string>> padName = \
            std::vector<std::vector<std::string>>(detectorType::COUNT);     
     
     std::vector<std::vector<std::vector<DMatch>>> matches = \
            std::vector<std::vector<std::vector<DMatch>>>(detectorType::COUNT);

     std::vector<std::vector<std::vector<ptoControlPoint>>> cps = \
            std::vector<std::vector<std::vector<ptoControlPoint>>>(detectorType::COUNT);

     // point pads for trial homography
     
     std::vector<ptoPointPad> trialPointPad;

     // Unified points1 and points2 and detectorType for the point

     std::vector<ptoPointPad> pointPad;

     // current homography matrix, if any
     Mat h;

     // list of strings proposed edits to Hugin PTO file
     std::vector<std::string> newCPlist;

     // ( Probabaly need something for control line )

     // return index to a new matches, padname, and cps vector for the specified detector. 
     int newMatchPad( int i_detector, std::string i_padName )
     {

        if( idx1 == idx2 ) cout << "FATAL: Uninitiallized ptoImagePair structure." << std::endl;

        if( loglevel > 8 ) cout << "TRACE: newMatchPad( " << i_detector << ", '" << i_padName << "' )" << std::endl;

        padName[i_detector].push_back( i_padName );

        std::vector<DMatch> newmatches;
        matches[i_detector].push_back( newmatches );

        std::vector<ptoControlPoint> newcps;
        cps[i_detector].push_back( newcps );
        
        if( matches[i_detector].size() != cps[i_detector].size() || matches[i_detector].size() != padName[i_detector].size() )
           cout << "FATAL: newMatchPad(" << i_detector << ", '" << i_padName
                << "'): cps.size() != matches.size() != padName.size()" << std::endl;
           
        return cps[i_detector].size() -1; // return index, not count.  Also cps, padname, matches have same size.
     }

     int currentMatchPad( int i_detector )
     {
        if( loglevel > 8 ) cout << "TRACE: currentMatchPad( " << i_detector << " )" << std::endl;
     
        if( cps[i_detector].size() == 0 )
           return newMatchPad( i_detector, "First Match Pad" );   
        return (cps[i_detector].size() - 1);
     }


     // Accumulates all cps[detector=0...N] and updates the cps pad specified, returns total points accumulated
     int refreshPointPad( )
     {
         if( idx1 == idx2 ) 
         {
            cout << "FATAL: refreshPointPad() called with uninitialized ptoImagePair structure.  idx1 == idx2 == " << idx1 << std::endl;
            cout.flush();
            loglevel = 10;
         }
            
         if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ..." << std::endl;

         int pp = currentPointPad(); // returns the index of the current point pad or a new one if none exists
         

         // clear the point pad         
         pointPad[pp].points1.clear();
         pointPad[pp].points2.clear();
         pointPad[pp].pointType.clear();
         pointPad[pp].lineIdx.clear();
         pointPad[pp].distance.clear();

/*
         // Remove all entries NOT imported from the PTO file ( else we loose the lineIdx values )
         for( int i=0; i < pointPad[pp].size() > 0; i++ )
         {
            if( pointPad[pp].pointType[i] != detectorType::PTO )
            {
                pointPad[pp].points1.erase( pointPad[pp].points1.begin() + i );
                pointPad[pp].points2.erase( pointPad[pp].points2.begin() + i );
                pointPad[pp].pointType.erase( pointPad[pp].pointType.begin() + i );
                pointPad[pp].lineIdx.erase( pointPad[pp].lineIdx.begin() + i );
                pointPad[pp].distance.erase( pointPad[pp].distance.begin() + i );
                // at this point i points at what was i+1, so decrement i so we examine that one nect time
                i--;                       
            }
         }
*/

         for( int detector=0; detector < detectorType::ALL; detector++ )
         {

             if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... extract points for " << detectorName[detector] << std::endl;
             // Look at the current pad for this detector
             int pad = currentMatchPad(detector);
             if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... extract points for " << detectorName[detector] 
                                << " MatchPad " << pad << " Matches: " << matches[detector][pad].size()  
                                << std::endl;             
             if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... Image Pair: " << idx1 << ":" << idx2 << std::endl;

             // For true detectors ...
             for( int i = 0; detector != detectorType::PTO && i < matches[detector][pad].size(); i++ )
             {
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... extract points " << i << " for " << detectorName[detector] << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... KP:" << images[idx1].keypoints.size() 
                                   << ":" << images[idx2].keypoints.size() << std::endl;                
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... KP:" << images[idx1].keypoints[detector].size() 
                                   << ":" << images[idx2].keypoints[detector].size() << std::endl;                
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... matches.size():" << matches.size() 
                                   << ":" << matches.size() << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... matches[].size():" << matches[detector].size() 
                                   << ":" << matches[detector].size() << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... matches[][].size():" << matches[detector][pad].size() 
                                   << ":" << matches[detector][pad].size() << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... Matches Pair:" << matches[detector][pad][i].queryIdx 
                                   << ":" << matches[detector][pad][i].trainIdx << std::endl;

                Point2f pt1 = images[idx1].keypoints[detector][ matches[detector][pad][i].queryIdx ].pt;
                Point2f pt2 = images[idx2].keypoints[detector][ matches[detector][pad][i].trainIdx ].pt;
                
                int detType = detector;
                
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... Extracted points" << pt1 << ":" << pt2 
                                   << " for detector " << detType << std::endl;

                pointPad[pp].points1.push_back( pt1 );
                pointPad[pp].points2.push_back( pt2 );
                pointPad[pp].pointType.push_back( detType );
                pointPad[pp].lineIdx.push_back( 0 ); // no PTO line
                pointPad[pp].distance.push_back( 0 );                           
             }
             
             // For points extracted from the PTO ...
             for( int i = 0; detector == detectorType::PTO && cps[detector].size() > 0 && i < cps[detector][pad].size(); i++ )
             {
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... extract point " << i << " for " << detectorName[detector] << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... cps.size():" << cps.size() << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... cps[" << detector << "].size():" << cps[detector].size() << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... cps[" << detector << "][" << pad << "].size():" 
                                                                    << cps[detector][pad].size() << std::endl;
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... cps[" << detector << "][" << pad << "][" << i << "]:" << std::endl;

                Point2f pt1, pt2;
                int line;
                
                pt1.x = cps[detector][pad][i].src.x;
                pt1.y = cps[detector][pad][i].src.y;
                pt2.x = cps[detector][pad][i].dst.x;
                pt2.y = cps[detector][pad][i].dst.y;
                line = cps[detector][pad][i].line;
                
                int detType = detector;
                
                if( loglevel > 8 ) cout << "TRACE: refreshPointPad( ) ... Extracted points" << pt1 << ":" << pt2 
                                   << " for detector " << detType << " PTO Line: " << line << std::endl;

                // Detected control points from homography are added to the in memory PTO,
                // so we only want ones where the detector type PTO.  So cue the kludge.

                pointPad[pp].points1.push_back( pt1 );
                pointPad[pp].points2.push_back( pt2 );
                pointPad[pp].pointType.push_back( detType );
                pointPad[pp].lineIdx.push_back( line ); // no PTO line
                pointPad[pp].distance.push_back( 0 );                          
             }

             
         } // for each detector

         if( loglevel > 8 ) cout << "... refreshPointPad( ) = " << pointPad[pp].points1.size() << std::endl;
         
         return pointPad[pp].points1.size();
     }


     // return current Point Pad, make one if needed
     int currentPointPad( )
     {
        if( loglevel > 8 ) cout << "TRACE: currentPointPad( ) ..." << std::endl;
        if( loglevel > 8 ) cout << "TRACE: currentPointPad( ) pointPad.size()=" << pointPad.size() << std::endl;
        if( pointPad.size() == 0 )
        {
           newPointPad( "Default pointPad for " + to_string( idx1 ) + ":" + to_string( idx2 ) );
        }
        if( loglevel > 8 ) cout << " ... currentPointPad( ) = " << pointPad.size() - 1 << std::endl;
        return pointPad.size() - 1;
     }

     // make an empty point pad
     int newPointPad( std::string i_padName )
     {

        if( loglevel > 8 ) cout << "TRACE: newPointPad( '" << i_padName << "' ) ..." << std::endl;
        ptoPointPad foo;
        foo.padName = i_padName;
        foo.idx1 = idx1;
        foo.idx2 = idx2;
        pointPad.push_back( foo );        
        if( loglevel > 8 ) cout << "... newPointPad( '" << i_padName << "' ) = " << pointPad.size() - 1 << std::endl;
        return pointPad.size() - 1;
     }
      

};  // ptoImgPair

// Represents a loop homography trial indicating both the pairwise links in image indices aas well as the H matrix
class ptoTrialPermutation {
  public:
     std::vector<int> chain;          // An image chain ( list indices of images[] array ) 
     std::vector<int> trialChain;     // The indices of each trialPointPad for each image pair in the chain
     std::vector<cv::Mat> hChain;     // The homography chain
     Point2f errorPoint;              // the error values returned by transitiveError( )
     float error;	              // the scalar magnitide of the error.
}; // ptoTrialPermutation


// Useful Macros ...

#define RECT2STR(R) "Corner: (" + to_string( R.x ) + ", " + to_string( R.y ) + ") Size: ("\
        + to_string( R.width ) + ", " + to_string( R.height ) + ")"


#define BENCHMARK(B,T) (B).push_back(benchmarkEpoc( (T), std::chrono::high_resolution_clock::now() ) )


#define IMGLOG sout << "alignImage( " << left0padint(idx1, 4) << " --> " << left0padint(idx2, 4) << " ): "


#define OPTIONBOOL(O,V,X) \
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], #O) ) \
       { \
           (V) = (X); \
           if( loglevel > 6 ) sout << "Parsed argv[" << j << "]='" << argv[j] \
                            << "' option ( " << #O << " ).  Setting " << #V << " to " << (V) << std::endl; \
           for(int i=1; i<argc; ++i) \
               argv[i]  = argv[i+1]; \
           --argc; \
           parsed = true; \
       }


#define OPTIONINT(O,V,MIN,MAX) \
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], #O) ) \
       { \
           std::string theoption = argv[j]; \
           std::string theparm; \
           int thenum = 0; \
           bool inbounds = false; \
           for(int i=1; i<argc; ++i) \
               argv[i]  = argv[i+1]; \
           --argc; \
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) ) \
           { \
              theparm = argv[j]; \
              thenum  = stoi( theparm ); \
              if( thenum >= (MIN) && thenum <= MAX ) \
              { \
                 (V) = thenum; \
           if( loglevel > 6 ) sout << "Parsed argv[" << j << "]='" << theoption << "' with value '" << theparm \
                            << "' option ( " << #O << " ). Range [" << (MIN) << "," << (MAX) << "]" \
                            << "  Setting " << #V << " to " << (V) << std::endl; \
              } \
              for(int i=1; i<argc; ++i) \
                  argv[i]  = argv[i+1]; \
              --argc; \
           } \
           if( thenum < (MIN) || thenum > (MAX) ) \
           { \
              sout << "ERROR: option " << theoption << " should be between " << (MIN) << " and " << (MAX) << std::endl; \
              return 1; \
           } \
           parsed = true; \
       }


#define OPTIONINTCSV(O,V,MIN,MAX) \
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], #O) ) \
       { \
           std::string theoption = argv[j]; \
           std::string theparm; \
           std::vector<int> thenums; \
           bool inbounds = false; \
           for(int i=1; i<argc; ++i) \
               argv[i]  = argv[i+1]; \
           --argc; \
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) ) \
           { \
              theparm = argv[j]; \
              if( theparm == "all" ) theparm = to_string(MIN) + "-" + to_string((MAX) - 1); \
              if( theparm.size() > 0 && theparm[theparm.size()-1] == '-' ) theparm += to_string((MAX) - 1); \
              if( theparm.size() > 0 && theparm[0] == '-' ) theparm.insert(0, to_string((MIN))); \
              thenums  = findIntList( theparm ); \
              if( thenums.size() >= (MIN) && thenums.size() <= MAX ) \
              { \
                 (V) = thenums; \
           if( loglevel > 6 ) sout << "Parsed argv[" << j << "]='" << theoption << "' with value '" << theparm \
                            << "' option ( " << #O << " ). Range [" << (MIN) << "," << (MAX) << "]" \
                            << "  Setting " << #V << " to " << vi2csv(V) << std::endl; \
              } \
              for(int i=1; i<argc; ++i) \
                  argv[i]  = argv[i+1]; \
              --argc; \
           } \
           if( thenums.size() < (MIN) || thenums.size() > (MAX) ) \
           { \
              sout << "ERROR: option " << theoption << " list size should be between " << (MIN) << " and " << (MAX) << std::endl; \
              return 1; \
           } \
           parsed = true; \
       }


#define OPTIONFLOAT(O,V,MIN,MAX) \
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], #O) ) \
       { \
           std::string theoption = argv[j]; \
           std::string theparm; \
           float thenum = 0; \
           bool inbounds = false; \
           for(int i=1; i<argc; ++i) \
               argv[i]  = argv[i+1]; \
           --argc; \
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) ) \
           { \
              theparm = argv[j]; \
              thenum  = stof( theparm ); \
              if( thenum >= (MIN) && thenum <= MAX ) \
              { \
                 (V) = thenum; \
           if( loglevel > 6 ) sout << "Parsed argv[" << j << "]='" << theoption << "' with value '" << theparm \
                            << "' option ( " << #O << " ). Range [" << (MIN) << "," << (MAX) << "]" \
                            << "  Setting " << #V << " to " << (V) << std::endl; \
              } \
              for(int i=1; i<argc; ++i) \
                  argv[i]  = argv[i+1]; \
              --argc; \
           } \
           if( thenum < (MIN) || thenum > (MAX) ) \
           { \
              sout << "ERROR: option " << theoption << " should be between " << (MIN) << " and " << (MAX) << std::endl; \
              return 1; \
           } \
           parsed = true; \
       }


//
// - - - - - - - - - Parallel Sort Template - - - - - - - - -- - - - - - -
//

/*  DoDo: Remove: Hopelessly broken

template <typename Q>
void swap(size_t i, size_t j, std::vector<Q>& v) {
  std::swap(v[i], v[j]);
}

template <typename Comp, typename Vec, typename... Vecs>
void parallel_sort(const Comp& comp, Vec& keyvec, Vecs&... vecs) {
  (assert(keyvec.size() == vecs.size()), ...);
  std::vector<size_t> index(keyvec.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
            [&](size_t a, size_t b) { return comp(keyvec[a], keyvec[b]); });

  for (size_t i = 0; i < index.size(); i++) {
    if (index[i] != i) {
      (swap(index[i], i, keyvec), ..., swap(index[i], i, vecs));
      std::swap(index[index[i]], index[i]);
    }
  }
}

*/

template< typename T, typename U >
std::vector<T> sortVecAByVecB( std::vector<T> & a, std::vector<U> & b ){

    // zip the two vectors (A,B)
    std::vector<std::pair<T,U>> zipped(a.size());
    for( size_t i = 0; i < a.size(); i++ ) zipped[i] = std::make_pair( a[i], b[i] );

    // sort according to B
    std::sort(zipped.begin(), zipped.end(), []( auto & lop, auto & rop ) { return lop.second < rop.second; }); 

    // extract sorted A
    std::vector<T> sorted;
    std::transform(zipped.begin(), zipped.end(), std::back_inserter(sorted), []( auto & pair ){ return pair.first; }); 

    return sorted;
}


template<typename T>
void appendVector(vector<T>& v1, vector<T>& v2)
{
    v1.insert(v1.end(), v2.begin(), v2.end());
}

template<typename T, typename ...Args>
void appendVector(vector<T>& v1, vector<T>& v2, Args... args)
{
     v1.insert(v1.end(), v2.begin(), v2.end());
     appendVector(v1, args...);
}

//
// - - - - - - - - Global MUTEXes for various concurrent processes- - - - - - - - -
//
std::mutex logger_guard;
std::mutex newcplist_guard;

//
// Allow output of vectors in the format {_,_,_,....,_}
//
template <typename S>
ostream& operator<<(ostream& os,
                    const vector<S>& vector)
{
    // Printing the vector as {_,_,_,...,}
    os << "{";
    for ( int i=0; i < vector.size(); i++ )
    {
       auto element = vector[i];
       os << element;
       if( i+1 != vector.size() ) os << ","; // add comma if not final element
    }
    os << "}";  
    return os;
}

//
// - - - - - - - - - - - Tee for logging to a file - - - - - - - - - - - -
//

class soutstream
{
  public:
  std::ofstream coss;
  std::string facility = " (main) ";
  // this is the type of std::cout
	typedef std::basic_ostream<char, std::char_traits<char> > CoutType;

	// this is the function signature of std::endl
	typedef CoutType& (*StandardEndLine)(CoutType&);

	// define an operator<< to take in std::endl
	soutstream& operator<<(StandardEndLine manip)
	{
		// call the function, but we cannot return it's value
		manip(std::cout);

		if(coss.is_open())
		{
		   coss << "\n";
		   coss.flush();
		}

		return *this;
	}
};

// std::mutex logger_guard;  // see globals...

template <class T>
soutstream& operator<< (soutstream& st, T val)
{
  logger_guard.lock();
  if(st.coss.is_open())
  {
     st.coss << val;
     st.coss.flush();
  }
  std::cout << val;
  cout.flush();   
  logger_guard.unlock();  
  return st;
}


//
// Horrific Date String Format
//
string GetDateTime()
{
	time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );

	int yearval=(now->tm_year + 1900);
	int monthval=(now->tm_mon + 1);
	int dayval=(now->tm_mday);
	int hourval=(now->tm_hour);
	int minval=(now->tm_min);
	int secval=(now->tm_sec);
	
	char c_year[12];
	sprintf(c_year, "%d", yearval);
	char c_month[12];
	sprintf(c_month, "%d", monthval);
	char c_day[12];
	sprintf(c_day, "%d", dayval);
	char c_hour[12];
	sprintf(c_hour, "%d", hourval);
	char c_min[12];
	sprintf(c_min, "%d", minval);
	char c_sec[12];
	sprintf(c_sec, "%d", secval);

	string output = "";

	output += c_year;
//	output += "-";
	output += c_month;
//	output += "-";
	output += c_day;
	output += "_";
	output += c_hour;
//	output += "-";
	output += c_min;
//	output += "-";
	output += c_sec;

	return output;
}

//
// Instantiate the global Tee stream logger
//
soutstream sout;

// return -1 if a preceeds b, 1 if b preceeds a, or 0 if one or other not found
int StringOrder( std::string s, std::string a, std::string b )
{ 
  std::size_t afound = s.find(a);
  std::size_t bfound = s.find(b);
  if( afound == bfound || afound == std::string::npos || bfound == std::string::npos )
     return 0;  // happens if both are std::string::npos
  int diff = (int)afound - (int)bfound;  // nagative if a before b 
  return abs( diff ) / diff;   // returns -1 or +1 
}


//
// imageIndexPair - The "idea" of a pair of image indices.
//
class imageIndexPair {
  public: 
  int idx1;
  int idx2;

  bool operator == (const imageIndexPair &p)
  {
     if (idx1 == p.idx1 && idx1 == p.idx1)
        return true;
     return false;
  }

  friend std::ostream& operator<<(std::ostream& o, imageIndexPair const& i_iip)
  {
     o << "(" << i_iip.idx1 << ":" << i_iip.idx2 << ")";
     return o;
  }
};

//
// Global list of image pairs of interest - this is the punch list for image processing.
//
std::vector<imageIndexPair> candidatePairs; // a list of index pairs to process 

//
// my own function to left pad a positive integer with zeros,
// because COBOL has been doing this since 1953.
//

// left pad with zeroes
inline std::string left0padint(int n, int len)
{
    std::string result(len--, '0');
    for (int val=(n<0)?-n:n; len>=0&&val!=0; --len,val/=10)
       result[len]='0'+val%10;
    if (len>=0&&n<0) result[0]='-';
    return result;
}

// left pad with spaces
inline std::string left_padint(int n, int len)
{
    std::string result(len--, ' ');
    if( n==0 ) result[len] = '0';
    for (int val=(n<0)?-n:n; len>=0&&val!=0; --len,val/=10)
       result[len]='0'+val%10;
    if (len>=0&&n<0) result[0]='-';
    return result;
}


//
// Calculate mean "the Bob way"
//
float calculateMean(std::vector<float> i_data)
{
  float sum = 0.0, mean, standardDeviation = 0.0;
  int i;
  
  if( i_data.size() == 0 ) return (float) 0;
  for(i = 0; i < i_data.size(); ++i) sum += i_data[i];
  mean = sum / i_data.size();

  if( false ) sout << " INFO: calculateMean(): " << i_data.size() << " data points, sum=" << sum << ", mean=" << mean 
                    << ", sd=" << sqrt(standardDeviation / i_data.size()) << std::endl;

  return mean;
}


//
// Calculate standard deviation - "the Bob Way"
//
float calculateSD(std::vector<float> i_data)
{
  float sum = 0.0, mean, standardDeviation = 0.0;
  int i;
  
  if( i_data.size() == 0 ) return (float) 0;

  for(i = 0; i < i_data.size(); ++i) sum += i_data[i];
  mean = sum / i_data.size();

  for(i = 0; i < i_data.size(); ++i) standardDeviation += pow(i_data[i] - mean, 2);

  if( false ) sout << " INFO: calculateSD(): " << i_data.size() << " data points, sum=" << sum << ", mean=" << mean 
                    << ", sd=" << sqrt(standardDeviation / i_data.size()) << std::endl;

  return sqrt(standardDeviation / i_data.size());
}


//
// findToken() finds values of the format [delimiter]VALUE or [delimiter]"VALUE" as used throughout Hugin PTO files
//
std::string findToken( std::string s, std::string delim )
{
	// Find the delimiter string and extract right substring following it.
   std::string right = s.substr(s.find(delim) + delim.length(), s.length());
        // If the value starts with a double quote, find the end quote
	// strip and return the value
   if( right[0] == '"' )
   {
       return right.substr(1,right.substr(1,right.length()-1).find('"'));
   }
        // ToDo: bounds checking when last parameter ( no final space )
   return right.substr(0,right.find(' '));
}

//
// findIntList() - takes string containing a comma separated list of INTs and returns a vector of ints
// 
std::vector<int> findIntList ( std::string s )
{
   std::vector<int> vect;
   std::string recon;
   std::stringstream ss(s);  // get string stream of input string
   // reads an int from ss into i, until it cannot.    
   for (int i=0; ss >> i;)
   {
      // handle 'N-M' case like ...,4-21,... 4 is added on prior itteration so add N+1 thru M
      if( i < 0 )
      {
         if( vect.size() == 0 ) vect.push_back( 0 ); // add a leading 0 to the list if list empty
         for( int j = vect.back() + 1; j < abs(i); j++ ) vect.push_back( j ); // for N+1...M-1
         i = abs(i); // "remove" the '-' 
      }     
      vect.push_back(i);    
      if (ss.peek() == ',')
          ss.ignore();
   }
   // sort and deduplicate  
   sort( vect.begin(), vect.end() );
   vect.erase( unique( vect.begin(), vect.end() ), vect.end() );
   if( loglevel > 8 ) cout << "findIntList( '" << s << "' ) = " << vect << std::endl;   
   return vect;
}    


//
// uniqueify() - accepts a vector of strings and removes duplicate lines while preserving the order of the vector
//

std::vector<std::string> uniqueifyVecString( std::vector<std::string> lines )
{
    std::unordered_set< std::string > theSet;
    std::vector<std::string> retval = lines;
    auto iter = std::stable_partition(retval.begin(), retval.end(), [&](std::string s) 
           { bool ret = !theSet.count(s); theSet.insert(s); return ret; }); // returns true if the item has not been "seen"
    retval.erase(iter, retval.end());           
    // for(auto p : retval)
    //    std::cout "retval: " << p << std::endl;
    return retval;
}

//
// Horrific block of global variables, mostly operational config settings specified by the user
// or their defaults.   ToDo: reduce to a config object as would be needed if the code is split into proper modules 
//

const int MAX_FEATURES = 10000;            // Number of KP's to create per images
const float GOOD_MATCH_PERCENT = 0.50f;    // Keep the top "best" 50%

std::string cmdline = "";	 // The command line
// bool dotrace = false;         // enable trace output
int  exitDelay = 0;		 // after everything is completed, sleep exitDelay seconds ( useful for --hugin debugging )
bool showhelp = false;		 // Show help page and terminate.
bool showversion = false;        // Show version information and build options, and so on.
bool dotest = false;             // Perform a self check, then exit with error level.
                                 //
bool save_all_images = false;    // Keep diagnostic markup images even when it goed badly
bool save_good_images = false;   // Keep diagnostic markup images when match it good
bool saveDiagTrials = false;	 // Homography trial markups  
bool saveDiagPairs = false;	 // Pairs markup 
bool saveDiagAligned = false;	 // Aligned overlap from markup 
bool saveDiagMasks = false;	 // Masks markup ( original image, with masked off area )
bool saveDiagNeighbors = false;	 // Neighbors markup
bool saveDiagDecimation = false; // Neighbors markup
bool saveDiagGood = true;	 // 
bool saveDiagBad = true;	 // 
                                 //
bool threaded = true;            // Enable parallel threads when it makes sense
int  threadsMax = 1;             // Updated at runtime or by a command line option; 
bool verbose = false;            // Be blabby 
bool benchmarking = false;       // Output timing info
bool nocpfind = false;           // reverts to "chippy" mode skips all image alignment
bool nochippy = true;            // disables removal of control points based on chippy analysis.
bool hugin = false;		 // disable file safeguards when run from hugin
bool quadrantFilter = false;	 // Apply quadrant filter
bool distanceFilter = true;	 // Apply distance filter
bool cpoverlapFilter = false;    // Apply overlap filter to control points
bool kpoverlapFilter = false;    // Apply overlap filter to keypoints entering the matcher
bool duplicateFilter = false;	 // Check for duplicates of source or destination control points
bool originalFilter = false;     // Filter origial control points 
float overlapRatio = 0.66f;	 // User specified overlap
float overlapMargin = 0.00f;	 // Overlap margain
int overlapRatioPct = 66;	 // User specified overlap as a percentage
int overlapMarginPct = 0;	 // Overlap margain as a percentage
int centerFilter = 1;		 // allow match when N or less of the 2 points are in the center
bool alignLogging = false;       // When true, output from the alignment jobs are saved to individual log files.
bool globalLogging = false;      // when true the global sout object saves console output to a file. 
int  linearMatchLen = 100000;    // maximum interval between indexes in a pair.
int  rowSize = 0;		 // user provided number of images per row.  Does nothing if less than 4
int  rowSizeTolerance = 1;	 // When calculating impossible matches, assume +/- this many images
int  peakWidth = 1;		 // To reject spurious matches, use a window this wide for peak detection
string shootingPattern = "rltb"; // shooting pattern [rl|lr][tb|bt] right / left / top / bottom
int  shootingLR = 0;             // sign of COM[N++].x - COM[N].x mirrors sign.  For L-->R -, R-->L +, 0 Undefined 
int  shootingTB = 0;             // sign of COM[N++].y - COM[N].y mirrors sign.  For T-->B +. B-->T -, 0 Undefined
int  shootingAxis = 0;           // If Rows then Columns -1, if Columns then Rows, +1
                                 // 
bool doHomography = false;       // force final homography computation ( may be eliminatesd by trial homography eventually
float minDist = 3.0;		 // when validating homography, points below minDist error distance are "good"
float maxDist = 9.0;		 // when validating homography, points above maxDist error distance are "bad"
bool trialHomography = false;	 // Try to identify idealScalar via trial homography
int  trialHomSize = 100;         // each trial uses this range around the scalar being tested.
int  trialHomStep = 33;		 // advance the window this much.
bool loopAnalysis = false;       //
bool loopFilter = true;  	 // Identify bad / good pairs via transitive homography
float loopL0Threshold = 40;	 // Bad loops exceed this error on a per-hop basis.
float loopL1Threshold = sqrt( loopL0Threshold ); // Good matches will be below this thtreshold
float loopL2Threshold = sqrt( loopL1Threshold ); // Exceptional matches will be below this threshold
bool loopRepair = false;         // Use loop homography to find valid homography for bad pairs.
int loopRepairMaxBad = 1;	 // Maximum number of bad pairs per loop to try to repair 
                                 //
int  cellMinCP = 0;              // Target minimum CP's in a given cell
int  cellMaxCP = 5;              // Target max CP's in a given cell
int  cellSize = 100;             // default cell size: 100 x 100 pixels
int  cellCPDupDist = 0;          // minimum distance allowed between control points in a cell
int  cpPerPairMin = 15;          // want at least 15
int  cpPerPairMax = 200;         // want no more than 200
int  neighborsMin = 3;		 // want at least three neighbors
int  neighborsMax = 8;		 // want at most eight neighbors
int  featureMax = 10000;         // features to detect
int  detectorEnabledCnt = 0;     // Number of enabled feature detection methods
int  detectorDisabledCnt = 0;    // Number of enabled feature detection methods

bool detectorAkaze = false;	 // enable AKAZE
float akazeThreshold = 0.003;    // number of control points detected VERY sensitive to this value.  THis gives ~10000 / image.
bool akazeAutoTune = true;	 // Auto tune threshold to to get approximately the featureMax features

bool detectorOrb = false;	 // enable ORB

bool detectorBrisk = false;	 // enable BRISK

bool detectorBlob = false;	 // enable Blob detection

bool detectorCorner = false;	 // enable Corner detection

bool detectorGFTT = false;	 // GFTT = Good Features To Track ( Harris Corners, etc. ) 
double gfttQualityLevel = 0.01;  // No idea, but here they are.
double gfttMinDistance = 10;     //
int    gfttBlockSize = 3;        //
bool   gfttUseHarrisDetector = false;
double gfttK = 0.04;             //

bool detectorLine = false;	 // enable BinaryDescriptor Line detection
int  lineMaxDistThresh = 25;	 // MATCHES_DIST_THRESHOLD

bool detectorLSD = false;	 // enable LSDDetector Line Segment detection
int  lsdScale = 2;		 // scale factor for pyamid
int  lsdOctaves = 1;		 // number of pyramid levels

bool detectorSurf = false;	 // enable SURF detector
int  surfMinHessian = 400;	 //

bool detectorSift = false;	 // enable SIFT detector

bool detectorDenseSift = false;	 // enable Dense SIFT detector
int  denseSiftWindowSize = 5;    // Use a 5 pixel sliding window

bool detectorMasks = false;	 // enable use of masks based on shooting hints
bool listDetectors = false;	 // lists detectors during --help

bool detectorPto = true;	 // read control pionts from the PTO and act as if they were "detected"
bool ptoRewriteCPs = true;	 // control points learned from the PTO are removed from the live PTO ( and re-added before export )

bool huginFlipOut = false;       // when adding control points from cvfind, make ( xSize - x, ySize - y ) for some reason
bool huginFlipIn = false;        // when reading control point, make them ( xSize - x, ySize - y ) for some reason

std::string globalLogFN = "cvfind.log";
std::string outputPTO = "";
std::string inputPTO = "";
std::filesystem::path inputPTOPath;      // path object of the inpout PTO
std::string inputPTOImagePrefix = ".";	 // set to the relative path provided in the PTO.  Used if a referenced image is relative.
std::string runID = ""; 		 // Uniq ID for this run, e.g. a timestamp

std::vector<std::string> newCPlist; // List of control point config lines created during image processing

std::vector<int> mandatoryImages;	// list of allowed image indices for image processing
std::vector<int> forbiddenImages;	// list of banned image indices for image processing

std::vector<imageIndexPair> mandatoryPairs;	// list of allowed image pairs for image processing
std::vector<imageIndexPair> forbiddenPairs;	        // list of banned image pairs for image processing   

std::vector<std::vector<float>> statsScalars;  // byDistScalars[idx2 - idx1][ samples ]
std::vector<std::vector<float>> statsWeights;  // byDistWeights[idx2 - idx1][ samples ]
std::vector<float> statsScalarHint;          // weighted average

//
//  Globals for chippy and cvfind
//
std::vector<std::string> lines{};            // Holds each line from the source.pto
// std::vector<ptoImage> images{};              // Holds a class instance for each image
std::vector<ptoControlPoint> cplist{};            // Control point list parsed from the source PTO
std::vector<std::vector<int>> groupMembers;  // Each root group has a list of image[] indices
                                             // Shares structure for image pairs, valid indices [i][j]
                                             // where i > j

// std::vector<std::vector<ptoImgPair>> pairs( MAX_IMAGES, std::vector<ptoImgPair> (MAX_IMAGES));
std::vector<std::vector<ptoImgPair>> pairs;

//
// commonElements() - returns a vector with common elements from the specified vectors
//
template<typename L>
std::vector<L> commonElements( std::vector<L> vector1, std::vector<L> vector2 )
{
    std::vector<L> retval;
    
    if( vector1.size() == 0 )
    {
       if( loglevel > 8 ) cout << "commonElements(): passed an empty vector, returning an empty vector." << std::endl; 
       return retval;
    }

// Sort the vector 
    sort(vector1.begin(), vector1.end()); 
    sort(vector2.begin(), vector2.end()); 
  
    if( loglevel > 8 )
    {
       // Print the vector 
       cout << "commonElements(): Vector1 = { "; 
       for (int i = 0; i < vector1.size(); i++)
       { 
           cout << vector1[i];
           if( i + 1 < vector1.size() ) cout << ", ";
       }
       cout << " }" << endl; 
  
       // Print the vector 
       cout << "commonElements(): Vector2 = { "; 
       for (int i = 0; i < vector2.size(); i++)
       { 
           cout << vector2[i];
           if( i + 1 < vector2.size() ) cout << ", ";
       }
       cout << " }" << endl; 
    }
    
    // Initialise a vector 
    // to store the common values 
    // and an iterator 
    // to traverse this vector 
    vector<L> v(vector1.size() + vector2.size()); 
    typename vector<L>::iterator it, st; 
  
    it = set_intersection(vector1.begin(), 
                          vector1.end(), 
                          vector2.begin(), 
                          vector2.end(), 
                          v.begin()); 

    // Make a copy to return to the caller                      
    for (st = v.begin(); st != it; ++st) 
       retval.push_back( *st ); 
  
    if( loglevel > 8 )
    {
       cout << "commonElements(): retval = { "; 
       for (int i = 0; i < retval.size(); i++)
       { 
           cout << retval[i];
           if( i + 1 < retval.size() ) cout << ", ";
       }
       cout << " }" << endl; 
    }
    
    return retval;
 }



//
// Print a label with a black background with parameters as close as possible to cv::putText 
// 
void setLabel(cv::Mat& im, std::string label, cv::Point pt, int fontface, double scale, Scalar color, int thickness )
{
    int baseline = 0;
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), color, cv::FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(255,255,255), thickness, 8);
    return;
}

//
// Print a label with a black background with as close as possible calling as cv::putText 
// 
void setLabelCenter(cv::Mat& im, std::string label, cv::Point i_pt, int fontface, double scale, Scalar color, int thickness )
{
    int baseline = 0;
    cv::Point pt;
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    pt.x = i_pt.x - ( text.width / 2 );
    pt.y = i_pt.y - ( text.height / 2 );
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), color, cv::FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(255,255,255), thickness, 8);
    return;
}


 
// - - - - - - - - - - - - - - - - - - - - - - -
//
// checkHomography()
//
// Takes a ptoPointPad and returns a Homography modified ptoPointPad
//
// - - - - - - - - - - - - - - - - - - - - - - - 

Mat checkHomography( ptoPointPad *pp )
{
  Mat retval; //
  
  if( loglevel > 6 ) cout << " INFO: checkHomography():  Do Homography using " 
                     << pp->size() << " matches." << std::endl;

  if( ( true ) && pp->points1.size() > 5 && pp->points2.size() > 5 )
  {
     // Registered image will be resotred in imReg. 
     // The estimated homography will be stored in h. 
     Mat imReg, h;

     try
     {
        // Find homography
        h = findHomography( pp->points1, pp->points2, RANSAC );
        retval = h;

        // if the homography matrix is empty, we return.   Its not an error state, in that homography may just be bad.
        if (h.empty())
        {
            if( loglevel > 6 ) cout << " INFO: checkHomography(): cv::findHomography() returned empty matrix." << std::endl;
            return retval;
        }

     } // try
     catch(const cv::Exception& ex)
     {
        const char* err_msg = ex.what();
        if( loglevel > 6 ) cout << "ERROR: checkHomography(): cv::findHomography(): Caught cv::Exception: " << err_msg << std::endl;
        return retval;
     }    

     // Print estimated homography
     if( loglevel > 6 ) cout << " INFO: checkHomography(): Estimated homography : " << endl << h << endl;


     if( loglevel > 6 ) cout << " INFO: checkHomography(): Grade Homography in " 
                         << pp->size() << " matches." << std::endl;

     int hom_inliers = 0;
     int hom_outliers = 0;
     int hom_distcount[110];
     std::vector<std::vector<std::vector<int>>> cell;

     // Clear Homography Distance accumulators     
     for (int i = 0; i < 110; i++) hom_distcount[i]=0;
     
     // For each point compute the error distance from the reprojected point ( a cheat to look inside the RANSAC )
     for (int i = 0; i < pp->size(); i++)
     {
        /* ToDo: Remove: This often seen snippet is just wrong and failes to work with H.inv() 
         // cv::Mat H is filled by findHomography(points1, points2, CV_RANSAC, 3, mask)
         const double* H2 = h.ptr<double>();

         // Do Math Stuff ...
         float Hf[] = { (float)H2[0], (float)H2[1], (float)H2[2], (float)H2[3], (float)H2[4], (float)H2[5], (float)H2[6], (float)H2[7] }; 
         float ww = 1.f/(Hf[6]*pp->points1[i].x + Hf[7]*pp->points1[i].y + 1.f);
         float dx = (Hf[0]*pp->points1[i].x + Hf[1]*pp->points1[i].y + Hf[2])*ww - pp->points2[i].x;
         float dy = (Hf[3]*pp->points1[i].x + Hf[4]*pp->points1[i].y + Hf[5])*ww - pp->points2[i].y;
         float dist = (float)(dx*dx + dy*dy);        
        */
        
         // perspectiveTransform() accepts multiple points, but we call with a trivial array of size() = 1.
         vector<Point2f> pts1, pts2;
         pts1.push_back( pp->points1[i] );
         perspectiveTransform( pts1, pts2, h);
         float dx = pts2[0].x - pp->points2[i].x;
         float dy = pts2[0].y - pp->points2[i].y;
         float dist = (float)(dx*dx + dy*dy);        

         pp->distance[i] = dist;

         if( dist < maxDist ) // RANSAC reproj threshold 3 so 3 x 3 = 9
         { 
           if( loglevel > 8 ) cout << "Match: Good: ";
           hom_inliers++;
           hom_distcount[ (int)((float)dist * (float)10) ]++;
         }
         else 
         { 
           if( loglevel > 8 ) cout << "Match:  Bad: ";
           hom_outliers++;
           hom_distcount[ 101 ]++;
         }
      }
     
      if( loglevel > 6 ) cout << " INFO: Inliers:" << hom_inliers << " outliers:" << hom_outliers << std::endl; 

   } // If enough points for homography ( 5 ) 

   return retval;   
    
} // checkHomography()



// - - - - - - - - - - - - - - - - - - - - - - -
//
// ppFilterScalar()
//
// Takes a ptoPointPad and returns a new ptoPointPad with points matching a specific scalar range. 
//
// - - - - - - - - - - - - - - - - - - - - - - - 

ptoPointPad ppFilterScalar( ptoPointPad *pp, float sMin, float sMax )
{
  ptoPointPad retval;
  
  // copy essential stuff from the original
  retval.prime( pp );
  
//  bool debug = true;
  
  if( loglevel > 6 ) cout << " INFO: ppFilterScalar():  Filtering " 
                     << pp->size() << " matches for sMin=" << sMin << " sMax=" << sMax << std::endl;

  if( ( true ) && pp->size() > 0 )
  {
     int xSize = images[pp->idx1].img.cols;
     int ySize = images[pp->idx2].img.rows;
     
     // Loop backwards so we can delete non-matching pairs from the end of
     // the vector without disturbing the indices of earlier entries
     for( int i = pp->size() - 1; i >= 0; i-- )
     {
        if( loglevel > 8 ) cout << " INFO: ppFilterScalar(): Examining match index " << i
                         << " of " << pp->size() << " matches for sMin=" << sMin << " sMax=" << sMax 
                         << " retval.size()= " << retval.size() << std::endl;

        float x1, y1, x2, y2, dx, dy, scalar, slope;
  
        x1 = pp->points1[i].x;
        x2 = pp->points2[i].x;
        y1 = pp->points1[i].y;
        y2 = pp->points2[i].y;

        dx = x2 - x1;
        dy = y2 - y1;

        scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

        if( dx == 0 )
        {
           slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
        }
        else
        {	
           slope=dy / dx;
        }

        // Any matches not meeting the filter are removed
        if( scalar >= sMin && scalar <= sMax )
        {
             if( loglevel > 8 ) cout << " INFO: ppFilterScalar(): erasing match index " << i << " of " << retval.size() << std::endl; 
           retval.points1.push_back(   pp->points1[i] );
           retval.points2.push_back(   pp->points2[i] );
           retval.pointType.push_back( pp->pointType[i] );
           retval.lineIdx.push_back(   pp->lineIdx[i] );
           retval.distance.push_back(  pp->distance[i] );
             if( loglevel > 8 ) cout << " INFO: ppFilterScalar(): erased match index " << i << " of " << retval.size() << std::endl; 
        }

      } // For each point in the pp

   } // If enough points for filtering 

   if( loglevel > 6 ) cout << " INFO: ppFilterScalar(): Returning " << retval.size() << " of " << pp->size()
                    << " Matches with Scalars from " << sMin << " to " << sMax << std::endl; 

   return retval;   
    
} // ppFilterScalar()


// - - - - - - - - - - - - - - - - - - - - - - -
//
// ppFilterDistance()
//
// Takes a ptoPointPad and returns a modified ptoPointPad with points matching a specific homography distance range. 
//
// - - - - - - - - - - - - - - - - - - - - - - - 

ptoPointPad ppFilterDistance( ptoPointPad *pp, float sMin, float sMax )
{
  ptoPointPad retval;
  
  // copy essential stuff from the original
  retval.prime( pp );
  
//  bool debug = true;
  
  if( loglevel > 6 ) cout << " INFO: ppFilterDistance():  Filtering " 
                     << pp->size() << " matches for sMin=" << sMin << " sMax=" << sMax << std::endl;

  if( ( true ) && pp->size() > 0 )
  {
     // Loop backwards so we can delete non-matching pairs from the end of
     // the vector without disturbing the indices of earlier entries
     for( int i = pp->size() - 1; i >= 0; i-- )
     {
        // Any matches not meeting the filter are removed
        if( pp->distance[i] >= sMin && pp->distance[i] <= sMax )
        {
             if( loglevel > 8 ) cout << " INFO: ppFilterScalar(): erasing match index " << i << " of " << retval.size() << std::endl; 
           retval.points1.push_back(   pp->points1[i] );
           retval.points2.push_back(   pp->points2[i] );
           retval.pointType.push_back( pp->pointType[i] );
           retval.lineIdx.push_back(   pp->lineIdx[i] );
           retval.distance.push_back(  pp->distance[i] );
             if( loglevel > 8 ) cout << " INFO: ppFilterScalar(): erased match index " << i << " of " << retval.size() << std::endl; 
        }

      } // For each point in the pp

   } // If enough points for filtering 

   if( loglevel > 6 ) cout << " INFO: ppFilterDistance(): Returning " << retval.size() << " of " << pp->size()
                    << " Matches with Scalars from " << sMin << " to " << sMax << std::endl; 

   return retval;   
    
} // ppFilterDistance()


// - - - - - - - - - - - - - - - - - - - - - - -
//
// ppFilterDetectors()
//
// Takes a ptoPointPad and returns a modified ptoPointPad with points matching a specific homography distance range. 
//
// - - - - - - - - - - - - - - - - - - - - - - - 

ptoPointPad ppFilterDetectors( ptoPointPad *pp, std::vector<int> i_dlist )
{
  ptoPointPad retval;
  
  // copy essential stuff from the original
  retval.prime( pp );
  
  
  if( loglevel > 6 ) cout << " INFO: ppFilterDetectors():  Filtering " 
                     << pp->size() << " matches for detectors " << i_dlist << std::endl;

  if( ( true ) && pp->size() > 0 )
  {
     // Loop backwards so we can delete non-matching pairs from the end of
     // the vector without disturbing the indices of earlier entries
     for( int i = pp->size() - 1; i >= 0; i-- )
     {
        // Any matches not meeting the filter are removed
        if( std::find(i_dlist.begin(), i_dlist.end(), pp->pointType[i]) != i_dlist.end() )
        {
             if( loglevel > 6 ) cout << " INFO: ppFilterScalar(): erasing match index " << i << " of " << retval.size() << std::endl; 
           retval.points1.push_back(   pp->points1[i] );
           retval.points2.push_back(   pp->points2[i] );
           retval.pointType.push_back( pp->pointType[i] );
           retval.lineIdx.push_back(   pp->lineIdx[i] );
           retval.distance.push_back(  pp->distance[i] );
             if( loglevel > 6 ) cout << " INFO: ppFilterScalar(): erased match index " << i << " of " << retval.size() << std::endl; 
        }

      } // For each point in the pp

   } // If enough points for filtering 

   if( loglevel > 6 ) cout << " INFO: ppFilterDetectors(): Returning " << retval.size() << " of " << pp->size()
                    << " Matches with detectors " << i_dlist << std::endl; 

   return retval;   
    
} // ppFilterDetectors()


// Invocation for a single detector passed as an int
ptoPointPad ppFilterDetectors( ptoPointPad *pp, int i_type )
{
    std::vector<int> dlist;
    dlist.push_back( i_type );
    return ppFilterDetectors( pp, dlist );
}


//
// transitiveChains( idx1, idx2, ... ) - computes the transitive images between images based on their homography
// only 
// 
std::vector<std::vector<int>> transitiveChains( int idx1, int idx2, int i_depth )
{
    std::vector<std::vector<int>> retval;
    
    // Add trivial chain { idx1, idx2 }
    std::vector<int> lchain, rchain, chain;
    
    lchain.push_back( idx1 );
    rchain.push_back( idx2 );
    
    if( loglevel > 8 ) cout << "transitiveChains(): appendVector():" << lchain << " + " << chain << " + " << rchain << std::endl;
    cout.flush();
    
    appendVector( chain, lchain, rchain );
    retval.push_back( chain );

    if( loglevel > 8 ) cout << "transitiveChains(): Adding Chain : " << chain << " to retval " << std::endl;
    
    if( i_depth == 0 ) return retval;

    // Add chains for each common neighbor { idx1, (neighbor), idx2 }
    vector<int> common = commonElements( images[idx1].candidates, images[idx2].candidates );

    if( loglevel > 8 ) cout << "transitiveChains(): commonElements ( " << images[idx1].candidates 
                       << ", " << images[idx2].candidates << " ) = " << common << std::endl;
    cout.flush();

    for( int i = 0; i < common.size(); i++ )
    {

       if( loglevel > 8 ) cout << "transitiveChains(): common[" << i << "] = " << common[i] << std::endl;
       cout.flush();

       vector<int> chain;
       appendVector( chain, lchain );
       chain.push_back( common[i] );
       appendVector( chain, rchain );
       retval.push_back( chain );
       if( loglevel > 8 ) cout << "transitiveChains(): Adding Chain : " << chain << " to retval " << std::endl;
       
    }

    // Add chains for the central pair { (neighbor), idx1, idx2, (neighbor) } for sets of distinct image indices
    vector<int> lbros = images[idx1].candidates;
    vector<int> rbros = images[idx2].candidates;
    for( int l = 0; l < lbros.size(); l++ )
       for( int r = 0; r < rbros.size(); r++ )
       {
          // make sure (l), idx1, idx2, (r) are distinct
          if( lbros[l] != rbros[r] && lbros[l] != idx1 && lbros[l] != idx2 && rbros[r] != idx1 && rbros[r] != idx2 )
          {
              vector<int> chain;
              chain.push_back( lbros[l] );
              chain.push_back( idx1 );
              chain.push_back( idx2 );
              chain.push_back( rbros[r] );
              
              retval.push_back( chain );
              if( loglevel > 8 ) cout << "transitiveChains(): Adding Chain : " << chain << " to retval " << std::endl;
          }

       }


    if( loglevel > 8 ) cout << "transitiveChains(): retval.size()" << retval.size() << std::endl;
    cout.flush();
    
    return retval;
}


//
// validChain() - checks to see that each link in the chain has a valid forward homography
//                if indices are ( high --> low ), H.inv() for ( low --> high ) is used 
//

// Check for a valid chain of homography matrices
bool validChain( vector<Mat> i_h )
{

    if( loglevel > 8 ) cout << "validChain(): Called with i_h.size() = " << i_h.size() << std::endl;

    // A chain has to have more than 1 element
    if( i_h.size() < 2 ) return false;
    bool retval = true;
    for( int i = 0; retval == true && i < i_h.size(); i++ )
    {
       if( i_h[i].empty() )
       {
          if( loglevel > 8 ) cout << "validChain(): i_h[" << i << "].empty() == true." << std::endl;
          retval = false;
       }         
    } // For each link in the chain

    if( loglevel > 8 && retval ) cout << "validChain(): Chain is valid." << std::endl;
    
    return retval;
}


// Check for homography matrices, and when checkMinCP = true that the minimum control points between pairs are present.
bool validChain( vector<int> i_chain, bool checkMinCP )
{

    if( loglevel > 8 ) cout << "validChain(): Called with chain " << i_chain << std::endl;

    // A chain has to have more than 1 element
    if( i_chain.size() < 2 ) return false;
    bool retval = true;
    for( int i = 0; retval == true && i + 1 < i_chain.size(); i++ )
    {
       if(    pairs[min(i_chain[i], i_chain[i+1])][max(i_chain[i], i_chain[i+1])].h.empty() ) 
       {
          if( loglevel > 8 ) cout << "validChain(): Pair ( " << i_chain[i] << ", " << i_chain[i+1] 
                             << " ) in " << i_chain << " has no homography." << std::endl;
          retval = false;
       }
       else
       if( checkMinCP && pairs[min(i_chain[i], i_chain[i+1])][max(i_chain[i], i_chain[i+1])].points.size() < cpPerPairMin )
       {
          if( loglevel > 8 ) cout << "validChain(): Pair ( " << i_chain[i] << ", " << i_chain[i+1] 
                             << " ) in " << i_chain << " has only " 
                             << pairs[min(i_chain[i], i_chain[i+1])][max(i_chain[i], i_chain[i+1])].points.size() 
                             << " CPs." << std::endl;
          retval = false;
       }         
    } // For each link in the chain

    if( loglevel > 8 && retval ) cout << "validChain(): Chain " << i_chain << " is valid." << std::endl;
    
    return retval;
}

// Checks only for homography matrices being available
bool validChain( vector<int> i_chain )
{
   return validChain( i_chain, false );
}


//
// transitiveError()
// computes the error in transitive homography for an image chains of arbitrary length.
// Returns a point representing the error in x and y coordinates.  To evaluate a loop
// provide a chain beginning and ending at the same image.  e.g. A, B, C, A
// the chain must be valid ( validChain() == true ) or the returned value is invalid.
//
// Normally the chain provided forms a loop.  However the function does not require
// this.   In this instance the "error" returned in just the delta between starting and
// ending point.
//
// The main function accepts an vector of cv::Mat objects which encapsulate the
// homography matrix chain to be teststed.   The other overloaded functions supply the
// chain as a list of images.
//

Point2f transitiveError( vector<Mat> i_h, Point2f i_test )
{

    if( loglevel > 8 ) cout << "transitiveError() called with i_h.size()= " << i_h.size() << " for test point " << i_test << std::endl;
    Point2f retval;
    
    // Pick a test point in the center of the image
    Point2f test = i_test;
    Point2f reference = i_test;
    
    // sequentially transform the test point provided with each homography matrix
    for( int i = 0; i < i_h.size(); i++ )
    {
        if( i_h[i].empty() )
        {
           cout << " ERROR: transitiveError(): i_h[" << i << "].empty() " << std::endl;
           return retval;
        }

        // perspectiveTransform() accepts multiple points, but we call with a trivial array of size() = 1.
        vector<Point2f> pts1, pts2;
        pts1.push_back( test );
        perspectiveTransform( pts1, pts2, i_h[i]);
        test = pts2[0];   
    }
    // If any of this worked out properly, test should approximately be reference.
    retval = reference - test;
        
    return retval;
}


// image chain based invocation.   Constructs a chain of homography matrices and calls
// its sibling function above.
Point2f transitiveError( vector<int> i_chain )
{
    Point2f retval;
    std::vector<cv::Mat> hchain;
    
    // Pick a test point in the center of the image
    Point2f test( images[i_chain[0]].img.cols / 2, images[i_chain[0]].img.cols / 2 );
    
    for( int i = 0; i + 1 < i_chain.size(); i++ )
    {
        Mat h;

        // find the homography matrix from image 1 to image 2, use its inverse if idx1 > idx2
        if( ! pairs[i_chain[i]][i_chain[i+1]].h.empty() ) h = pairs[i_chain[i]][i_chain[i+1]].h;
        if( ! pairs[i_chain[i+1]][i_chain[i]].h.empty() ) h = pairs[i_chain[i+1]][i_chain[i]].h.inv();

        if( h.empty() ) // This should never happen
        {
           cout << " ERROR: transitiveError() called with invalid chain { " << i_chain << " } at index " << i
           << ", image pair: ( " << i_chain[i] << ", " << i_chain[i+1] << " )" << std::endl;
           return retval;
        }

        hchain.push_back( h );
    }
    // If any of this worked out properly, test should approximately be reference.
    retval = transitiveError( hchain, test );
    
    return retval;
}

//
// This invocation checks all subchains, returning the worst error found among the provided chain and all
// valid sub-chains ( which are turned into loops ).   While it can be called without the chain forming
// a loop, the results are not guranteed to be meaningful.
//
Point2f transitiveErrorSub( vector<int> i_chain )
{

    if( loglevel > 4 ) cout << "transitiveErrorSub(" << i_chain << ") : " << std::endl;

    std::vector<Point2f> errors;
    std::vector<float> scalars;
    std::vector<std::vector<int>> chains;

    // If chain is trivial, e.g. A-->B-->A or A-->B-->C-->A there are no subchains
    if( i_chain.size() < 5 ) return transitiveError( i_chain );
    
    // if chain is longer, e.g. slice it up into smaller chains and test each valid chain.
    // A-->B-->C-->D-->A     ==> A-->B-->C-->A, B-->C-->D-->B, C-->D-->A-->C
    // A-->B-->C-->D-->E-->A ==> A-->B-->C-->D-->A, B-->C-->D-->E-->B, C-->D-->E-->A-->C
    //                           A-->B-->C-->A, B-->C-->A-->B, C-->D-->E-->C, D-->E-->A-->D   
    
    
    for( int l = 3; l < i_chain.size(); l++ )  // Length of fragment
       for( int f = 0; f < i_chain.size() - ( l - 1 ); f++ ) // Each fragment of length l
       {
           vector<int> chain;
           Point2f error;
           float scalar;
           for( int i = 0; i < l; i++ )
              chain.push_back( i_chain[f+i] );
           chain.push_back( i_chain[f] );  // make chain into a loop
           // Its possible a subchain is not valid since its turned into a loop
           // so for loop A-->B-->C-->D-->E-->A, subchain B-->C-->D-->B is not necessarily so 
           if( validChain( chain ) )
           {
               error = transitiveError( chain );
               scalar = sqrt( ( error.x * error.x ) + ( error.y * error.y ) );
               errors.push_back( error );
               scalars.push_back( scalar );
               chains.push_back( chain );  

              if( loglevel > 4 ) cout << "transitiveErrorSub(" << i_chain << ") : " << chain 
                                      << " Error: " << scalar / (float)chain.size() << std::endl;
           }   
       }
       
    std::vector<float> sortIndex = scalars;
    std::vector<float> sortScalars = scalars;
    std::vector<Point2f> sortErrors = errors;
    std::vector<std::vector<int>> sortChains = chains;

    if( sortIndex.size() > 1 ) sortScalars = sortVecAByVecB( scalars, sortIndex );
    if( sortIndex.size() > 1 ) sortErrors = sortVecAByVecB( errors, sortIndex );
    if( sortIndex.size() > 1 ) sortChains = sortVecAByVecB( chains, sortIndex );

    // std::reverse(sortScalar.begin(),sortScalar.end());
    // std::reverse(sortWeight.begin(),sortWeight.end());    

     if( loglevel > 4 ) cout << "transitiveErrorSub(" << i_chain << ") : " << sortScalars.size() << " Sub-chains, " 
                                      << " Worst: " << sortChains.back()
                                      << " Error: " << sortScalars.back() / (float)sortChains.back().size() << std::endl;

    // return the worst error result   
    return sortErrors.back();
}



//
// Uses the homography matrices from trials to identify the "best" combination of trials between all
// links in the chain.   The idea here is that if a good homography is not found via pair wise trial
// homography, it can be identified by trying various combinations of trials.
// if i_useValid is true, the final homography stored in the pair is used rather than the trial,
// if ALL are valid, the error results should be identical to that reported by transitiveError()
// i_minValid requires at least this many already valid pairs for a homography loop to be evaluated.
//

std::vector<ptoTrialPermutation> transitiveTrialError( vector<int> i_chain, bool i_useValid, int i_maxBad )
{

    std::vector<int> index; // The current "counters" for each index into the point paids for a pair in the chain

    std::vector<ptoTrialPermutation> retval; // a list of all permutations to return to the caller    

    // Pick a test point in the center of the image
    Point2f test( images[i_chain[0]].img.cols / 2, images[i_chain[0]].img.cols / 2 );
    
    if( loglevel > 8 ) cout << "transitiveTrialError( " << i_chain << " ): " << std::endl;

    // make a vector of indexes into each pair's homography trials.   
    for( int i = 0; i + 1 < i_chain.size(); i++ ) index.push_back( 0 );
    

    int validCnt = 0; // Number of valid pairs ( with only one homography to test )
    int permCnt = 1;  // Number of permutations ( all of them with 
    // Determine how many degrees of freedom for bad pairs       
    for( int q = 0; q < index.size(); q++ )
    {
        // short hand for low and high indexes into pair.
        int il = min(i_chain[q], i_chain[q+1]);
        int ih = max(i_chain[q], i_chain[q+1]);
        if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
        {
            validCnt++;
            if( ! i_useValid )
            {
                 permCnt *= pairs[il][ih].trialPointPad.size();
            }
        }
        else
        {
            permCnt *= pairs[il][ih].trialPointPad.size();
        }

    }
    
    if( loglevel > 6 ) cout << "transitiveTrialError( " << i_chain << " ): Validated Pairs: " << validCnt << " Permutations: " << permCnt << std::endl;

    if( index.size() - validCnt > i_maxBad )
    {
       if( loglevel > 6 ) cout << "transitiveTrialError( " << i_chain << " ): Validated Pairs: " << validCnt << " Bad Pairs: " << index.size() - validCnt
            << " >  " << i_maxBad << " allowed.  Returning no permutations." << std::endl;
       return retval;
    }

    // starting with index { 0,0,0,0... } itterate through all combinations of trial point pads, grab their H matrix
    // and obtain the errors for each one.  While thwe final index has not been ovderflowed, increment and carry the others
    // to permute all combinations of trialPointPads.
    bool done = false;
    while( ! done )
    {
        if( loglevel > 8 ) cout << "transitiveTrialError( " << i_chain << " ): trial: " << index << ": " << std::endl; 

        std::vector<cv::Mat> hchain; 
        // Build out the homography chain
        for( int q = 0; q < index.size(); q++ )
        {            

            // short hand for low and high indexes into pair.
            int il = min(i_chain[q], i_chain[q+1]);
            int ih = max(i_chain[q], i_chain[q+1]);

            if( index[q] >= pairs[il][ih].trialPointPad.size() )
            {
                cout << "FATAL: transitiveTrialError(): index[" << q << "] >= "
                     << " pairs[" << il << "][" << ih << "].trialPointPad.size() == "
                     << pairs[il][ih].trialPointPad.size();
                cout.flush();
            }     

            // get the homograph matric for the pair, if the chain referes to a "backwards" pair, invert the
            // the homography matrix, then add it to the homography matrix chain.
            cv::Mat h, trialh;
            
            // if the pair is already valid, then skip using the trials and use the final homography
            // ( later we skip ittrerating through the trials in the index incrementing / overflow logic )
            if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
            {
                trialh = pairs[il][ih].h;
            }
            else
            {    
               trialh = pairs[il][ih].trialPointPad[index[q]].h;
            }

            // use the inverse of the homography matrix if projecting from a higher to lower index image
            if( i_chain[q] > i_chain[q+1] )
            {  
               h = trialh.inv();
            }
            else
            {
               h = trialh;
            } 
            
            hchain.push_back( h );
        }

        // So at this point we have a homography chain.  I guess.  Or we crashed.
        
        ptoTrialPermutation p;

/*        
     std::vector<int> chain;          // An image chain ( list indices of images[] array ) 
     std::vector<int> trialChain;     // The indices of each trialPointPad for each image pair in the chain
     std::vector<cv::Mat> hChain;     // The homography chain
     Point2f errorPoint; // the error values returned by transitiveError( )
     float error;	      // the scalar magnitide of the error.
*/
        
        p.chain = i_chain;
        p.trialChain = index;
        p.hChain = hchain;
        p.errorPoint = transitiveError( hchain, test );
        p.error = sqrt( ( p.errorPoint.x * p.errorPoint.x ) + ( p.errorPoint.y * p.errorPoint.y ) );

        retval.push_back( p );

        if( loglevel > 8 ) cout << "transitiveTrialError( " << i_chain << " ): trial: " << index 
                           << ": transitiveError() = " << p.errorPoint << " Scalar: " << p.error << std::endl; 


        // Increment 1st index, cascade carry when an index overflows
        for( int q = 0; q < index.size(); q++ )
        {
            // short hand for low and high indexes into pair.
            int il = min(i_chain[q], i_chain[q+1]);
            int ih = max(i_chain[q], i_chain[q+1]);

            index[q] ++; // increment the current index.  Then check for overflow...

            // If the index has not overflowed and the pair is not valid then increment the index 
            if( index[q] < pairs[il][ih].trialPointPad.size()
                && ! ( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
              )
            {
               break; // No overflow
            }
            else
            {
                index[q] = 0;  // clear
                if( q+1 < index.size() )
                { 
                   index[q+1] ++; // carry
                }
                else
                {
                   done = true;  // overflowed last index
                }
            }
        }
    
    } // while... not done
     
    return retval;

}




/*
// image chain based invocation
Point2f transitiveError( vector<int> i_chain )
{
    Point2f retval;
    
    // Pick a test point in the center of the image
    Point2f test( images[i_chain[0]].img.cols / 2, images[i_chain[0]].img.cols / 2 );
    Point2f reference = test;
    
    for( int i = 0; i + 1 < i_chain.size(); i++ )
    {
        Mat h;
        // find the homography matrix from image 1 to image 2, use its inverse if idx1 > idx2
        if( ! pairs[i_chain[i]][i_chain[i+1]].h.empty() ) h = pairs[i_chain[i]][i_chain[i+1]].h;
        if( ! pairs[i_chain[i+1]][i_chain[i]].h.empty() ) h = pairs[i_chain[i+1]][i_chain[i]].h.inv();

        if( h.empty() )
        {
           cout << " ERROR: transitiveError() called with invalid chain { " << i_chain << " } at index " << i
           << ", image pair: ( " << i_chain[i] << ", " << i_chain[i+1] << " )" << std::endl;
           return retval;
        }

    vector<Point2f> pts1, pts2;
    pts1.push_back( test );
    perspectiveTransform( pts1, pts2, h);
    test = pts2[0];   

    }
    // If any of this worked out properly, test should approximately be reference.
    retval = reference - test;
    
    return retval;
}

*/



// - - - - - - - - - - - - - - - - - - - - - - -
//
// Check that size of point clouds are approximately the same.  Method 2.
//
// - - - - - - - - - - - - - - - - - - - - - - - 

std::string badOverlapCause( ptoPointPad pp )
{

     if( pp.idx1 == pp.idx2 ) cout << "ERROR: badOverlapCause() called with invalid point pad.  Crashing." << std::endl;

     if( loglevel > 9 ) cout << " INFO: badOverlapCause() called for image pair (" << pp.idx1 << ", " << pp.idx2 
                               << ") with " << pp.size() << " matches." << std::endl; 

     // Compute overlap "stuff" 
     float ovlXpct, ovlXsense, ovlXdelta, ovlYpct, ovlYsense, ovlYdelta, xSize, ySize;

     pp.refresh();
      
     xSize = images[pp.idx1].img.cols;
     ySize = images[pp.idx1].img.rows;
     
     Rect overlap = pp.rect1 & pp.rect2;
     if( pp.size() > 0 )
     {
//         if( loglevel > 3 ) IMGLOG << " INFO: badOverlapCause(): Point Clouds: rect1: " << RECT2STR(pp.rect1) 
//                              << " rect2: " << RECT2STR(pp.rect2) << std::endl;
//         if( loglevel > 3 ) IMGLOG << " INFO: badOverlapCause(): Point Clouds: Overlap: " << RECT2STR(overlap) << std::endl;    
     }

     std::string badOverlapCause = "";

     // protect from divide by zero
     if ( pp.size() > 0 && ( pp.rect1.width + pp.rect2.width ) != 0 && ( pp.com2.x - pp.com1.x ) != 0 )
     { 
        // Compute overlap percentage, direction sense, and delta for each axis
         
        ovlXpct = ( 100 * abs( pp.rect1.width - pp.rect2.width )) / ( pp.rect1.width + pp.rect2.width ); 
        ovlXsense = (float)( abs( pp.com2.x - pp.com1.x ) / ( pp.com2.x - pp.com1.x ) * shootingLR );
        ovlXdelta = (float)( ( pp.com2.x - pp.com1.x ) );
        ovlYpct = ( 100 * abs( pp.rect1.height - pp.rect2.height )) / ( pp.rect1.height + pp.rect2.height ); 
        ovlYsense = (float)( abs( pp.com2.y - pp.com1.y ) / ( pp.com2.y - pp.com1.y ) * shootingTB );
        ovlYdelta = (float)( ( pp.com2.y - pp.com1.y ) );

        // ppoint cloud size should be nearly identical for properly overlapped images
/*
        if( loglevel > 3 ) IMGLOG << " INFO: Point Cloud X Size Difference %: " << ovlXpct 
                             << " Delta: " << ovlXdelta << " Expected Sign: " << shootingLR <<" Sense: " << ovlXsense;   
        if( loglevel > 3 && ( idx2 - idx1 ) > ( rowSize + rowSizeTolerance ) ) sout << " (Supressed) ";
        if( loglevel > 3 ) sout << std::endl; 

        if( loglevel > 3 ) IMGLOG << " INFO: Point Cloud Y Size Difference %: " << ovlYpct 
                             << " Delta: " << ovlYdelta << " Expected Sign: " << shootingTB <<" Sense: " << ovlYsense;   
        if( loglevel > 3 && ( idx2 - idx1 ) < ( rowSize - rowSizeTolerance ) ) sout << " (Supressed) ";
        if( loglevel > 3 ) sout << std::endl; 
*/
     
        // ToDo : Protect from div by zero
        
        // Check for differences in width / height of the point cloud

        if( ovlXpct > 2 )
        {
           badOverlapCause += "Point Cloud Width Differs By " + to_string( (int)ovlXpct ) + "%.  " ;
        }

        if( ovlYpct > 2 )
        {
           badOverlapCause += "Point Cloud Height Differs By " + to_string( (int)ovlYpct ) + "%.  " ;
        }
     
        // Check that overlap is consistent with shooting axis and pattern.
        
        // shootingAxis < 0 implies sequential images are in horizontal rows

        if( shootingAxis < 0 && ovlXsense < 0 && ( pp.idx2 - pp.idx1 ) == 1 )
        {
           badOverlapCause +=   "X Overlap Sense Is: " + to_string( ovlXsense ) \
                               + " But should be " + to_string( shootingLR )  + "; ";
        }
        
        if( shootingAxis < 0 && ovlYsense < 0 && ( pp.idx2 - pp.idx1 ) > ( rowSize - rowSizeTolerance ) )
        {
           badOverlapCause +=   "Y Overlap Sense Is: " + to_string( ovlYsense ) \
                              + " But should be " + to_string( shootingTB )  + "; ";
        }
        
        // shootingAxis > 0 implies sequential images are in vertical columns

        if( shootingAxis > 0 && ovlXsense < 0 && ( pp.idx2 - pp.idx1 ) > ( rowSize - rowSizeTolerance ) )
        {
           badOverlapCause +=   "X Overlap Sense Is: " + to_string( ovlXsense ) \
                               + " But should be " + to_string( shootingLR )  + "; ";
        }
        
        if( shootingAxis > 0 && ovlYsense < 0 && ( pp.idx2 - pp.idx1 ) == 1 )
        {
           badOverlapCause +=   "Y Overlap Sense Is: " + to_string( ovlYsense ) \
                              + " But should be " + to_string( shootingTB )  + "; ";
        }


        // Check that point clouds vertical and horizontal delta is appropriate.
        // The difference between the x or y coordinates for horizontal or vertical adjacent images
        // should be approximately [ xSize or ySize ] * ( 1 - overlapRatio )

        // shootingAxis < 0 implies sequential images are in horizontal rows

        if(    shootingAxis < 0
            && ( ! shootingLR == 0 ) 
            && abs( ovlXdelta ) < xSize * ( 0.9f - overlapRatio )
            && ( pp.idx2 - pp.idx1 ) == 1 )
        {
           badOverlapCause +=   "X abs(delta) Is: " + to_string( abs( ovlXdelta ) ) \
                               + " But should be > " + to_string( xSize * ( 0.9f - overlapRatio ) ) + "; ";
        }
        
        if(      shootingAxis < 0
              && ( ! shootingTB == 0 ) 
              && abs( ovlYdelta ) < ySize * ( 0.9f - overlapRatio ) 
              && ( pp.idx2 - pp.idx1 ) > ( rowSize - rowSizeTolerance ) )
        {
           badOverlapCause +=   "Y abs(delta) Is: " + to_string( abs( ovlYdelta ) ) \
                              + " But should be > " + to_string( ySize * ( 0.9f - overlapRatio ) ) + "; ";
        }

        // shootingAxis > 0 implies sequential images are in vertical columns

        if(    shootingAxis > 0 
            && ( ! shootingLR == 0 ) 
            && abs( ovlXdelta ) < xSize * ( 0.9f - overlapRatio )
            && ( pp.idx2 - pp.idx1 ) > ( rowSize - rowSizeTolerance ) )
        {
           badOverlapCause +=   "X abs(delta) Is: " + to_string( abs( ovlXdelta ) ) \
                               + " But should be > " + to_string( xSize * ( 0.9f - overlapRatio ) ) + "; ";
        }
        
        if(     shootingAxis > 0 
              && ( ! shootingTB == 0 ) 
              && abs( ovlYdelta ) < ySize * ( 0.9f - overlapRatio ) 
              && ( pp.idx2 - pp.idx1 ) == 1 )
        {
           badOverlapCause +=   "Y abs(delta) Is: " + to_string( abs( ovlYdelta ) ) \
                              + " But should be > " + to_string( ySize * ( 0.9f - overlapRatio ) ) + "; ";
        }

    
        if( overlap.width > 0 || overlap.height > 0 )
        {
           // retval = true;
           badOverlapCause +=   "Control points cover same coordinates.  Overlapping by " + std::string( RECT2STR(overlap) ) + ".  "; 
        } 

        // Give more info... as 
        if( badOverlapCause.length() > 0 )
           if( ( pp.idx2 - pp.idx1 ) == 1 )        
              badOverlapCause += " For sequential pair (" + to_string( pp.idx1 ) + " --> " + to_string( pp.idx2 ) \
                              + ") with shooting pattern " + shootingPattern + ".  ";
           else
              badOverlapCause += " For non-sequential pair (" + to_string( pp.idx1 ) + " --> " + to_string( pp.idx2 ) \
                              + ") with shooting pattern " + shootingPattern + ".  ";

     
     }  // If ( we can avoid a FP error )
     else
     {
        // At this point we have invalid cloud points
        badOverlapCause +=   "CP Cloudpoint(s) Empty. ";
     }

     if( badOverlapCause.size() > 0 && loglevel > 9 ) cout << " INFO: badOverlapCause(): Image pair (" << pp.idx1 << ", " << pp.idx2 
                               << "):  Bad Overlap Cause: " << badOverlapCause << std::endl; 
                                     
   return badOverlapCause;
}


//
// Self test routine: check dynamic bindings for OpenCV and possibly other things.
// used by --test option and ultimately in the "check" Makefile target
//
bool testSuccessful()
{
   sout << " TEST: Binding AKAZE ... " << std::endl;
   Ptr<Feature2D> akaze = AKAZE::create();
   sout << " TEST: Binding ORB ... " << std::endl;
   Ptr<Feature2D> orb = ORB::create( featureMax );
   sout << " TEST: Binding BRISK ... " << std::endl;
   Ptr<Feature2D> brisk = BRISK::create( );
   #if defined(__OPENCV_XFEATURES2D_HPP__)
   sout << " TEST: Binding SURF ... " << std::endl;
     Ptr<Feature2D> surf = SURF::create(surfMinHessian);
   #endif
   sout << " TEST: Binding SIFT ... " << std::endl;
   Ptr<Feature2D> sift = SIFT::create();
   sout << " TEST: Binding SimpleBlobDetector ... " << std::endl;
   SimpleBlobDetector::Params blobParams;
   Ptr<SimpleBlobDetector> blob = SimpleBlobDetector::create(blobParams);
   sout << " TEST: Binding BinaryDescriptor ... " << std::endl;
   Ptr<BinaryDescriptor> line = BinaryDescriptor::createBinaryDescriptor(  );
   sout << " TEST: Binding LSDDetector ... " << std::endl;
   Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
   sout << " TEST: All Tests Passed" << std::endl;
   // ( Or we just exploded with a dynamic binding error )
   return true;
}

//
// detectAKAZE()
//
std::string detectAKAZE( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Detect AKAZE features and compute descriptors.  If not already done before.
  // c.f. https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
  //

  Ptr<Feature2D> akaze = AKAZE::create();
  // Alternative with parameter passing:
  // Ptr<Feature2D> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, akazeThreshold, 4, 4, KAZE::DIFF_PM_G2);
  if( detectorAkaze == true )
  {  
     
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection AKAZE" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::AKAZE].size() == 0 )
        akaze->detectAndCompute(images[idx1].imgGray, overlapMask, images[idx1].keypoints[detectorType::AKAZE], images[idx1].descriptors[detectorType::AKAZE]);
        
     images[idx1].unlock();

     BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 AKAZE" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::AKAZE].size() == 0 )
        akaze->detectAndCompute(images[idx2].imgGray, overlapMask, images[idx2].keypoints[detectorType::AKAZE], images[idx2].descriptors[detectorType::AKAZE]);
     images[idx2].unlock();

     BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 AKAZE" );
     
     // adjust thresholds the first time through, subsequent runs would use cached data
     int kp_found = images[idx1].keypoints[detectorType::AKAZE].size() + images[idx2].keypoints[detectorType::AKAZE].size();
     if( idx1 + 1 == idx2 )
     {  
        // Auto tuning of threshold to limit number of features returned.
        float oldThreshold = akazeThreshold;
        int kp_found = images[idx1].keypoints[detectorType::AKAZE].size() + images[idx2].keypoints[detectorType::AKAZE].size();
        // Converge quicker if its really out of wack
        if( kp_found < featureMax / 2 ) akazeThreshold = akazeThreshold * 0.98;
        if( kp_found > featureMax * 4 ) akazeThreshold = akazeThreshold * 1.05;
        // Increase / decrease by 1% around the ideal number ( 
        if( kp_found < featureMax * 2 ) akazeThreshold = akazeThreshold * 0.99;
        if( kp_found > featureMax * 2 ) akazeThreshold = akazeThreshold * 1.01;
        if( loglevel > 3 ) IMGLOG << " INFO: AKAZE Threshold " << oldThreshold 
                << " Yielded " << kp_found << " keypoints.";
        if( oldThreshold != akazeThreshold )
        if( loglevel > 3 ) sout << "  Tuned to: " << akazeThreshold;
        if( loglevel > 3 ) sout << std::endl;
     }
     else
     {
        if( loglevel > 3 ) IMGLOG << " INFO: Using " << kp_found << "cached features for this image pair." << std::endl;
     }
  }  // If AKAZE
  return sout.str();
}  


//
// detectORB()
//
std::string detectORB( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;
  //
  // Detect ORB features and compute descriptors - if not already done before
  //
  Ptr<Feature2D> orb = ORB::create( featureMax );
  if( detectorOrb == true )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection ORB" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::ORB].size() == 0 )

        orb->detectAndCompute(images[idx1].imgGray, overlapMask, images[idx1].keypoints[detectorType::ORB], images[idx1].descriptors[detectorType::ORB]);
     images[idx1].unlock();

     BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 ORB" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::ORB].size() == 0 )

        orb->detectAndCompute(images[idx2].imgGray, overlapMask, images[idx2].keypoints[detectorType::ORB], images[idx2].descriptors[detectorType::ORB]);
     images[idx2].unlock();

     BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 ORB" );
  } // If ORB
  return sout.str();
}  



//
// detectBLOB()
//
std::string detectBLOB( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Simple Blob Detector ( as opposed to simple Bob detector )
  //
  
  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params blobParams;

  // Change thresholds
  blobParams.minThreshold = 5;
  blobParams.maxThreshold = 250;
  // Filter by Area.
  blobParams.filterByArea = true;
  blobParams.minArea = 5;
  // Filter by Circularity
  blobParams.filterByCircularity = false;
  blobParams.minCircularity = 0.1;
  // Filter by Convexity
  blobParams.filterByConvexity = false;
  blobParams.minConvexity = 0.87;
  // Filter by Inertia
  blobParams.filterByInertia = false;
  blobParams.minInertiaRatio = 0.01;
  
  // Set up detector with params
  Ptr<SimpleBlobDetector> blob = SimpleBlobDetector::create(blobParams);
  
  if( detectorBlob == true )
  {  

     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection BLOB" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::BLOB].size() == 0 )
     {
        // Make a copy of the overlap mask
        Mat maskedImg = overlapMask;
        // Copy the image data into the masked area of the image
        images[idx1].imgGray.copyTo(maskedImg, maskedImg);

        // if(loglevel > 6) imwrite(outputName + "_blob_dmask1.png", maskedImg);
        
        blob->detect(maskedImg, images[idx1].keypoints[detectorType::BLOB]);

        BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 BLOB - Detect" );

        Ptr<DescriptorExtractor> extractor1 = SIFT::create();
        extractor1->compute(images[idx1].imgGray, images[idx1].keypoints[detectorType::BLOB], images[idx1].descriptors[detectorType::BLOB]);

        BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 BLOB - Extract" );
     }
     images[idx1].unlock(); // protect from concurrency 
     
     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::BLOB].size() == 0 )
     {
        // Make a copy of the overlap mask
        Mat maskedImg = overlapMask;
        // Copy the image data into the masked area of the image
        images[idx1].imgGray.copyTo(maskedImg, maskedImg);

        // if(loglevel > 6)imwrite(outputName + "_blob_dmask2.png", maskedImg);

        blob->detect(maskedImg, images[idx2].keypoints[detectorType::BLOB]);

        BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 BLOB - Detect" );

        Ptr<DescriptorExtractor> extractor2 = SIFT::create();
        extractor2->compute(images[idx2].imgGray, images[idx2].keypoints[detectorType::BLOB], images[idx2].descriptors[detectorType::BLOB]);

        BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 BLOB - Extract" );
     }
     images[idx2].unlock(); // protect from concurrency 

  } // If Blob

  return sout.str();
}  

//
// detectBRISK()
std::string detectBRISK( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Detect BRISK features and compute descriptors.
  //
  
  Ptr<Feature2D> brisk = BRISK::create( );

  if( detectorBrisk == true )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection BRISK" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::BRISK].size() == 0 )
        brisk->detectAndCompute(images[idx1].imgGray, overlapMask, images[idx1].keypoints[detectorType::BRISK], images[idx1].descriptors[detectorType::BRISK]);
     images[idx1].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 BRISK" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::BRISK].size() == 0 )
        brisk->detectAndCompute(images[idx2].imgGray, overlapMask, images[idx2].keypoints[detectorType::BRISK], images[idx2].descriptors[detectorType::BRISK]);
     images[idx2].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 BRISK" );
  } // If BRISK
  
  return sout.str();
}  

std::string detectSURF( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  // SURF requires opencv_contrib to be bound into opencv 
  #if defined(__OPENCV_XFEATURES2D_HPP__)
  
  //
  // Detect SURF features and compute descriptors
  //
  
  Ptr<Feature2D> surf = SURF::create(surfMinHessian);
  if( detectorSurf == true )
  {  
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection SURF" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::SURF].size() == 0 )
        surf->detectAndCompute(images[idx1].imgGray, overlapMask, images[idx1].keypoints[detectorType::SURF], images[idx1].descriptors[detectorType::SURF]);
     images[idx1].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 SURF" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::SURF].size() == 0 )
        surf->detectAndCompute(images[idx2].imgGray, overlapMask, images[idx2].keypoints[detectorType::SURF], images[idx2].descriptors[detectorType::SURF]);
     images[idx2].unlock();

      BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 SURF" );  
  }  // IF SURF
  
  #else
     IMGLOG << "ERROR: cvfind: detectSURF() was called, but cvfind was built without OpenCV Contrib support" << std::endl;
  #endif  

  return sout.str();
}  

//
// detectSIFT()
//
std::string detectSIFT( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Detect SIFT features and compute descriptors
  //
  
  Ptr<Feature2D> sift = SIFT::create();
  if( detectorSift == true )
  {  
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection SIFT" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::SIFT].size() == 0 )
        sift->detectAndCompute(images[idx1].imgGray, overlapMask, images[idx1].keypoints[detectorType::SIFT], images[idx1].descriptors[detectorType::SIFT]);
     images[idx1].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 SIFT" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::SIFT].size() == 0 )
        sift->detectAndCompute(images[idx2].imgGray, overlapMask, images[idx2].keypoints[detectorType::SIFT], images[idx2].descriptors[detectorType::SIFT]);
     images[idx2].unlock();

      BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 SIFT" );  
  }  // IF SIFT

  return sout.str();
}  

// = = = = = = =


//
// detectGFTT()
//
std::string detectGFTT( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

/*    NOT IMPLEMENTED
  //
  // Detect GFTT features and compute descriptors - if not already done before
  //
               
  if( detectorGFFT == true )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection GFTT" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keypoints[detectorType::GFTT].size() == 0 )
        orb->detectAndCompute(images[idx1].imgGray, overlapMask, images[idx1].keypoints[detectorType::ORB], images[idx1].descriptors[detectorType::ORB]);
     images[idx1].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 ORB" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keypoints[detectorType::ORB].size() == 0 )
        orb->detectAndCompute(images[idx2].imgGray, overlapMask, images[idx2].keypoints[detectorType::ORB], images[idx2].descriptors[detectorType::ORB]);
     images[idx2].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 ORB" );
  } // If GFFT
*/

  return sout.str();
}  

//
// detectDSIFT()
//
std::string detectDSIFT( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Detect Dense SIFT features and compute descriptors
  //        ~~~~~

  Ptr<Feature2D> dsift = SIFT::create(0, 3, 0.02, 20, 0.5);
  if( detectorDenseSift == true )
  {  
     int detector = detectorType::DSIFT;
     int x_max  = (int)( (float)images[idx1].img.cols * ( overlapRatio + overlapMargin ) );
     int y_max  = (int)( (float)images[idx1].img.rows * ( overlapRatio + overlapMargin ) );
     int x_size = images[idx1].img.cols;
     int y_size = images[idx1].img.rows;
     
     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection Dense SIFT" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     // make a bunch of key points for 1st image in a grid pattern, starting from the edges growing in.
     if( images[idx1].keypoints[detectorType::DSIFT].size() == 0 )
     for( int x = 0; x < x_max; x += denseSiftWindowSize )
        for( int y = 0; y < y_max; y += denseSiftWindowSize )
        {
           images[idx1].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x, y, denseSiftWindowSize ) );      
           images[idx1].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x_size - x, y, denseSiftWindowSize ) );      
           images[idx1].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x, y_size - y, denseSiftWindowSize ) );      
           images[idx1].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x_size - x, y_size - y, denseSiftWindowSize ) );      
        }
        
     if( images[idx1].descriptors[detectorType::DSIFT].rows == 0 )
        dsift->compute(images[idx1].imgGray, images[idx1].keypoints[detectorType::DSIFT], images[idx1].descriptors[detectorType::DSIFT]);
     images[idx1].unlock();

      BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 DSIFT" );

     images[idx2].lock(); // protect from concurrency            
     // make a bunch of key points for 2nd image in a grid pattern
     // make a bunch of key points for 1st image in a grid pattern, starting from the edges growing in.
     if( images[idx2].keypoints[detectorType::DSIFT].size() == 0 )
     for( int x = 0; x < x_max; x += denseSiftWindowSize )
        for( int y = 0; y < y_max; y += denseSiftWindowSize )
        {
           images[idx2].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x, y, denseSiftWindowSize ) );      
           images[idx2].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x_size - x, y, denseSiftWindowSize ) );      
           images[idx2].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x, y_size - y, denseSiftWindowSize ) );      
           images[idx2].keypoints[detectorType::DSIFT].push_back( cv::KeyPoint( x_size - x, y_size - y, denseSiftWindowSize ) );      
        }
     
     if( images[idx2].descriptors[detectorType::DSIFT].rows == 0 )
        dsift->compute(images[idx2].imgGray, images[idx2].keypoints[detectorType::DSIFT], images[idx2].descriptors[detectorType::DSIFT]);
     images[idx2].unlock();

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 DSIFT" );  
     
  }  // IF Dense SIFT

  return sout.str();
}  

//
// detectLINE()
//
std::string detectLINE( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Detect LINE features with BinaryDescriptor and compute descriptors
  //
  
  Ptr<BinaryDescriptor> line = BinaryDescriptor::createBinaryDescriptor(  );
  if( detectorLine == true )
  {  

     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection LINE" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keylines[detectorType::LINE].size() == 0 )
     {
        std::vector<KeyLine> keylines1;
        cv::Mat descr1;
        // Replaced by overlapMask 
        // cv::Mat mask1 = Mat::ones( images[idx1].img.size(), CV_8UC1 );
        
        ( *line )( images[idx1].img, overlapMask, keylines1, descr1, false, false );

        // Copy keylines from octave 0 ( original scale )
        for ( int i = 0; i < (int) keylines1.size(); i++ )
        {
           if( keylines1[i].octave == 0 )
           {
              images[idx1].keylines[detectorType::LINE].push_back( keylines1[i] ); 
              images[idx1].descriptors[detectorType::LINE].push_back( descr1.row( i ) );
              KeyPoint kp;
              kp.pt = keylines1[i].pt;
              images[idx1].keypoints[detectorType::LINE].push_back( kp );
           }
        }
     } // If image had no keypoints
     images[idx1].unlock();
 
      BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 LINE" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keylines[detectorType::LINE].size() == 0 )
     {
        std::vector<KeyLine> keylines2;
        cv::Mat descr2;
        // replaced by overlapMask 
        // cv::Mat mask2 = Mat::ones( images[idx2].img.size(), CV_8UC1 );
        
        ( *line )( images[idx2].img, overlapMask, keylines2, descr2, false, false );

        // Copy keylines & descriptors for octave 0 to images data structure
        for ( int i = 0; i < (int) keylines2.size(); i++ )
        {
           if( keylines2[i].octave == 0 )
           {
              images[idx2].keylines[detectorType::LINE].push_back( keylines2[i] ); 
              images[idx2].descriptors[detectorType::LINE].push_back( descr2.row( i ) );
              KeyPoint kp;
              kp.pt = keylines2[i].pt;
              images[idx2].keypoints[detectorType::LINE].push_back( kp );
           }
        }
     } // If image had no keypoints
     images[idx2].unlock();
 
     BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 LINE" );  
  }  // IF LINE

  return sout.str();
}  

//
// detectLSD()
//
std::string detectLSD( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  //
  // Detect LINE features with LSDDetector and compute descriptors
  //

//  Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(0);  
  Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
  if( detectorLSD == true )
  {  

     if( loglevel > 3 ) IMGLOG << " INFO: Feature Detection SEGMENT" << std::endl;

     images[idx1].lock(); // protect from concurrency 
     if( images[idx1].keylines[detectorType::SEGMENT].size() == 0 )
     {
        std::vector<KeyLine> keylines1;
        cv::Mat descr1;
        
        lsd->detect( images[idx1].imgGray, keylines1, lsdScale, lsdOctaves, overlapMask );

        if( loglevel > 3 ) IMGLOG << " INFO: SEGMENT: kl1.size()=" << keylines1.size() << std::endl;

        // Copy keylines from octave 0 ( original scale )
        for ( int i = 0; i < (int) keylines1.size(); i++ )
        {
           if( keylines1[i].octave == 0 )
           {
              images[idx1].keylines[detectorType::SEGMENT].push_back( keylines1[i] ); 
              // images[idx1].descriptors[detectorType::SEGMENT].push_back( descr1.row( i ) );
              KeyPoint kp;
              kp.pt = keylines1[i].pt;
              images[idx1].keypoints[detectorType::SEGMENT].push_back( cv::KeyPoint( keylines1[i].pt.x, keylines1[i].pt.y, denseSiftWindowSize ) );
           }
        }
        
        if( loglevel > 3 ) IMGLOG << " INFO: SEGMENT: kl1.size()=" << images[idx1].keylines[detectorType::SEGMENT].size()
               << " kp1.size()= " << images[idx1].keypoints[detectorType::SEGMENT].size() << std::endl;
        
        Ptr<DescriptorExtractor> extractor1 = SIFT::create();
        extractor1->compute(images[idx1].imgGray, images[idx1].keypoints[detectorType::SEGMENT], \
                            images[idx1].descriptors[detectorType::SEGMENT]);

      BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 SEGMENT - Extract" );

     } // If image had no keypoints
     images[idx1].unlock();
 
     BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 1 SEGMENT" );

     images[idx2].lock(); // protect from concurrency 
     if( images[idx2].keylines[detectorType::SEGMENT].size() == 0 )
     {
        std::vector<KeyLine> keylines2;
        cv::Mat descr2;
    
        lsd->detect( images[idx2].imgGray, keylines2, lsdScale, lsdOctaves, overlapMask );
        
        if( loglevel > 3 ) IMGLOG << " INFO: SEGMENT: kl2.size()=" << keylines2.size() << std::endl;

        // Copy keylines & descriptors for octave 0 to images data structure
        for ( int i = 0; i < (int) keylines2.size(); i++ )
        {
           if( keylines2[i].octave == 0 )
           {
              images[idx2].keylines[detectorType::SEGMENT].push_back( keylines2[i] ); 
              // images[idx2].descriptors[detectorType::SEGMENT].push_back( descr2.row( i ) );
              KeyPoint kp;
              kp.pt = keylines2[i].pt;
              images[idx2].keypoints[detectorType::SEGMENT].push_back( cv::KeyPoint( keylines2[i].pt.x, keylines2[i].pt.y, denseSiftWindowSize ) );

           }
        }
  
        if( loglevel > 3 ) IMGLOG << " INFO: SEGMENT: kl2.size()=" << images[idx2].keylines[detectorType::SEGMENT].size()
               << " kp2.size()= " << images[idx2].keypoints[detectorType::SEGMENT].size() << std::endl;

        Ptr<DescriptorExtractor> extractor2 = SIFT::create();
        extractor2->compute(images[idx2].imgGray, images[idx2].keypoints[detectorType::SEGMENT], \
                            images[idx2].descriptors[detectorType::SEGMENT]);

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 SEGMENT - Extract" );

     } // If image had no keypoints
     images[idx2].unlock();
 
      BENCHMARK( pairs[idx1][idx2].benchmarks, "Image 2 SEGMENT" );  
  }  // IF SEGMENT

  return sout.str();
}  


// = = = = = = =



std::string detectPROTOTYPE( int idx1, int idx2, Mat overlapMask )
{
  std::stringstream sout;

  return sout.str();
}  


//
//  alignImages - pairwise image processing behemoth.  Accepts pair of image[] indexes and an output name
//                and performs many image processing tasks including feature generation, matching, point
//                extraction, analysis, and dumping new Hugin CP proposals
//

void alignImages(int idx1, int idx2, std::string outputName, Mat overlapMask )
{

   // benchmarks table.  Usage:
   // BENCHMARK( benchmarks, "3rd Benchmark" );
   // benchmarks.push_back(benchmarkEpoc( "First Benchmark.", std::chrono::high_resolution_clock::now() ) );
   
   BENCHMARK( pairs[idx1][idx2].benchmarks, "Thread Starts ..." );
      
   // Setup private debug log for each image pair.
   soutstream sout;

   if( alignLogging )
   {
      sout.coss.open( outputName + "_" + runID + ".log", std::ios::out | std::ios::app );
      sout.coss << "##### CVFIND TASK LOG STARTS " << GetDateTime() << " FOR RUN ID: " << runID << " ######" << std::endl;
   }

   if( loglevel > 3 ) IMGLOG << " INFO: alignImages(...,overlapMask) called using global defaults." << std::endl;
  
   if( loglevel > 6 ) IMGLOG << " INFO: alignImage(): cvfind cmdline=" << cmdline << std::endl;
  
   // IMGOUT << "TRACE: " << __FILE__ << ": " << __LINE__ << " argc=" << argc << std::endl;

   BENCHMARK( pairs[idx1][idx2].benchmarks, "Feature Detection ..." );

  //
  // Feature Detection
  // - - - - - - - - - 
  //

  if( detectorAkaze      ) sout << detectAKAZE(  idx1, idx2, overlapMask );
  if( detectorOrb        ) sout << detectORB(    idx1, idx2, overlapMask );
  if( detectorBlob       ) sout << detectBLOB(   idx1, idx2, overlapMask );
  if( detectorBrisk      ) sout << detectBRISK(  idx1, idx2, overlapMask );
  if( detectorSurf       ) sout << detectSURF(   idx1, idx2, overlapMask );
  if( detectorSift       ) sout << detectSIFT(   idx1, idx2, overlapMask );
  if( detectorDenseSift  ) sout << detectDSIFT(  idx1, idx2, overlapMask );
  if( detectorLine       ) sout << detectLINE(   idx1, idx2, overlapMask );
  if( detectorLSD        ) sout << detectLSD(    idx1, idx2, overlapMask );

  //
  // Feature Detection & Descriptor Computation Complete
  // Report on state of pairs[i][j].keypoints[enum_TYPE]...
  //
  
  int kp_detected = 0;
  int detector_used = 0;
  for( int i = 0; i < detectorType::ALL; i++ )
  {
     // Features would be detected When processing image pair is (N, N+1)  
     if( idx1 + 1 == idx2 )
     {  
        if(loglevel > 6) IMGLOG << " INFO: Detector[" << i << "] '" << detectorName[i]
                 << "' images[" << idx1 << "].kps[" << i << "].size()=" << images[idx1].keypoints[i].size() 
                 << "  images[" << idx1 << "].des[" << i << "].size()=" << images[idx1].descriptors[i].size() << std::endl; 
        else if(loglevel > 3 && images[idx1].keypoints[i].size() > 1) IMGLOG << " INFO: " << detectorName[i]
                 << " Detector Found " << images[idx1].keypoints[i].size() << " keypoints in image " << idx1 << std::endl; 

        if(loglevel > 6) IMGLOG << " INFO: Detector[" << i << "] '" << detectorName[i]
                 << "' images[" << idx2 << "].kps[" << i << "].size()=" << images[idx2].keypoints[i].size() 
                 << "  images[" << idx2 << "].des[" << i << "].size()=" << images[idx2].descriptors[i].size() << std::endl; 
        else if(loglevel > 3 && images[idx2].keypoints[i].size() > 1) IMGLOG << " INFO: " << detectorName[i]
                 << " Detector Found " << images[idx2].keypoints[i].size() << " keypoints in image " << idx2 << std::endl; 

     }

     kp_detected += images[idx1].keypoints[i].size() + images[idx2].keypoints[i].size();
     
     // ToDo: Kludge: untill downstream is fixed, select one good detector to use
     if( images[idx1].keypoints[i].size() * images[idx2].keypoints[i].size() > 0 ) detector_used = i; // Yea!  Found stuff.
  
  } // for( each detector type )


  //
  // For each feature detector, perform matching appropriate to that detector.   Some use common methods.
  // each image pair has vector<DMatch> scratch pads availabel for intermediate results.  The last of which is
  // considered the final one.
  // 
      
    BENCHMARK( pairs[idx1][idx2].benchmarks, "Match Features ..." );

  // ToDo: KLUDGE: These need to be replaced by the pad based versions.  This represents a pipeline easily represented
  //               By the pad based versions.   Also once matches are obtained, all the rest of the code uses the
  //               points they reference.   
  //
  // Match features structures 
  std::vector<DMatch> matches; // raw matches from the matcher
  std::vector<DMatch> filteredMatches; // filtered matches after identifying distance guess 
  std::vector<DMatch> finalMatches; // final matches after identifying overlap type ( edge / corner ) and pruning

  // For each detector type ...
  for( int detector = 0; detector < detectorType::ALL; detector++ )
  {
      int pad;
      
      if( detector == detectorType::PTO )
      {
         pad = pairs[idx1][idx2].currentMatchPad( detector ); // keep the current one ( from PTO, if it exists ) 
         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 
                              << "].currentMatchPad(" << detectorName[detector] << ") returned " << pad << std::endl;
      }
      else
      {
         pad = pairs[idx1][idx2].newMatchPad(detector, "alighImage() Detection" ); // this gets us a new empty vector<DMatch> to use
         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 
                              << "].newMatchPad(" << detectorName[detector] << ") returned " << pad << std::endl;
      }
      

      // Sanity checks: if these fail, likey a crash will follow
      if( pairs[idx1][idx2].matches[detector].size() == 0 )
         IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].size() = "
                << pairs[idx1][idx2].matches[detector].size() << " - likely fatal." << std::endl;

      // Sanity checks: if these fail, likey a crash will follow
      if( ( ! detector == detectorType::PTO ) && ( ! pairs[idx1][idx2].matches[detector].back().size() == 0 ) )
         IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].back().size() = "
                << pairs[idx1][idx2].matches[detector].back().size() << " - likely fatal." << std::endl;

      // 
      // BruteForce-Hamming Detector for ORB, AKAZE, BRISK, ...
      //

      if(    ( detector == detectorType::ORB || detector == detectorType::AKAZE || detector == detectorType::BRISK )
          && images[idx1].keypoints[detector].size() > 0
          && images[idx2].keypoints[detector].size() > 0 
        ) 
      {
         if( loglevel > 3 ) IMGLOG << " INFO: BFM:Hamming + " << detectorName[detector] << " Matching: "
              << " Keypoints: " << images[idx1].keypoints[detector].size() 
              << " x " << images[idx2].keypoints[detector].size() 
              << " Descriptors: " << images[idx1].descriptors[detector].rows 
              << " x " << images[idx2].descriptors[detector].rows 
              << std::endl;

         Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
         matcher->match(images[idx1].descriptors[detector], images[idx2].descriptors[detector], pairs[idx1][idx2].matches[detector].back(), Mat());
         BENCHMARK( pairs[idx1][idx2].benchmarks, "Brute Force Matcher" );

         if( loglevel > 3 ) IMGLOG << " INFO: BFM:Hamming + " << detectorName[detector] << " Sorting: " << pairs[idx1][idx2].matches[detector].back().size() << " matches." << std::endl;

         // Sort matches by score
//         std::sort(pairs[idx1][idx2].matches[detector].back().begin(), pairs[idx1][idx2].matches[detector].back().end());
         BENCHMARK( pairs[idx1][idx2].benchmarks, "Sort Match" );

    //     if( loglevel > 3 ) IMGLOG << " INFO: Culling to " 
    //                          << (int)(pairs[idx1][idx2].matches[detector].back().size() * GOOD_MATCH_PERCENT) << " matches." << std::endl;

         // Remove not so good matches leaving up to featureMax features
         int numAllMatches = pairs[idx1][idx2].matches[detector].back().size();
         int numGoodMatches = (int)std::min( (int)(pairs[idx1][idx2].matches[detector].back().size() * GOOD_MATCH_PERCENT), (int)featureMax );
         
         // This sort + erase process is probabaly where it all turns to shit.
         
//         matches.erase(pairs[idx1][idx2].matches[detector].back().begin()+numGoodMatches, pairs[idx1][idx2].matches[detector].back().end());
         BENCHMARK( pairs[idx1][idx2].benchmarks, "Cull Matches" );
         
         if( loglevel > 3 ) IMGLOG << " INFO: BFM:Hamming + " << detectorName[detector] << " Matcher: " << images[idx1].keypoints[detector].size() 
                << " x " << images[idx2].keypoints[detector].size() << " Found: " << numAllMatches
                << " Kept: " << pairs[idx1][idx2].matches[detector].back().size() << std::endl;

         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].size() = "
                << pairs[idx1][idx2].matches[detector].size() << " pad=" << pad << std::endl;


         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].size() = "
                << pairs[idx1][idx2].matches[detector].size() << std::endl;

         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "][" << pad << "].size() = "
                << pairs[idx1][idx2].matches[detector][pad].size() << std::endl;

         // ToDo: KLUDGE
         // pairs[idx1][idx2].matches[detector].back() is the last matches scratchpad for the idx1, idx2, detector combination
         // the highest detector ID "wins" by setting matches last

         for( int m = 0; loglevel > 8 && m < pairs[idx1][idx2].matches[detector].back().size(); m++ )
         {
            IMGLOG << "  -- Copy Match " << m + 1 << " of " << pairs[idx1][idx2].matches[detector].back().size() << std::endl;
            matches.push_back( pairs[idx1][idx2].matches[detector][pad][m] );            
         }

         if( loglevel > 8 ) IMGLOG << " -- Copy Finished!!" << std::endl;

         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].size() = "
                << pairs[idx1][idx2].matches[detector].size() << std::endl;

         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].back().size() = "
                << pairs[idx1][idx2].matches[detector].back().size() << std::endl;

         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "].size() = "
                << pairs[idx1][idx2].matches[detector].size() << std::endl;

         if( loglevel > 8 ) IMGLOG << " INFO: pairs[" << idx1 << "][" << idx2 << "].matches[" << detector << "][" << pad << "].size() = "
                << pairs[idx1][idx2].matches[detector][pad].size() << std::endl;


         if( loglevel > 8 ) IMGLOG << " -- Trying matches=... assignment" << std::endl;
          
         matches = pairs[idx1][idx2].matches[detector][pad];

         if( loglevel > 8 ) IMGLOG << " -- matches=... assignment Finished!!" << std::endl;
         
      } // If AKAZE || ORB || BRISK


      // 
      // FLANN Based Detector for BLOB-->SIFT, SURF, SIFT
      //
      if(    (    detector == detectorType::BLOB || detector == detectorType::SURF \
               || detector == detectorType::SIFT || detector == detectorType::DSIFT \
               || detector == detectorType::BLOB )
          && images[idx1].keypoints[detector].size() > 0
          && images[idx2].keypoints[detector].size() > 0 
        ) 
      {
        if( loglevel > 3 ) IMGLOG << " INFO: FLANN + " << detectorName[detector] << " Matcher. "
              << " Keypoints: " << images[idx1].keypoints[detector].size() 
              << " x " << images[idx2].keypoints[detector].size() 
              << " Descriptors: " << images[idx1].descriptors[detector].rows 
              << " x " << images[idx2].descriptors[detector].rows 
              << std::endl;
       
         Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
         std::vector<std::vector<DMatch>> knn_matches;

         matcher->knnMatch( images[idx1].descriptors[detector], images[idx2].descriptors[detector], knn_matches, 2 );

           BENCHMARK( pairs[idx1][idx2].benchmarks, "FLANN Match" );

         //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;

        for (size_t i = 0; i < knn_matches.size(); i++)
        {

            if ( loglevel > 6 && knn_matches[i][0].distance > 0 )
            sout << " INFO: knn_matchs[" << i << "][0].distance=" << knn_matches[i][0].distance
                       << " knn_matchs[" << i << "][1].distance=" << knn_matches[i][1].distance
                       << " ratio: " << knn_matches[i][0].distance / knn_matches[i][1].distance << std::endl;  
            
            if ( knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                // translation: add good match to the latest match pad
                pairs[idx1][idx2].matches[detector].back().push_back(knn_matches[i][0]);
            }
        }
     
          BENCHMARK( pairs[idx1][idx2].benchmarks, "FLANN Lowe's Ratio Test" );
       
     if( loglevel > 3 ) IMGLOG << " INFO: FLANN + " << detectorName[detector] << " Matching: " << images[idx1].keypoints[detector].size() 
          << " x " << images[idx2].keypoints[detector].size() << " Found: " << knn_matches.size()
          << " Kept: " << pairs[idx1][idx2].matches[detector].back().size() << std::endl;

         // ToDo: KLUDGE
         // pairs[idx1][idx2].matches[detector].back() is the last matches scratchpad for the idx1, idx2, detector combination
         // the highest detector ID "wins" by setting matches last

         matches = pairs[idx1][idx2].matches[detector].back();         
         
      } // If BLOB


      // 
      // Binary Descriptor Matcher for Line Detector 
      //
      if(    ( detector == detectorType::LINE || detector == detectorType::SEGMENT )
          && images[idx1].keylines[detector].size() > 0
          && images[idx2].keylines[detector].size() > 0 
        ) 
      {

         if ( loglevel > 3 ) IMGLOG << " INFO: BDM + " << detectorName[detector] << " Matcher. "
              << " Keylines: " << images[idx1].keylines[detector].size() 
              << " x " << images[idx2].keylines[detector].size() 
              << " Descriptors: " << images[idx1].descriptors[detector].rows 
              << " x " << images[idx2].descriptors[detector].rows 
              << std::endl;
       
       /* create a BinaryDescriptorMatcher object */
       Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
 
       /* require match */
       std::vector<DMatch> bdm_matches;
       bdm->match( images[idx1].descriptors[detector], images[idx2].descriptors[detector], bdm_matches );
 
       /* select best matches */
       std::vector<DMatch> bdm_good_matches;
       for ( int i = 0; i < (int) bdm_matches.size(); i++ )
       {
          if( bdm_matches[i].distance < lineMaxDistThresh )
          bdm_good_matches.push_back( bdm_matches[i] );
       }
 
       pairs[idx1][idx2].matches[detector].back() = bdm_good_matches;
 
           BENCHMARK( pairs[idx1][idx2].benchmarks, "BDS Match" );

       if( loglevel > 3 ) IMGLOG <<  " INFO: BDM + " << detectorName[detector] << " Matching: " << images[idx1].keylines[detector].size() 
              << " x " << images[idx2].keylines[detector].size() << " Found: " << bdm_matches.size()
              << " Kept: " << pairs[idx1][idx2].matches[detector].back().size() << std::endl;

         // ToDo: KLUDGE
         // pairs[idx1][idx2].matches[detector].back() is the last matches scratchpad for the idx1, idx2, detector combination
         // the highest detector ID "wins" by setting matches last

         matches = pairs[idx1][idx2].matches[detector].back();         
         
      } // If LINE


      // 
      // NULL Detector for PTO
      //
      if(  detector == detectorType::PTO )
      {
        if( loglevel > 3 ) IMGLOG << " INFO: PREMATCHED + " << detectorName[detector] << " Matcher.  " 
                             << pairs[idx1][idx2].cps[detector][pad].size() << " Pre-Matched Control Points."  
                             << std::endl;
      }


  } // For each detector
  

// - - - - - - - - - - - - - - - - - - - - - - -
//
// Extract Control ptoControlPoint(s) for all Matches
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 6 ) IMGLOG << " INFO: refreshAllPoints() called..." << std::endl;
  
  // dotrace = true;
  
  int ppcount = pairs[idx1][idx2].refreshPointPad();
  if( loglevel > 3 ) IMGLOG << " INFO: refreshPointPad() extracted " << ppcount << " points to pointPad[" << pairs[idx1][idx2].currentPointPad() << "] aka '" << pairs[idx1][idx2].pointPad.back().padName << "'" << std::endl;

  // dotrace = false;

  // Make a working copy of the latest point pad
  ptoPointPad oldpp;  
  ptoPointPad pp = pairs[idx1][idx2].pointPad.back();  

// - - - - - - - - - - - - - - - - - - - - - - -
//
// Loop through raw matches & Accumulate (x1,y1) and (x2,y2) tables for dup test.
// also dump raw matches for Debugging
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Raw Match Table: " 
                     << pp.size() << " matches." << std::endl;

  // Loop through each match, apply a quadrant based filter
  for( size_t i = 0; i < pp.size(); i++ )
  {
    int xSize = images[idx1].img.cols;
    int ySize = images[idx1].img.rows;
    float x1, y1, x2, y2, dx, dy, scalar, slope;
    std::string notes = "";

    x1 = pp.points1[i].x;
    x2 = pp.points2[i].x;
    y1 = pp.points1[i].y;
    y2 = pp.points2[i].y;

    dx = x2 - x1;
    dy = y2 - y1;
    
    scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

    if( dx == 0 )
    {
       slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
    }
    else
    {	
       slope=dy / dx;
    }
    
    if( loglevel > 6 ) sout << "Raw Matches:  ( " << x1 << ", " << y1 << " ) <---> ( " << x2 << ", " << y2 << " ) Scalar: " \
                     << scalar << " Slope: " << slope << " Notes: " << notes << std::endl;
    
  }


// - - - - - - - - - - - - - - - - - - - - - - -
//
// CALCULATE SCALARS TABLE
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  int idealScalar = 0;
  float idealScalarTolerance = 0.01f;
  std::vector<std::vector<int>> peaks; // peaks[n][0 = peak scalar,1=width, 2=depth, 3=weight]
  std::vector<int> scalars;
  std::vector<std::vector<float>> slopes;
  int xSize = images[idx1].img.cols;
  int ySize = images[idx1].img.rows;

  // Clearing tabulation table
  for( size_t i = 0; i < xSize + ySize + 2; i++ )
  {
    std::vector<float> foo;
    scalars.push_back( 0 );
    slopes.push_back( foo );
  }

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//
// Ideal Scalar determination method 3: Repetetive Homography, why not...
//
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  vector<ptoPointPad> trials; 

  if( loglevel > 3 && trialHomography ) IMGLOG << " INFO: Determine ideal scalar via repetetive homography..." << std::endl;
  
  for( float window = sqrt( ( xSize * xSize ) + ( ySize * ySize ) );
       trialHomography && window > ( 0.8f - (float)( overlapRatio + overlapMargin ) ) * (float)min( xSize, ySize );
       window -= trialHomStep ) 
  { 
     if( loglevel > 6 ) IMGLOG << " INFO: calling ppFilterScalar() ... " << std::endl;  
     ptoPointPad ppM1 = ppFilterScalar( &pp, window - (trialHomSize / 2), window + (trialHomSize / 2) );
     if( loglevel > 6 ) IMGLOG << " INFO: calling checkHomography() ... " << std::endl; 
     Mat homM1 = checkHomography( &ppM1 );   
     if( loglevel > 6 ) IMGLOG << " INFO: calling ppFilterDistance() ... " << std::endl;  
     ptoPointPad homV1 = ppFilterDistance( &ppM1, 0, 9 );
     if( loglevel > 6 ) IMGLOG << " INFO: ppFilterDistance() returned... " << std::endl;
     homV1.refresh(); // recalculate stuff 

     if( loglevel > 3 && homV1.size() > 5 && ! homM1.empty() ) IMGLOG << " INFO: Validated " << homV1.size() 
                          << " of " << ppM1.size() << " from " << pp.size() << " matches."
                          << " Trial= " << window 
                          << " COM Scalar= " << homV1.comscalar << " Slope= " << homV1.comslope 
                          << " Area= " << homV1.rect1.area();
                          
     // If the trial would survive final homography ( based on validated CP count )
     // keep it for later analysis.  Reject those where homography failed, the actual scalar is outside the window,
     // or the number of control points validated is less than the require number for a good pair.                     

     if( homV1.size() > 5 )
     {
        if( ! homM1.empty() ) 
        {
            homV1.h = homM1; // Update pointpad for this trial and for later analysis
            pairs[idx1][idx2].trialPointPad.push_back( homV1 );  
            
           if( homV1.size() > cpPerPairMin )
           {
              if( window + (trialHomSize / 2) > homV1.comscalar && window - (trialHomSize / 2) < homV1.comscalar )
              {
// Trying to roll all overlap logic into badOverlap...              
//                 if(    ( abs( homV1.comslope ) < 0.1f && idx1 + 1 == idx2 ) // strong horizontal overlap
//                     || ( abs( homV1.comslope ) > 0.4f && idx1 + rowSizeTolerance + 1 < idx2 ) ) // Some kind of vertical-ish overlap
//                 { 
                    std::string boCause = badOverlapCause( homV1 );
                    if( boCause.size() == 0 )
                    {
                       trials.push_back( homV1 );
                       if( loglevel > 3 ) sout << " <-- Good";
                    }
                    else
                    {
                       if( loglevel > 3 ) sout << " <-- Bad Overlap";
                       if( loglevel > 3 ) sout << " Cause: \"" << boCause << "\" ";
                    }
//                 }   
//                 else if( loglevel > 3 ) sout << " <-- Bad Slope";
              }
              else if( loglevel > 3 ) sout << " <-- Bad Scalar";
           }
           else if( loglevel > 3 ) sout << " <-- Too Few CPs";
        }
        else if( loglevel > 3 ) sout << " <-- H Matrix Empty";

        if( loglevel > 3 ) sout << std::endl;
    }
        
  }  // for each trial.   

  // Dump out a list of potentially "good" scalars.
  if( loglevel > 3 && trialHomography ) IMGLOG << " INFO: found " << trials.size() << " candidate scalars: { ";
  // For each "good" trial ...
  for( int t = 0; t < trials.size(); t++ )
  {
     if( loglevel > 3 && trialHomography ) sout << trials[t].comscalar;
     if( loglevel > 3 && t + 1 < trials.size() ) sout << ", ";
  }
  if( loglevel > 3 && trialHomography ) sout << " }" << std::endl;

   // For each "good" trial ... dump verbose info... 
   for( int t = 0; loglevel > 3 && t < trials.size(); t++ )
   { 
      IMGLOG << " INFO: Validated " << trials[t].size() 
             << " matches with CoM scalar " << trials[t].comscalar << " Slope= " << trials[t].comslope 
             << " Area= " << trials[t].rect1.area() << std::endl;
   }

   //
   // Save Diagnost Image for Homography Trials
   //
   if( saveDiagTrials )
   {  
      Mat imgpair;

      if( loglevel > 3 ) IMGLOG << " INFO: Saving Diagnostic Image to " << outputName + "_hom_trials.jpg" << std::endl; 

      hconcat( images[idx1].img, images[idx2].img, imgpair );

      // For each "good" trial ...
      for( int t = 0; t < trials.size(); t++ )
      { 
         // Pick a rando color for this scalars homography
         int r = ( rand() % 8 ) * 32;
         int g = ( rand() % 8 ) * 32;
         int b = ( rand() % 8 ) * 32;
         // get a "point" to use as an offset
         Point2f img2offset = Point2f( xSize,0 ); // relative location of img2 in the imgpair Mat 
         // Draw circles at each control point
         for( int i = 0; i < trials[t].size(); i++ )
         {
            circle(imgpair, trials[t].points1[i], 20, Scalar(r, g, b), 5);  
            circle(imgpair, trials[t].points2[i] + img2offset, 20, Scalar(r, g, b), 5);  
         }
         //draw a line between keypoints
         circle(imgpair, trials[t].com1, 1 + ( trials[t].size() / 5 ), Scalar(r, g, b), 1 + ( trials[t].size() / 10 ));  
         circle(imgpair, trials[t].com2 + img2offset, 1 + ( trials[t].size() / 5 ), Scalar(r, g, b), 1 + ( trials[t].size() / 10 ));  
         cv::line(imgpair, trials[t].com1, trials[t].com2 + img2offset, Scalar(r, g, b), 1 + ( trials[t].size() / 5 ), 8, 0);
         std::string label = "  <--" + to_string( trials[t].comscalar );
         putText(imgpair, label , trials[t].com1, FONT_HERSHEY_COMPLEX, (int)5, Scalar(r,g,b), (int)16 );
         putText(imgpair, label , trials[t].com2 + img2offset, FONT_HERSHEY_COMPLEX, (int)5, Scalar(r,g,b), (int)16 );
      }
      imwrite(outputName + "_hom_trials.jpg", imgpair);
   }
   
   
  vector<float> goodScalar;
  vector<int>   goodWeight;
  
  // Generate a list of best scalars by weight.  
  for( int t=0; t < trials.size(); t++ )
  {
     if( goodScalar.size() == 0 )
     {
        goodScalar.push_back( trials[t].comscalar );
        goodWeight.push_back( trials[t].size() );
     }
     else
     {
        bool handled = false;
        for( int gs = 0; gs < goodScalar.size(); gs++ )
        {
           if( trials[t].comscalar > goodScalar[gs] * 0.995f && trials[t].comscalar < goodScalar[gs] * 1.005f )
           {
              if( dotrace ) sout << "t=" << t << " gs=" << gs << goodScalar[gs] * 0.995f << " < " << trials[t].comscalar << " < " << goodScalar[gs] * 1.005f << " adding " << goodWeight[gs] << " += " << trials[t].size() << std::endl;
              goodWeight[gs] += trials[t].size();
              handled = true;
              break;
           }
        }        
        if( ! handled ) 
        {
           if( dotrace ) sout << "t=" << t << " scalar=" << trials[t].comscalar << " weight= " << trials[t].size() << " new " << std::endl;
           goodScalar.push_back( trials[t].comscalar );
           goodWeight.push_back( trials[t].size() );
        }
   
     } // else
  }

  // Dump out a list of potentially "good" scalars.
  if( loglevel > 3 && trialHomography ) IMGLOG << " INFO: found " << goodScalar.size() << " best scalar:weight unsorted : { ";

  // For each "good" trial ...
  for( int t = 0; t < goodScalar.size(); t++ )
  {
     if( loglevel > 3 && trialHomography ) sout << goodScalar[t] << ":" << goodWeight[t];
     if( t + 1 < goodScalar.size() ) if( loglevel > 3 ) sout << ", ";
  }
  if( loglevel > 3 && trialHomography ) sout << " }" << std::endl;


  // Sort good scalars by weight, descending
  // if( goodWeight.size() > 1 ) parallel_sort(std::less<>(), goodIndex, goodWeight, goodScalar);

  // what a fucking stupid shitshow, Excel has been able to do this since 1992
  std::vector<int> goodIndex = goodWeight;
  std::vector<float> sortScalar = goodScalar;
  std::vector<int> sortWeight = goodWeight;
  goodIndex = goodWeight;
  if( goodIndex.size() > 1 ) sortWeight = sortVecAByVecB( goodWeight, goodIndex );
  goodIndex = goodWeight;
  if( goodIndex.size() > 1 ) sortScalar = sortVecAByVecB( goodScalar, goodIndex );
  std::reverse(sortScalar.begin(),sortScalar.end());
  std::reverse(sortWeight.begin(),sortWeight.end());
  goodScalar = sortScalar;
  goodWeight = sortWeight;

  // Dump out a list of potentially "good" scalars.
  if( loglevel > 3 && trialHomography ) IMGLOG << " INFO: found " << goodScalar.size() << " best scalar:weight sorted : { ";
  // For each "good" trial ...
  for( int t = 0; t < goodScalar.size(); t++ )
  {
     if( loglevel > 3 && trialHomography ) sout << goodScalar[t] << ":" << goodWeight[t];
     if( loglevel > 3 && t + 1 < goodScalar.size() ) sout << ", ";
  }
  if( loglevel > 3 && trialHomography ) sout << " }" << std::endl;
  
  if( goodScalar.size() > 0 ) idealScalar = goodScalar[0];
  
  if( pairs[idx1][idx2].repairScalar > 0 )
  {
      if( loglevel > 3 ) IMGLOG << " INFO: Using Repair Scalar " << pairs[idx1][idx2].repairScalar << std::endl;
      idealScalar = pairs[idx1][idx2].repairScalar;
  }

  if( loglevel > 3 ) IMGLOG << " INFO: Setting idealScalar=" << idealScalar << std::endl;

  // Check for idealScalar hints
  if( (int)statsScalarHint.size() > idx2 - idx1 && statsScalarHint[idx2 - idx1] > 0 )
  {

     if( loglevel > 3 ) IMGLOG << " INFO: Ideal Scalar Hint: idealScalar: " << idealScalar << " Hint: " << statsScalarHint[idx2 - idx1]
                             << " % Diff: " 
                             << 100.0f * abs( (float)idealScalar - statsScalarHint[idx2 - idx1] ) / statsScalarHint[idx2 - idx1]
                             << std::endl;

     // Is the idealScalar > +/- 10% off 
     if( abs( (float)idealScalar - statsScalarHint[idx2 - idx1] ) / statsScalarHint[idx2 - idx1] > 0.1f )
     {
        if( loglevel > 3 ) IMGLOG << " INFO: Ideal idealScalar: " << idealScalar << " Replaced by Hint: " << statsScalarHint[idx2 - idx1]
                             << " % Diff: " 
                             << 100.0f * abs( (float)idealScalar - statsScalarHint[idx2 - idx1] ) / statsScalarHint[idx2 - idx1]
                             << " > 10% " << std::endl;
       //  idealScalar = statsScalarHint[idx2 - idx1];
       //  idealScalarTolerance = 0.05f; // Apply a wider tolerance.                          
     }
  }
  else
  {
     if( loglevel > 3 ) IMGLOG " INFO: No Ideal Scalar Hint for [" << idx1 << "] ---> [" << idx2 << "]" << std::endl;
  } 

  
  if( loglevel > 3 ) IMGLOG << " INFO: Distance filter Analyzing and filtering CP using idealScalar=" 
                     << idealScalar << " +/- 1% as a filter..." << std::endl;
                     
  if( loglevel > 3 && ! distanceFilter ) IMGLOG << " INFO: Distance Filter is disabled." << std::endl;
  //
  // END: Ideal Scalar determination method 3: repeated homography
  //

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //
  // DISTANCE FILTER using idealScalar
  //
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //  
  // Loop through each match, filter out implicitly bad CP pairs, and apply the distance filter if we
  // found an ideal distance above.
  //
  
  pairs[idx1][idx2].newPointPad( "Distance Filter" );
  
  for( size_t i = 0; i < pp.size(); i++ )
  {
    std::string notes;
    bool isGood, isBad;
    int xSize = images[idx1].img.cols;
    int ySize = images[idx1].img.rows;
    float overlap = 0.33;
    float overslope = 0.2; // Horizonatal slope filter, reciprocal used for vertical.
    float margin = 0.1;
    float x1, y1, x2, y2, dx, dy, scalar, slope;
     
    x1 = pp.points1[i].x;
    x2 = pp.points2[i].x;
    y1 = pp.points1[i].y;
    y2 = pp.points2[i].y;

    dx = x2 - x1;
    dy = y2 - y1;

    scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

    notes = "";

    if( dx == 0 )
    {
       slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
    }
    else
    {	
       slope=dy / dx;
    }

    isGood = false;
    isBad = false;

    if(     idealScalar > 0 
        && (    (float)scalar < (float)idealScalar * (float)( (float)1 - idealScalarTolerance )
             || (float)scalar > (float)idealScalar * (float)( (float)1 + idealScalarTolerance )
           )
      )
    { notes = notes + "BAD SCALAR "; isBad = true; }
    else
    { notes = notes + "GOOD SCALAR "; isGood = true; }

    int centercount = 0;

    // Reject impossible central features and reject slopes which cannot occur
    if(    ( x1 > xSize * ( overlap + margin ) && x1 < xSize * ( (float)1 - overlap - margin ) )
        && ( y1 > ySize * ( overlap + margin ) && y1 < ySize * ( (float)1 - overlap - margin ) ) ) centercount ++;

    if(    ( x2 > xSize * ( overlap + margin ) && x2 < xSize * ( (float)1 - overlap - margin ) )
        && ( y2 > ySize * ( overlap + margin ) && y2 < ySize * ( (float)1 - overlap - margin ) ) ) centercount ++;

    // Detect impossible central features
    if( centercount >= ( 2 - centerFilter ) )
    { notes = notes + "CENTER "; isBad = true; }  

    /* I'm not sure this makes sense ... at all.
                
    // Detect horizontal adjacency
    if(    xSize * ( (float)1 - overlap - margin ) < scalar 
        && xSize * ( (float)1 - overlap + margin ) > scalar
        && abs( slope ) < overslope )
    { notes = notes + "HA "; isGood = true; }

    // Detect vertical adjacency
    if(    ySize * ( (float)1 - overlap - margin ) < scalar 
        && ySize * ( (float)1 - overlap + margin ) > scalar
        && abs( slope ) > ( (float)1 / overslope ) )
    { notes = notes + "VA "; isGood = true; }

    */

    // Remember "good" matches or keep them all if filter is disabled
    if( ( isGood && ! isBad ) || ! distanceFilter )
    {
       pairs[idx1][idx2].pointPad.back().points1.push_back( pp.points1[i] );
       pairs[idx1][idx2].pointPad.back().points2.push_back( pp.points2[i] );
       pairs[idx1][idx2].pointPad.back().pointType.push_back( pp.pointType[i] );
       pairs[idx1][idx2].pointPad.back().lineIdx.push_back( pp.lineIdx[i] ); // no PTO line
       pairs[idx1][idx2].pointPad.back().distance.push_back( pp.distance[i] );

       
       if( loglevel > 6 ) sout << "Dist Filter Allowed: ( " << x1 << ", " << y1 << " ) <---> ( " << x2 << ", " << y2 << " ) Scalar: " \
                 << scalar << " Slope: " << slope << " Notes: " << notes << "match[" << i << "]" << std::endl;
    }
    else
    {
       if( loglevel > 6 ) sout << "Dist Filter Rejected: ( " << x1 << ", " << y1 << " ) <---> ( " << x2 << ", " << y2 << " ) Scalar: " \
                        << scalar << " Slope: " << slope << " Notes: " << notes << "match[" << i << "]" << std::endl;
    }
  }

   if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: " << pairs[idx1][idx2].pointPad.back().pointType.size() 
                        << " of " << pp.size() << " matches survived Distance Filter." << std::endl;    

   // Use filtered points
   pp = pairs[idx1][idx2].pointPad.back();
   //
   // END: DISTANCE FILTER
   //

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //
  // Loop through each match, try to find slope and distance clusters
  //
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  
  if( true )
  {
    int xSize = images[idx1].img.cols;
    int ySize = images[idx1].img.rows;
    int scalars[xSize + ySize];  

  for( size_t i = 0; i < xSize + ySize; i++ )
  {
    scalars[i]=0;
  }
  
  for( size_t i = 0; i < pp.size(); i++ )
  {
    float x1, y1, x2, y2, dx, dy, scalar, slope;
     
    x1 = pp.points1[i].x;
    x2 = pp.points2[i].x;
    y1 = pp.points1[i].y;
    y2 = pp.points2[i].y;

    dx = x2 - x1;
    dy = y2 - y1;

    scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

    if( dx == 0 )
    {
       slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
    }
    else
    {	
       slope=dy / dx;
    }

    scalars[(int)scalar]++;
  }

  if( idealScalar > 0 )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Filtered by Distance: " << idealScalar << std::endl;
  }
  else
  {
     if( loglevel > 3 && pp.size() > 0 ) IMGLOG << " WARN: No ideal distance detected, images are likely not overlapped." << std::endl;
  }
  
  if( loglevel > 3 || loglevel > 6 ) IMGLOG << " INFO: Distance occurence. " << std::endl; 

  for( size_t i = 0; loglevel > 6 && i < xSize + ySize; i++ )
  {
    if( loglevel > 6 ) if( scalars[i] > 2 ) sout << "Scalar: " << i << " Count: " << scalars[i] << std::endl;
  }
  
  }  // if( true )


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //
  // Loop through each plausible match, identify edge / corner matches, prune weakest
  //
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  int cornerCount[2][2][2];  // [image 0 or 1][x][y]
  cornerCount[0][0][0] = 0; 
  cornerCount[0][0][1] = 0;
  cornerCount[0][1][0] = 0; 
  cornerCount[0][1][1] = 0; // Oh shhh!
  cornerCount[1][0][0] = 0; 
  cornerCount[1][0][1] = 0;
  cornerCount[1][1][0] = 0; 
  cornerCount[1][1][1] = 0;

  bool quadrants[2][2][2];
  quadrants[0][0][0] = false;
  quadrants[0][0][1] = false;
  quadrants[0][1][0] = false;
  quadrants[0][1][1] = false; // No, really
  quadrants[1][0][0] = false;
  quadrants[1][0][1] = false;
  quadrants[1][1][0] = false;
  quadrants[1][1][1] = false;
  
  int x1com, y1com, x2com, y2com; // center of mass relative to the center of the image
  x1com = y1com = x2com = y2com = 0;
  int scalar1com, scalar2com;
  scalar1com = scalar2com = 0;
  int scalar1comN, scalar2comN; // normallized for image dimmensions with y multipled by xSize / ySize
  scalar1comN = scalar2comN = 0;
  bool img1comGood = false;
  bool img2comGood = false;

  // Hell, why not just make everything a global variable at this rate?

  float overlap = overlapRatio;
  float margin = overlapMargin;

  pairs[idx1][idx2].newPointPad( "CP Overlap Filter" );
          
  for( size_t i = 0; i < pp.size(); i++ )
  {
    std::string notes;
    bool isGood, isBad;
    int xSize = images[idx1].img.cols;
    int ySize = images[idx1].img.rows;
    float overlap = 0.33;
    float overslope = 0.2; // Horizonatal slope filter, reciprocal used for vertical.
    float margin = 0.1;
    float x1, y1, x2, y2, dx, dy, scalar, slope;
     
    // get coordinates of key points ( x1, y1 ) --> ( x2, y2 ) 
    x1 = pp.points1[i].x;
    x2 = pp.points2[i].x;
    y1 = pp.points1[i].y;
    y2 = pp.points2[i].y;

    // calculate displacements 
    dx = x2 - x1;
    dy = y2 - y1;
    
    // update center of mass buckets ( divided to make averages after loop exits )
    x1com += x1 - ( xSize / 2 );
    y1com += y1 - ( ySize / 2 );
    x2com += x2 - ( xSize / 2 );
    y2com += y2 - ( ySize / 2 );

    scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

    notes = "";

    if( dx == 0 )
    {
       slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
    }
    else
    {	
       slope=dy / dx;
    }


    int centercount = 0;

    // Reject impossible central features and reject slopes which cannot occur
    if(    ( x1 > xSize * ( overlap + margin ) && x1 < xSize * ( (float)1 - overlap - margin ) )
        && ( y1 > ySize * ( overlap + margin ) && y1 < ySize * ( (float)1 - overlap - margin ) ) ) centercount ++;

    if(    ( x2 > xSize * ( overlap + margin ) && x2 < xSize * ( (float)1 - overlap - margin ) )
        && ( y2 > ySize * ( overlap + margin ) && y2 < ySize * ( (float)1 - overlap - margin ) ) ) centercount ++;

    // Detect impossible central features
    if( centercount >= ( 2 - centerFilter ) )
    { notes = notes + "CENTER "; isBad = true; }      
                
    // Detect horizontal adjacency
    if(    xSize * ( (float)1 - overlap - margin ) < scalar 
        && xSize * ( (float)1 - overlap + margin ) > scalar
        && abs( slope ) < overslope )
    { notes = notes + "HA "; isGood = true; }

    // Detect vertical adjacency
    if(    ySize * ( (float)1 - overlap - margin ) < scalar 
        && ySize * ( (float)1 - overlap + margin ) > scalar
        && abs( slope ) > ( (float)1 / overslope ) )
    { notes = notes + "VA "; isGood = true; }

    bool u1 = true; // y less than half height
    bool r1 = true; // x less than half width
    bool u2 = true; // y less than half height
    bool r2 = true; // x less than half width
    
    if( x1 > xSize / 2 ) r1 = false; 
    if( y1 > ySize / 2 ) u1 = false;
    
    if( x2 > xSize / 2 ) r2 = false;
    if( y2 > ySize / 2 ) u2 = false;

    cornerCount[0][r1][u1]++;
    cornerCount[1][r2][u2]++;
    
    notes = notes + " [ r1=" + std::to_string( r1 ) + " u1=" + std::to_string( u1 ); 
    notes = notes + " r2=" + std::to_string( r2 ) + " u2=" + std::to_string( u2 ) + " ]"; 

    // Remember "good" matches
    if( ( isGood && ! isBad ) || ! cpoverlapFilter )
    {
       pairs[idx1][idx2].pointPad.back().points1.push_back( pp.points1[i] );
       pairs[idx1][idx2].pointPad.back().points2.push_back( pp.points2[i] );
       pairs[idx1][idx2].pointPad.back().pointType.push_back( pp.pointType[i] );       
       pairs[idx1][idx2].pointPad.back().lineIdx.push_back( pp.lineIdx[i] ); // no PTO line
       pairs[idx1][idx2].pointPad.back().distance.push_back( pp.distance[i] );       
       if( loglevel > 6 ) sout << "  Passed By";
    }
    else
       if( loglevel > 6 ) sout << "Filtered By ";


    if( loglevel > 6 ) sout << "Overlap Filter ( " << x1 << ", " << y1 << " ) <---> ( " << x2 << ", " << y2 << " ) Scalar: " \
                 << scalar << " Slope: " << slope << " Notes: " << notes << std::endl;


  } // while loop for each control point



   if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: " << pairs[idx1][idx2].pointPad.back().pointType.size()
                        << " of " << pp.size() << " matches survived Overlap Filter." << std::endl;    

   // Use filtered points
   pp = pairs[idx1][idx2].pointPad.back();


  
  int foo = pp.size();
  if( foo == 0 )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: No Matches To Process.  Images Not Overlapped. )" << std::endl;
     foo = 1;
  }
     
  if( loglevel > 6 )  // needed because  x = x / fn(...)  != foo = fn(...); x = x / foo !!!
  {  
     sout << " Center of mass sums for " << foo << " control points: " << std::endl;
     sout << " Image 1: x= " << x1com << ", y=" << y1com << std::endl;
     sout << " Image 2: x= " << x2com << ", y=" << y2com << std::endl;
  }

  // make averages for each center of mass coordinate.   This is a vector from center of image.
  x1com = x1com / foo;
  y1com = y1com / foo;
  x2com = x2com / foo;
  y2com = y2com / foo;
  // compute scalar from center of image to COM.  "Good" is beyond to overlap.
  scalar1com =sqrt( ( x1com * x1com ) + ( y1com * y1com ) );
  scalar2com =sqrt( ( x2com * x2com ) + ( y2com * y2com ) );
  // compute image size normallized scalar from center of image to COM.  "Good" is beyond to overlap.
  scalar1comN =sqrt( ( x1com * x1com ) + ( ( ( y1com * xSize ) / ySize ) * ( ( y1com * xSize ) / ySize ) ) );
  scalar2comN =sqrt( ( x2com * x2com ) + ( ( ( y2com * xSize ) / ySize ) * ( ( y2com * xSize ) / ySize ) ) );
  // compute a minimum scalar for reasonable overlap, since for scalarN y is scaled we avoid Pythagoras 
  int idealScalarNmax = ( xSize / 2 ) * (float)( 1 - ( ( overlap - margin ) )) * 1.41;  // 
  int idealScalarNmin = ( xSize / 2 ) * (float)( 1 - ( ( overlap + margin ) ));  // 

  if( loglevel > 3 )
  {  
     IMGLOG << " INFO: Center of mass analysis: " << std::endl;
     IMGLOG << "       Ideal scalarN: " << idealScalarNmin << " to " << idealScalarNmax << std::endl;  // COM should be in center of the overlap
     IMGLOG << "       Image 1: x= " << x1com << ", y=" << y1com << " scalar=" << scalar1com << " scalarN=" << scalar1comN << std::endl;
     IMGLOG << "       Image 2: x= " << x2com << ", y=" << y2com << " scalar=" << scalar2com << " scalarN=" << scalar2comN << std::endl;
  }


  if( loglevel > 3 )
  {  
     IMGLOG << "       Quadrant analysis: " << std::endl;
     IMGLOG << "       Image 1: " << cornerCount[0][0][0] << " | " << cornerCount[0][1][0] << std::endl;
     IMGLOG << "                " << cornerCount[0][0][1] << " | " << cornerCount[0][1][1] << std::endl;
     IMGLOG << "       Image 2: " << cornerCount[1][0][0] << " | " << cornerCount[1][1][0] << std::endl;
     IMGLOG << "                " << cornerCount[1][0][1] << " | " << cornerCount[1][1][1] << std::endl;
  }

    BENCHMARK( pairs[idx1][idx2].benchmarks, "Quadrant Analysis" );
  
  for( size_t i = 0; i < 2; i++ )
  {

  int hi1 = 0;
  int hi2 = 0;

  for( size_t j = 0; j < 2; j++ )
  { 
     for( size_t k = 0; k < 2; k++ )
     {
         if( cornerCount[i][j][k] > hi1 ) { hi2 = hi1; hi1 = cornerCount[i][j][k]; }
         if( cornerCount[i][j][k] > hi2 && cornerCount[i][j][k] < hi1 ) { hi2 = cornerCount[i][j][k]; }
     }
  }   

  for( size_t j = 0; j < 2; j++ )
  { 
     for( size_t k = 0; k < 2; k++ )
     {
        if( cornerCount[i][j][k] >= hi2 ) { quadrants[i][j][k] = true; }
     }
  } 
  
    if( loglevel > 3 || loglevel > 6 ) IMGLOG << " INFO: Quadrant analysis: Img " << i + 1 << " hi1=" << hi1 << ", hi2=" << hi2 << std::endl;
     
  }
  
    if( loglevel > 3 || loglevel > 6 ) IMGLOG << "       Img 1: " << quadrants[0][0][0] << " | " << quadrants[0][1][0]  
                                  << "       Img 2: " << quadrants[1][0][0] << " | " << quadrants[1][1][0] << std::endl;
    if( loglevel > 3 || loglevel > 6 ) IMGLOG << "              " << quadrants[0][0][1] << " | " << quadrants[0][1][1]  
                                  << "              " << quadrants[1][0][1] << " | " << quadrants[1][1][1] << std::endl;
//  sout << quadrants[0][1] << " | " << quadrants[1][1] << std::endl;

  int maskX1min, maskX1max, maskY1min, maskY1max;
  int maskX2min, maskX2max, maskY2min, maskY2max;
  
  if( true )
  {
    float overlap = 0.33;
    float overslope = 0.2; // Horizonatal slope filter, reciprocal used for vertical.
    float margin = 0.01;
    int xSize = images[idx1].img.cols;
    int ySize = images[idx1].img.rows;

  // ALL THE CASES!!!
  //     if(    ( x1 > xSize * ( overlap + margin ) && x1 < xSize * ( (float)1 - overlap - margin ) )

   // Set default masks to whole image
   maskX1min = maskY1min = maskX2min = maskY2min = 0;
   maskX1max = maskX2max = xSize;
   maskY1max = maskY2max = ySize;

  bool isOverlapped = false;
  int overlapCount = 0; // number of overlap types triggered.  > 1 means fubar
  
  // assume (0, 0) --> (xSize, ySize) and modify only the coordinates that differ
  
  if(    ( quadrants[0][0][0] && quadrants[1][1][0] && quadrants[0][0][1] && quadrants[1][1][1] )
//      || ( x1com + ( xSize / 2 ) < xSize * (float)( overlap ) )
      || ( x1com + ( xSize / 2 ) > xSize * (float)( (float)1 - overlap ) )
      || ( x2com + ( xSize / 2 ) < xSize * (float)( overlap ) )
//      || ( x2com + ( xSize / 2 ) > xSize * (float)( (float)1 - overlap ) )
    )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Overlap: Img 1 Right Edge <--> Img 2 Left Edge" << std::endl;
     maskX1min = xSize * ( (float)1 - overlap - margin ); 
     maskX2max = xSize * ( overlap + margin );
     isOverlapped = true;
     overlapCount++;
  }

  if(    ( quadrants[0][1][0] && quadrants[1][0][0] && quadrants[0][1][1] && quadrants[1][0][1] )
      || ( x1com + ( xSize / 2 ) < xSize * (float)( overlap ) )
//      || ( x1com + ( xSize / 2 ) > xSize * (float)( (float)1 - overlap ) )
//      || ( x2com + ( xSize / 2 ) < xSize * (float)( overlap ) )
      || ( x2com + ( xSize / 2 ) > xSize * (float)( (float)1 - overlap ) )
    )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Overlap: Img 1 Left Edge <--> Img 2 Right Edge" << std::endl;
     maskX2min = xSize * ( (float)1 - overlap - margin );  // right
     maskX1max = xSize * ( overlap + margin ); // left
     isOverlapped = true;
     overlapCount++;
  }

  if(   ( quadrants[0][0][0] && quadrants[0][1][0] && quadrants[1][0][1] && quadrants[1][1][1] )
//      || ( y1com + ( ySize / 2 ) < ySize * (float)( overlap ) )
      || ( y1com + ( ySize / 2 ) > ySize * (float)( (float)1 - overlap ) )
      || ( y2com + ( ySize / 2 ) < ySize * (float)( overlap ) )
//      || ( y2com + ( ySize / 2 ) > ySize * (float)( (float)1 - overlap ) )
    )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Overlap: Img 1 Bottom Edge <--> Img 2 Top Edge" << std::endl;
     maskY1min = ySize * ( (float)1 - overlap - margin ); // bottom
     maskY2max = ySize * ( overlap + margin ); // top
     isOverlapped = true;
     overlapCount++;
  }

  if(    ( quadrants[1][0][0] && quadrants[1][1][0] && quadrants[0][0][1] && quadrants[0][1][1] )
      || ( y1com + ( ySize / 2 ) < ySize * (float)( overlap ) )
//      || ( y1com + ( ySize / 2 ) > ySize * (float)( (float)1 - overlap ) )
//      || ( y2com + ( ySize / 2 ) < ySize * (float)( overlap ) )
      || ( y2com + ( ySize / 2 ) > ySize * (float)( (float)1 - overlap ) )
    )
  {
     if( loglevel > 3 ) IMGLOG << " INFO: Overlap: Img 1 Top Edge <--> Img 2 Bottom Edge" << std::endl;
     maskY2min = ySize * ( (float)1 - overlap - margin );  // bottom
     maskY1max = ySize * ( overlap + margin ); // top
     isOverlapped = true;
     overlapCount++;
  }

  if( ! isOverlapped || overlapCount == 0 )
  {
     IMGLOG << " WARN: Quadrant Test Failed To Find Overlap!  Default mask used. " << std::endl;
  }

  if( loglevel > 3 && overlapCount > 1 )
  {
     IMGLOG << " WARN: Conflicting Overlaps Detected!" << std::endl;
  }  
  
    if( loglevel > 3 || loglevel > 6 )
       IMGLOG << " INFO: Overlap Mask: Img 1: ( " << maskX1min << ", " << maskY1min << " ) <--> ( " << maskX1max << ", " << maskY1max << " )" << std::endl;
    if( loglevel > 3 || loglevel > 6 )
       IMGLOG << "                     Img 2: ( " << maskX2min << ", " << maskY2min << " ) <--> ( " << maskX2max << ", " << maskY2max << " )" << std::endl;
  
  } // if( true )


// - - - - - - - - - - - - - - - - - - - - - - -
//
// APPLY QUADRANT FILTER
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Filter CP Pairs for overlapping quadrants in " 
                     << pp.size() << " matches." << std::endl;

  pairs[idx1][idx2].newPointPad( "Quadrant Filter" );

  // Loop through each match, apply a quadrant based filter
  for( size_t i = 0; i < pp.size(); i++ )
  {
    std::string notes;
    bool isGood, isBad;
    int xSize = images[idx1].img.cols;
    int ySize = images[idx1].img.rows;
    float overlap = 0.33;
    float overslope = 0.2; // Horizonatal slope filter, reciprocal used for vertical.
    float margin = 0.1;
    float x1, y1, x2, y2, dx, dy, scalar, slope;

    notes = "";     

    x1 = pp.points1[i].x;
    x2 = pp.points2[i].x;
    y1 = pp.points1[i].y;
    y2 = pp.points2[i].y;

    dx = x2 - x1;
    dy = y2 - y1;
    
    scalar=sqrt( ( dx * dx ) + ( dy * dy ) );

    if( dx == 0 )
    {
       slope=dy * 1000000; // instead of divide by zero, multiply by 1000000
    }
    else
    {	
       slope=dy / dx;
    }
    

    // Remember "good" matches or all matches if filter is disabled.
    if(    x1 > maskX1min && x1 < maskX1max && y1 > maskY1min && y1 < maskY1max
          && x2 > maskX2min && x2 < maskX2max && y2 > maskY2min && y2 < maskY2max 
      )
    {
       pairs[idx1][idx2].pointPad.back().points1.push_back( pp.points1[i] );
       pairs[idx1][idx2].pointPad.back().points2.push_back( pp.points2[i] );
       pairs[idx1][idx2].pointPad.back().pointType.push_back( pp.pointType[i] );       
       pairs[idx1][idx2].pointPad.back().lineIdx.push_back( pp.lineIdx[i] ); // no PTO line
       pairs[idx1][idx2].pointPad.back().distance.push_back( pp.distance[i] );

       notes = notes + " Added";
    }
    else
    {
       // add anyway if filter is disabled 
       if( ! quadrantFilter )
       {
          pairs[idx1][idx2].pointPad.back().points1.push_back( pp.points1[i] );
          pairs[idx1][idx2].pointPad.back().points2.push_back( pp.points2[i] );
          pairs[idx1][idx2].pointPad.back().pointType.push_back( pp.pointType[i] );       
          pairs[idx1][idx2].pointPad.back().lineIdx.push_back( pp.lineIdx[i] ); // no PTO line
          pairs[idx1][idx2].pointPad.back().distance.push_back( pp.distance[i] );

          notes = notes + " Would Be";
       }
       notes = notes + " Rejected";
    }
    if( loglevel > 6 ) sout << "Quadrant Mask Filter:  ( " << x1 << ", " << y1 << " ) <---> ( " << x2 << ", " << y2 << " ) Scalar: " \
                     << scalar << " Slope: " << slope << " Notes: " << notes << std::endl;
    
  }


   if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: " << pairs[idx1][idx2].pointPad.back().pointType.size() 
                        << " of " << pp.size()  << " matches survived Quadrant Filter." << std::endl;    

   // Use filtered points
   pp = pairs[idx1][idx2].pointPad.back();

   matches = filteredMatches;
   filteredMatches.clear();

  bool isGoodMatch = false;
  if( pp.size() > 5 && idealScalar > 0 )
  {
     isGoodMatch = true;
  }

   if( loglevel > 3 ) IMGLOG << " INFO: Connected by " << pp.size() << " CPs."
                       << "  idealScalar=" << idealScalar << " isGoodMatch=" << isGoodMatch << std::endl;



    BENCHMARK( pairs[idx1][idx2].benchmarks, "Apply Quadrant Filter" );


  // Extract location of good matches, or all matches if the filter is disabled.
  std::vector<Point2f> points1, points2;

  points1 = pp.points1;
  points2 = pp.points2;

    BENCHMARK( pairs[idx1][idx2].benchmarks, "Extract Points for Homography" );

// - - - - - - - - - - - - - - - - - - - - - - -
//
// PREP FOR HOMOGRAPHY
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Prep for Homography in " 
                     << pp.size() << " matches, " << points1.size() << " points1, " << points2.size() << " points2." << std::endl;

  // Protect newCPlist from concurrency
  // newcplist_guard.lock();
     int cpadded;
     cpadded = 0;   
  // potential deadlock as sout also protected by mutex
  if( loglevel > 3 ) IMGLOG << " INFO: Hugin PTO Edits Proposed for image pair ( " << idx1 << ", " << idx2 << " ) : " << std::endl;
  for( size_t i = 0; isGoodMatch && i < pp.size(); i++ )
  {
     // typical format: c n101 N106 x83.237684 y1542.237846 Y83.237684 Y2154.237846 t0
     // I'm sure I could use the hugin / panotools sources to get this correct, but I'm arguably impatient. 

     if( ! doHomography ) // if we do homography, we use those results for Hugin config elements
     {   
        std::string s;

        // If Hugin expects (0,0) in lower right, then mirror x and also y coordinates...
        if( huginFlipOut )
        s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
            + " x" + std::to_string( xSize - pp.points1[i].x )   
            + " y" + std::to_string( ySize - pp.points1[i].y )   
            + " X" + std::to_string( xSize - pp.points2[i].x )   
            + " Y" + std::to_string( ySize - pp.points2[i].y )
            + " t0" ;  // no idea what t0 does
        else 
        s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
            + " x" + std::to_string( pp.points1[i].x )   
            + " y" + std::to_string( pp.points1[i].y )   
            + " X" + std::to_string( pp.points2[i].x )   
            + " Y" + std::to_string( pp.points2[i].y )
            + " t0" ;  // no idea what t0 does

        pairs[idx1][idx2].newCPlist.push_back( s ); // add to the list
        cpadded++;                
       // potential deadlock as sout also protected by mutex
       if( loglevel > 3 ) sout << s << std::endl;
     }
     
  }
  if( loglevel > 3 || cpadded > 0 )
       IMGLOG << " INFO: Chippy Analysis Added " << cpadded << " control points to Hugin proposal." << std::endl;      
  if( loglevel > 3 && cpadded < cpPerPairMin ) IMGLOG << " INFO: Quadrant Analysis Added Less Than " 
                                      << cpPerPairMin << " control points to Hugin proposal." << std::endl;      
  if( loglevel > 3 && cpadded > cpPerPairMax ) IMGLOG << " INFO: Quadrant Analysis Added More Than " 
                                      << cpPerPairMin << " control points to Hugin proposal." << std::endl;      

  // NB: probabaly need to keep these internally, then publish them if "good" 

  // unProtect newCPlist from concurrency
  //  newcplist_guard.unlock();  

    BENCHMARK( pairs[idx1][idx2].benchmarks, "Quadrant Filter to HUGIN config elements" );

// - - - - - - - - - - - - - - - - - - - - - - -
//
// DO HOMOGRAPHY
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Do Homography using " 
                     << pp.size() << " matches." << std::endl;

  if( ( doHomography || isGoodMatch || loglevel > 6 ) && pp.points1.size() > 5 && pp.points2.size() > 5 )
  {
     // Registered image will be resotred in imReg. 
     // The estimated homography will be stored in h. 
     Mat imReg, h;

     try {

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Homography ..." );

     // Find homography
     h = findHomography( pp.points1, pp.points2, RANSAC );
       BENCHMARK( pairs[idx1][idx2].benchmarks, "findHomography()" );
       
     pairs[idx1][idx2].h = h;  

     // if the homography matrix is empty, we return.   Its not an error state, in that homography may just be bad.
     if (h.empty())
     {
         if( loglevel > 6 ) IMGLOG << " INFO: findHomography() returned empty matrix." << std::endl;
         return;
     }

     // Disabled warpPerspective, see above
     if( saveDiagAligned && ( ( saveDiagGood && isGoodMatch ) || ( saveDiagBad && ! isGoodMatch ) ) )
     {
        // // Use homography to warp image, this was likely hiding h = empty
        if( loglevel > 3 ) IMGLOG << " INFO: calling warpPerspective()" << endl; 
        warpPerspective(images[idx1].img, imReg, h, images[idx2].img.size());
        BENCHMARK( pairs[idx1][idx2].benchmarks, "warpPerspective()" );
        if( loglevel > 3 ) IMGLOG << " INFO: Saving aligned diagnostic image : " << outputName + "_aligned.jpg" << endl; 
        imwrite(outputName + "_aligned.jpg", imReg);
        BENCHMARK( pairs[idx1][idx2].benchmarks, "Save Aligned Image" );
     }

         } // try
         catch(const cv::Exception& ex) {
         const char* err_msg = ex.what();
         IMGLOG << "ERROR: Caught cv::Exception: " << err_msg << std::endl;
         return;
         }    

     // Print estimated homography
     if( loglevel > 6 ) IMGLOG << "Estimated homography : " << endl << h << endl;

// - - - - - - - - - - - - - - - - - - - - - - -
//
// Check that size of point clouds are approximately the same.  Method 1.
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Check Point Cloud Areas for " 
                     << pp.size() << " matches using Method 1" << std::endl;

     std::vector<std::vector<std::vector<int>>> zone; // zone[x/cellSize][y/cellSize][0 | 1]

     if( loglevel > 6 ) IMGLOG << " INFO: Creating zone matrix for area analysis..." << std::endl;

     // Inserting elements into vector, owing to rounding we need one more than logic dictates
     for (int i = 0; i < ( xSize / cellSize ) + 1; i++)
     {
        vector< vector< int >> v1;  
        for (int j = 0; j < ( ySize / cellSize ) + 1; j++)
        {
            vector< int > v2;
            v2.push_back( 0 ); // ...[0]
            v2.push_back( 0 ); // ...[1]            
            v1.push_back(v2);
        }
        zone.push_back(v1);
     }     
     
     // Tally points into zones
     for( int i=0; i < pp.size(); i++ )
     {
        zone[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize][0] ++; // Tally the idx1 image points 
        zone[pp.points2[i].x / cellSize][pp.points2[i].y / cellSize][1] ++; // Tally the idx1 image points 
     }

     int zone1cnt, zone2cnt;
     zone1cnt = zone2cnt = 0;

     // Inserting elements into vector, owing to rounding we need one more than logic dictates
     for (int i = 0; i < ( xSize / cellSize ) + 1; i++)
     {
        vector< vector< int >> v1;  
        for (int j = 0; j < ( ySize / cellSize ) + 1; j++)
        {
            if( zone[i][j][0] > 0 ) zone1cnt++;
            if( zone[i][j][1] > 0 ) zone2cnt++;
        }
     }     
     
     float zoneSprawlRatio = (float)((float)abs( zone1cnt - zone2cnt ) / (float)( zone1cnt + zone2cnt ));
     
     if( loglevel > 3 ) IMGLOG << " INFO: Raw Zone Count: Image [" << idx1 << "]:" << zone1cnt << " Img [" << idx2 << "]: " << zone2cnt 
                          << " with cellSize=" << cellSize << std::endl;
     if( loglevel > 3 ) IMGLOG << " INFO: Raw Zone Analysis: Image [" << idx1 << "] --> [" << idx2 << "]: Sprawl Ratio: " 
                          << abs( zone1cnt - zone2cnt ) << " / " << zone1cnt + zone2cnt
                          << " = " << zoneSprawlRatio << std::endl;

// - - - - - - - - - - - - - - - - - - - - - - -
//
// Check that size of point clouds are approximately the same.  Method 2.
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Check Point Cloud Areas for " 
                     << pp.size() << " matches using Method 2" << std::endl;

  pp.refresh();
  if( pp.size() > 0 )
  {
      Rect overlap = pp.rect1 & pp.rect2;
      if( loglevel > 3 ) IMGLOG << " INFO: Raw Point Clouds: rect1: " << RECT2STR(pp.rect1) << " rect2: " << RECT2STR(pp.rect2) << std::endl;
      if( loglevel > 3 ) IMGLOG << " INFO: Raw Point Clouds: Overlap:" << RECT2STR(overlap) << std::endl;    
  }

/*

// Begin Old decimation code

// - - - - - - - - - - - - - - - - - - - - - - -
//
// APPLY DISTANCE FILTER BASED ON HOMOGRAPHY
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Grade and Decimate Homography in " 
                     << pp.size() << " matches." << std::endl;


     pairs[idx1][idx2].newPointPad( "Graded Homography Filter" );


     int hom_inliers = 0;
     int hom_outliers = 0;
     int hom_distcount[110];
     std::vector<std::vector<std::vector<int>>> cell;
     float distance[ points1.size() + 1 ];

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Homography Based Filter ..." );

     if( loglevel > 6 ) IMGLOG << " INFO: Creating cell matrix for decimation..." << std::endl;

     // Inserting elements into vector, owing to rounding we need one more than logic dictates
     for (int i = 0; i < ( xSize / cellSize ) + 1; i++)
     {
        vector< vector< int >> v1;  
        for (int j = 0; j < ( ySize / cellSize ) + 1; j++)
        {
            vector< int > v2;
            v1.push_back(v2);
        }
        cell.push_back(v1);
     }     
     
     if( loglevel > 3 ) IMGLOG << " INFO: Cell vector is [x=" << cell.size() << "][y=" << cell[0].size() << "]"
          << " From xSize=" << xSize << ", ySize=" << ySize << " with cellSize=" << cellSize << std::endl;

     // Clear Homography Distance accumulators     
     for (int i = 0; i < 110; i++) hom_distcount[i]=0;
     
     // For each point compute the error distance from the reprojected point ( a cheat to look inside the RANSAC )
     for (int i = 0; i < pp.size(); i++)
     {
        
         // perspectiveTransform() accepts multiple points, but we call with a trivial array of size() = 1.
         vector<Point2f> pts1, pts2;
         pts1.push_back( pp.points1[i] );
         perspectiveTransform( pts1, pts2, h);
         float dx = pts2[0].x - pp.points2[i].x;
         float dy = pts2[0].y - pp.points2[i].y;
         float dist = (float)(dx*dx + dy*dy);        
         distance[i] = dist;

         if( dist < maxDist ) // RANSAC reproj threshold 3 so 3 x 3 = 9
         { 
           if( loglevel > 6 ) sout << "Match: Good: ";
           hom_inliers++;
           hom_distcount[ (int)((float)dist * (float)10) ]++;
         }
         else 
         { 
           if( loglevel > 6 ) sout << "Match:  Bad: ";
           hom_outliers++;
           hom_distcount[ 101 ]++;
         }
         
         // for each point, find its cell and update a list of point indexes sorted by goodness
         
         // if the cell's list is empty, just shove it in.
         if( cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].size() == 0 )
         {
             if( loglevel > 6 ) sout << "cell[" << pp.points1[i].x / cellSize << "][" << pp.points1[i].y / cellSize << "] <-- {" << i << "}" << std::endl;
             cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].push_back(i);
         }
         // if the list has entries, scan and insert in order.
         else
         {
             if( loglevel > 6 ) sout << "cell[" << pp.points1[i].x / cellSize << "][" << pp.points1[i].y / cellSize << "] <-- {";
             int k = 0;
             bool isInserted = false;
             while( k < cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].size() )
             {
                // lines.insert(lines.begin() + line + 1, newCPlist[k] ); // Note should be pairs[idx1][idx2].newCPlist 
                
                 if( loglevel > 8 ) sout << " i=" << i << " k=" << k << " isInserted=" << isInserted << " dist=" << dist << std::endl;
                 if( loglevel > 8 ) sout << "distance[ cell[pp.points1[i=" << i << "].x=" << pp.points1[i].x << " / cellSize=" << cellSize << "][pp.points1[i" << i << "].y=" << pp.points1[i].y << " / cellSize][k=" << k << "] ] )" << std::endl;;
                 if( loglevel > 8 ) sout << "pp.points1.size()=" << pp.size() << " cell.size()=" << cell.size() << " distance.size()=" << points1.size() + 1 << std::endl;

                if( ! isInserted && dist < distance[ cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize][k] ] )
                {
                   if( loglevel > 6 )   sout << std::endl << "cell[" << pp.points1[i].x / cellSize << "][" << pp.points1[i].y / cellSize << "][" << k << "] <-- {" << i << "}" << std::endl;
                   cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].insert( cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].begin() + k, i );
                   isInserted = true;
                   if( loglevel > 6 ) sout << "(+)";
                }
                if( loglevel > 6 ) sout << cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize][k] 
                     << "(" << distance[ cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize][k] ] << ")";
                if( loglevel > 6 && k+1 < cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].size() ) sout << ",";               
                k++;
             }
             if( loglevel > 6 ) sout << "}" << std::endl;
         }
                    
         if( loglevel > 6 ) sout << "i=" << i << " (" << pp.points1[i].x << ", " << pp.points1[i].y << ")"
                            << " <--> (" << pp.points2[i].x << ", " << pp.points2[i].y << ") dx=" << dx << " dy=" << dy
                            << " dist=" << dist << std::endl;
      // i++;

      } // For each point

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Binning Into Cells" );

      int k = 0;
      int kcount = 0;
      while( k < 100 && kcount * 2 < hom_inliers )
      {
         kcount += hom_distcount[k];
         k++;
      }
      float hom_inlier50 = (float)((float)k / (float)10);

      if( loglevel > 3 )
      {
         // The fefault is to decimate to the best 50% of inliers, but this may kill good pairs
         if( hom_inliers * 2 < cpPerPairMin )
            IMGLOG << " WARN: ";
         else
            IMGLOG << " INFO: ";
               
         sout << "Inliers: " << hom_inliers << " ( 50% with error below: " << hom_inlier50 
                                << " ), Outliers: " << hom_outliers << " ( above error: " << maxDist << " )" << std::endl;

         if( hom_inlier50 < minDist ) IMGLOG << " INFO: 50% threshold of " << hom_inlier50 
                                             << " below minimum of " << minDist 
                                             << " adjusting RANSAC error threshold to minimum." << std::endl;

         if( hom_inliers / 2 < cpPerPairMin ) IMGLOG << " INFO: To few Inliers "  
                                             << " adjusting RANSAC error threshold to maximum." << std::endl;


      }
      
      if( hom_inlier50 < minDist ) hom_inlier50 = minDist;
      if( hom_inliers / 2 < cpPerPairMin ) hom_inlier50 = maxDist;

      if( loglevel > 6 ) IMGLOG << "Error Distance Distribution:" << std::endl;
      for (int i = 0; loglevel > 6 && i < 102 ; i++)
      {
         if( loglevel > 6 ) IMGLOG << "dist=" << (float)((float)i / (float)10) << " count=" << hom_distcount[i] << std::endl;
      }

      // dump cell info and count good cells.
      int hom_cellcountgood = 0;
      int hom_cellcountbad = 0;

      for (int i = 0; i < cell.size(); i++)
      {
         for (int j = 0; j < cell[i].size(); j++)
         {
            if( cell[i][j].size() > 0 )
            {
                // if the cell has at least 1 50% inlier its "good"
                if( distance[ cell[i][j][0] ] < hom_inlier50 )
                {
                   if( loglevel > 6 ) IMGLOG << " good ";
                   hom_cellcountgood++;
                }
                else
                {
                   hom_cellcountbad++;
                   if( loglevel > 6 ) IMGLOG << "  bad ";
                }
                if( loglevel > 6 ) sout << "cell[" << i << "][" << j << "][0..." << cell[i][j].size() << "] = {";
                for (int k = 0; k < cell[i][j].size(); k++)
                {
                   if( loglevel > 6 ) sout << cell[i][j][k] << "(" << distance[ cell[i][j][k] ] << ")";
                   if( loglevel > 6 && k+1 < cell[i][j].size() ) sout << ",";                               
                } // k
                if( loglevel > 6 ) sout << "}" << std::endl;
            } // if...
         } // j
      } // i   

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Cell inlier / outlier Analysis" );
      
      if( loglevel > 3 ) IMGLOG << " INFO: Cell Statistics: Total: " << cell.size() * cell[0].size() 
           << " inliers:" << hom_cellcountgood << " outliers:" << hom_cellcountbad << std::endl; 


// - - - - - - - - - - - - - - - - - - - - - - -
//
// PICK "GOOD" CONTROL POINTS AND PROPOSE HUGIN CHANGES
//
// - - - - - - - - - - - - - - - - - - - - - - - 

// ToDo: Cleanup - no longer modified Hugin proposal

  vector<Point2f> newPoints1, newPoints2;

  if( loglevel > 3 ) IMGLOG << " INFO: Decimate " << pp.size() << " Homography matches using RANSAC error distance " 
                            << hom_inlier50 << std::endl;
                     
      // Dump inliers to hugin config elements

      // Protect newCPlist from concurrency
      newcplist_guard.lock();
      int cpadded = 0; // number of control points dumped
      // use up to cellMaxCP from each cell up to cpPerPairMax per pair
      for(int k = 0; cpadded < cpPerPairMax && k < cellMaxCP; k++)
      { 
         if(loglevel > 8) sout << "k=" << k;
         for (int i = 0; i < cell.size(); i++)
         {
            if(loglevel > 8) sout << " i=" << k;
            for (int j = 0; j < cell[i].size(); j++)
            {
               if(loglevel > 8) sout << " j=" << j;
               // if the cell has a CP to look at AND we have not already used too many from this cell
               if( k < cellMaxCP && k < cell[i][j].size() )
               {
                   int c = cell[i][j][k]; // shorthand for use as an index
                   if( loglevel > 6 ) sout << " cell[" << i << "][" << j << "][" << k << "] = " << cell[i][j][k] 
                                    << ": distance.size()=" << points1.size() + 1 // distance is array, see def
                                    << ": pp.size()=" << pp.size() << " :";                       
                   // if the cell has at least 1 50% inlier its "good" OR if the minimum number per cell is not met 
                   if( distance[ cell[i][j][k] ] < hom_inlier50 || cellMinCP > k )
                   { 
                       // Dump config element
                       std::string s;

                       pairs[idx1][idx2].pointPad.back().points1.push_back( pp.points1[c] );
                       pairs[idx1][idx2].pointPad.back().points2.push_back( pp.points2[c] );
                       pairs[idx1][idx2].pointPad.back().pointType.push_back( pp.pointType[c] );
                       pairs[idx1][idx2].pointPad.back().lineIdx.push_back( pp.lineIdx[i] ); // no PTO line
                       pairs[idx1][idx2].pointPad.back().distance.push_back( pp.distance[i] );
                       
                       // Hugin rotates the images so origin in lower right.  bodging for the moment.
                       if( huginFlipOut )
                       s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
                            + " x" + std::to_string( xSize - pp.points1[c].x )   
                            + " y" + std::to_string( ySize - pp.points1[c].y )   
                            + " X" + std::to_string( xSize - pp.points2[c].x )   
                            + " Y" + std::to_string( ySize - pp.points2[c].y )
                            + " t0" ;  // no idea what t0 does
                       else 
                       s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
                            + " x" + std::to_string( pp.points1[c].x )   
                            + " y" + std::to_string( pp.points1[c].y )   
                            + " X" + std::to_string( pp.points2[c].x )   
                            + " Y" + std::to_string( pp.points2[c].y )
                            + " t0" ;  // no idea what t0 does

                      // newCPlist.push_back( s ); // add to the list ( should be pairs[idx1][idx2].newCPlist )
                       if( loglevel > 6 ) sout << s << std::endl;
                      // cpadded++;                
                   }
                   if( loglevel > 6 ) sout << std::endl;

               } // if...
            } // j
         } // i   
       } // k
       
       
      newcplist_guard.unlock();

   if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: " << pairs[idx1][idx2].pointPad.back().pointType.size() 
                        << " of " << pp.size() << " matches survived Homography Decimation." << std::endl;    

// End Old Decimation Code

*/



// - - - - - - - - - begin new decimation code - - - - - - - -

// - - - - - - - - - - - - - - - - - - - - - - -
//
// APPLY DISTANCE FILTER BASED ON HOMOGRAPHY
//
// - - - - - - - - - - - - - - - - - - - - - - - 

     if( loglevel > 3 ) IMGLOG << " INFO: Grade and Decimate " << pp.size() << " Matches Based on Homography." << std::endl;


     pairs[idx1][idx2].newPointPad( "Graded Homography Filter" );

     int hom_inliers = 0;
     int hom_outliers = 0;
     int hom_distcount[110];
     // Each cell is a list of indixed into the point pads list of points
     std::vector<std::vector<std::vector<int>>> cell;
     std::vector<std::vector<std::vector<int>>> cell_filt;
     // derr[][][] has a 1:1 correspondence to cell[][][] holding the RANSAC error, used as an index to sort cell[][]
     std::vector<std::vector<std::vector<float>>> derr;
     std::vector<std::vector<std::vector<float>>> derr_filt;
     // a flat list of RANSAC error distances, probabaly can be eliminated
     float distance[ points1.size() + 1 ];

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Homography Based Filter ..." );

     if( loglevel > 6 ) IMGLOG << " INFO: Creating cell matrix for decimation..." << std::endl;

     // Inserting elements into vector, owing to rounding we need one more than logic dictates
     if( true )
     {
        vector<int> vi;
        vector<vector<int>> vvi;
        vector<float> vf;
        vector<vector<float>> vvf;
        
        for (int j = 0; j < ( ySize / cellSize ) + 1; j++)
        {
            vvi.push_back(vi);
            vvf.push_back(vf);
        }
        
        for (int i = 0; i < ( xSize / cellSize ) + 1; i++)
        {
           cell.push_back(vvi);
           derr.push_back(vvf);
        }
      }

     // Clear Homography Distance accumulators     
     for (int i = 0; i < 110; i++) hom_distcount[i]=0;

     if( loglevel > 3 ) IMGLOG << " INFO: Cell vector is [x=" << cell.size() << "][y=" << cell[0].size() << "]"
          << " From xSize=" << xSize << ", ySize=" << ySize << " with cellSize=" << cellSize << std::endl;

     // project all points from image 1 to corresponding points in image2
     std::vector<Point2f> pts2;
     perspectiveTransform( pp.points1, pts2, h);
     
     // For each point compute the error distance from the reprojected point ( a cheat to look inside the RANSAC )
     for (int i = 0; i < pp.size(); i++)
     {
         float dx = pts2[i].x - pp.points2[i].x;
         float dy = pts2[i].y - pp.points2[i].y;
         float dist = (float)(dx*dx + dy*dy);  // note this is intentionally NOT =sqrt(...) to match RANSAC     
         distance[i] = dist;

         // apply a distance filter and tabulate inliers and outliers
         if( dist < maxDist ) // RANSAC reproj threshold 3 so 3 x 3 = 9
         { 
           hom_inliers++;
           hom_distcount[ (int)((float)dist * (float)10) ]++;
           cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].push_back(i);
           derr[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].push_back(dist);
           if( loglevel > 8 ) IMGLOG << " INFO: Inserted cell[" << pp.points1[i].x / cellSize 
                                     << "][" << pp.points1[i].y / cellSize 
                                     << "] <-- {" << i << "}, dist=" << dist 
                                     << " .size()=" << cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].size() << std::endl;
         }
         else 
         { 
           if( loglevel > 8 ) IMGLOG << " INFO: Rejected cell[" << pp.points1[i].x / cellSize 
                                     << "][" << pp.points1[i].y / cellSize 
                                     << "] <--X-- {" << i << "}, dist=" << dist 
                                     << " .size()=" << cell[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize].size() << std::endl;

           hom_outliers++;
           hom_distcount[ 101 ]++;
         }
      }
      
      // All points from the point pad are binned into cells, but currently unsorted.
      // This sorts each cell by error distance ( lowest i.e. "best" first )
      if( loglevel > 6 ) IMGLOG << " INFO: sorting cell matrix for decimation..." << std::endl;

      for( int i=0; i < cell.size(); i++ )
      {
         for( int j=0; j < cell[i].size(); j++ )
         {
            // do we nedd to sort?
            if( cell[i][j].size() > 1 )
            {
               vector<float> index;
               index = derr[i][j];

               if( loglevel > 6 ) IMGLOG << " INFO: UnSorted cell[" << i << "][" << j << "].size()=" 
                                         << cell[i][j].size() << " = " << cell[i][j] << std::endl;

               derr[i][j] = sortVecAByVecB( derr[i][j], index );
               cell[i][j] = sortVecAByVecB( cell[i][j], index );
               
               if( loglevel > 6 ) IMGLOG << " INFO: Sorted cell[" << i << "][" << j << "].size()=" 
                                         << cell[i][j].size() << " = " << cell[i][j] << std::endl;               
             }
             else
             if( cell[i][j].size() > 0 && loglevel > 6 ) IMGLOG << " INFO: Sorted cell[" << i << "][" << j << "].size()=" 
                                        << cell[i][j].size() << " = " << cell[i][j] << std::endl;               
             
          }  // for each cel[][]
       }  // For each cell[]       

      
      //  Go through the sorted cells and produce a filtered version eliminating "duplicated" control points keeping the
      // "best" ones that are the minimum distance apart, also enforce the minimum and maximum number of points / cell

      if( loglevel > 6 ) IMGLOG << " INFO: Decimating duplicates and enforcing cell depth..." << std::endl;

      for( int i=0; i < cell.size(); i++ )
      {
         for( int j=0; j < cell[i].size(); j++ )
         {

            if( loglevel > 6 ) IMGLOG << " INFO: Sorted cell[" << i << "][" << j << "].size()=" 
                                         << cell[i][j].size() << " = " << cell[i][j] << std::endl;

            for( int k=0; k < cell[i][j].size(); k++ )
            {
               // remove all lower confidence nearby points, work backwards
               for( int l = ( cell[i][j].size() - 1 ); l > k; l-- )
               {
                  
                  float dx = pp.points1[ cell[i][j][k] ].x - pp.points1[ cell[i][j][l] ].x;
                  float dy = pp.points1[ cell[i][j][k] ].y - pp.points1[ cell[i][j][l] ].y;
                  float dist = sqrt( (float)(dx*dx + dy*dy) );  // Note intentionally =sqrt(...) as NOT RANSAC error        

                  if( dist < (float)cellCPDupDist )
                  {
                     if( loglevel > 8 ) IMGLOG << " INFO: delete cell[" << i << "][" << j << "][" << l << "]=" 
                                               << cell[i][j][l] << " is too close to"
                                               <<       " cell[" << i << "][" << j << "][" << k << "]="
                                               << cell[i][j][k] << " distance=" << dist << "px units." << std::endl;

                     cell[i][j].erase(cell[i][j].begin() + l);
                     derr[i][j].erase(derr[i][j].begin() + l);
                  }                                   
                  else
                  if( loglevel > 8 ) IMGLOG << " INFO: retain cell[" << i << "][" << j << "][" << l << "]=" 
                                            << cell[i][j][l] << " is far enough from "
                                            <<       " cell[" << i << "][" << j << "][" << k << "]="
                                            << cell[i][j][k] << " distance=" << dist << "px units." << std::endl;

               }
            }  // for each cell[][][]
            
            if( loglevel > 6 ) IMGLOG << " INFO: Deduplicated cell[" << i << "][" << j << "].size()=" 
                                      << cell[i][j].size() << " = " << cell[i][j] << std::endl;

          }  // for each cell[][]
       }  // For each cell[]       


       BENCHMARK( pairs[idx1][idx2].benchmarks, "Binning Into Cells" );


// - - - - - - - - - - - - - - - - - - - - - - -
//
// PICK "GOOD" CONTROL POINTS AND PROPOSE HUGIN CHANGES
//
// - - - - - - - - - - - - - - - - - - - - - - - 

// ToDo: Cleanup - no longer modified Hugin proposal

  vector<Point2f> newPoints1, newPoints2;

  if( loglevel > 3 ) IMGLOG << " INFO: Decimate " << pp.size() << " Homography matches using RANSAC error distance " 
                            << maxDist << std::endl;
                     
      // Dump inliers to hugin config elements

      // Protect newCPlist from concurrency
      newcplist_guard.lock();
      
      int cpadded = 0; // number of control points dumped
      // use up to cellMaxCP from each cell up to cpPerPairMax per pair
      for(int k = 0; cpadded < cpPerPairMax && k < cellMaxCP; k++)
      { 
         if(loglevel > 8) sout << "k=" << k;
         for (int i = 0; i < cell.size(); i++)
         {
            if(loglevel > 8) sout << " i=" << k;
            for (int j = 0; j < cell[i].size(); j++)
            {
               if(loglevel > 8) sout << " j=" << j;

               // Compute pseudo random prime mod indices to prevent harvesting from adjacent cells. 
               
               int ir = ( i * 379 ) % cell.size();
               int jr = ( j * 973 ) % cell[i].size();

               // if the cell has a CP to look at AND we have not already used too many from this cell
               if( k < cellMaxCP && k < cell[ir][jr].size() && pairs[idx1][idx2].pointPad.back().size() < cpPerPairMax )
               {

                   int c = cell[ir][jr][k]; // shorthand for use as an index
                   
                   if( loglevel > 6 ) sout << " cell[" << i << "][" << j << "][" << k << "]"
                                    << " --mod--> cell[" << ir << "][" << jr << "][" << k << "] = " << cell[ir][jr][k] 
                                    << ": distance[" << c << "]=" << distance[c] << ", ";                       

                   // If an inlier and the minimum number of CP's for this cell have not been used
                   if( distance[c] < maxDist || cellMinCP > k )
                   { 
                       // Dump config element
                       std::string s;

                       pairs[idx1][idx2].pointPad.back().points1.push_back( pp.points1[c] );
                       pairs[idx1][idx2].pointPad.back().points2.push_back( pp.points2[c] );
                       pairs[idx1][idx2].pointPad.back().pointType.push_back( pp.pointType[c] );
                       pairs[idx1][idx2].pointPad.back().lineIdx.push_back( pp.lineIdx[c] ); // no PTO line
                       pairs[idx1][idx2].pointPad.back().distance.push_back( pp.distance[c] );
                       
                       // Hugin rotates the images so origin in lower right.  bodging for the moment.
                       if( huginFlipOut )
                       s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
                            + " x" + std::to_string( xSize - pp.points1[c].x )   
                            + " y" + std::to_string( ySize - pp.points1[c].y )   
                            + " X" + std::to_string( xSize - pp.points2[c].x )   
                            + " Y" + std::to_string( ySize - pp.points2[c].y )
                            + " t0" ;  // no idea what t0 does
                       else 
                       s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
                            + " x" + std::to_string( pp.points1[c].x )   
                            + " y" + std::to_string( pp.points1[c].y )   
                            + " X" + std::to_string( pp.points2[c].x )   
                            + " Y" + std::to_string( pp.points2[c].y )
                            + " t0" ;  // no idea what t0 does

                      // newCPlist.push_back( s ); // add to the list ( should be pairs[idx1][idx2].newCPlist )
                       if( loglevel > 6 ) sout << "added '" << s << "'" << std::endl;
                      // cpadded++;                
                   }
                   else
                   if( loglevel > 6 ) sout << " not added." << std::endl;

               } // if...
            } // j
         } // i   
       } // k
            
      newcplist_guard.unlock();

   if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: " << pairs[idx1][idx2].pointPad.back().pointType.size() 
                        << " of " << pp.size() << " matches survived Homography Decimation." << std::endl;    

// - - - - - - - - - end new decimation code - - - - - - - -

   // Use filtered points
   pp = pairs[idx1][idx2].pointPad.back();


// - - - - - - - - - - - - - - - - - - - - - - -
//
// Check that size of point clouds are approximately the same.  Method 1.
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Check Point Cloud Areas for " 
                     << pp.size() << " matches using Method 1" << std::endl;

     // std::vector<std::vector<std::vector<int>>> zone; // zone[x/cellSize][y/cellSize][0 | 1]

     if( loglevel > 6 ) IMGLOG << " INFO: Creating zone matrix for area analysis..." << std::endl;

     zone.clear(); // it already exists 
     // Inserting elements into vector, owing to rounding we need one more than logic dictates
     for (int i = 0; i < ( xSize / cellSize ) + 1; i++)
     {
        vector< vector< int >> v1;  
        for (int j = 0; j < ( ySize / cellSize ) + 1; j++)
        {
            vector< int > v2;
            v2.push_back( 0 ); // ...[0]
            v2.push_back( 0 ); // ...[1]            
            v1.push_back(v2);
        }
        zone.push_back(v1);
     }     
     
     // Tally points into zones
     for( int i=0; i < pp.size(); i++ )
     {
        zone[pp.points1[i].x / cellSize][pp.points1[i].y / cellSize][0] ++; // Tally the idx1 image points 
        zone[pp.points2[i].x / cellSize][pp.points2[i].y / cellSize][1] ++; // Tally the idx1 image points 
     }

     // int zone1cnt, zone2cnt;
     zone1cnt = zone2cnt = 0;

     // Inserting elements into vector, owing to rounding we need one more than logic dictates
     for (int i = 0; i < ( xSize / cellSize ) + 1; i++)
     {
        vector< vector< int >> v1;  
        for (int j = 0; j < ( ySize / cellSize ) + 1; j++)
        {
            if( zone[i][j][0] > 0 ) zone1cnt++;
            if( zone[i][j][1] > 0 ) zone2cnt++;
        }
     }     
     
     float newzoneSprawlRatio = (float)((float)abs( zone1cnt - zone2cnt ) / (float)( zone1cnt + zone2cnt ));
     
     if( loglevel > 3 ) IMGLOG << " INFO: Raw Zone Count: Image [" << idx1 << "]:" << zone1cnt << " Img [" << idx2 << "]: " << zone2cnt 
                          << " with cellSize=" << cellSize << std::endl;
     if( loglevel > 3 ) IMGLOG << " INFO: Raw Zone Analysis: Image [" << idx1 << "] --> [" << idx2 << "]: Sprawl Ratio: " 
                          << abs( zone1cnt - zone2cnt ) << " / " << zone1cnt + zone2cnt
                          << " = " << newzoneSprawlRatio << std::endl;

// - - - - - - - - - - - - - - - - - - - - - - -
//
// Check that size of point clouds are approximately the same.  Method 2.
//
// - - - - - - - - - - - - - - - - - - - - - - - 

     if( loglevel > 3 ) IMGLOG << " INFO: Check Homography Point Cloud Areas for " 
                     << pp.size() << " matches using Method 2" << std::endl;

     pp.refresh();
     Rect overlap = pp.rect1 & pp.rect2;
     if( pp.size() > 0 )
     {
         if( loglevel > 3 ) IMGLOG << " INFO: Homography Point Clouds: rect1: " << RECT2STR(pp.rect1) 
                              << " rect2: " << RECT2STR(pp.rect2) << std::endl;
         if( loglevel > 3 ) IMGLOG << " INFO: Homography Point Clouds: Overlap: " << RECT2STR(overlap) << std::endl;    
     }

     // Get bad overlap cause if any
     std::string boCause = badOverlapCause( pp );

     // Bad overlaps in sequential images usually mean a row boundary
     if( boCause.size() > 0 && ( idx2 - idx1 ) == 1 )
     {
        images[idx1].rowEndHint ++;
        images[idx2].rowStartHint ++;
     }
     
     // Tell user the bad news... if any.
     if( loglevel > 3 && boCause.size() > 0 ) IMGLOG << " INFO: Bad Overlap: " << boCause << std::endl;
      
     if( boCause.size() > 0 )
        if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: Filtered all " << pp.size() << " Control Points.  Bad Overlap." << std::endl;
     else
        if( loglevel > 3 ) IMGLOG << " INFO: FILTERING: Passed Overlap Tests, Likely Good Pair." << std::endl;

      // Dump inliers to hugin config elements

      // Protect newCPlist from concurrency
      // newcplist_guard.lock();
      cpadded = 0; // number of control points dumped
      // use up to cellMaxCP from each cell up to cpPerPairMax per pair
      for(int c = 0; c < pp.size(); c++)
      {                   
         // Dump config element
         std::string s;
         std::string p;
         p = "";
                       
         // Hugin rotates the images so origin in lower right.  bodging for the moment.
         if( huginFlipOut )
            s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
                + " x" + std::to_string( xSize - pp.points1[c].x )   
                + " y" + std::to_string( ySize - pp.points1[c].y )   
                + " X" + std::to_string( xSize - pp.points2[c].x )   
                + " Y" + std::to_string( ySize - pp.points2[c].y )
                + " t0" ;  // no idea what t0 does
         else 
            s =   "c n" + to_string(idx1) + " N" + to_string(idx2)
                + " x" + std::to_string( pp.points1[c].x )   
                + " y" + std::to_string( pp.points1[c].y )   
                + " X" + std::to_string( pp.points2[c].x )   
                + " Y" + std::to_string( pp.points2[c].y )
                + " t0" ;  // no idea what t0 does

         if( pp.size() >= cpPerPairMin )
         {
             if( newzoneSprawlRatio < 0.2 )
             {
                 if( ! boCause.size() > 0 )
                 {
                      cpadded++;
                 }
                 else
                 p = "# CVFIND # BAD OVERLAP DETECTED # ";     
             } // sprawl Check
             else
             p = "# CVFIND # SPRAWL TEST FAILED # ";             
         } // if cpPerPairMin requirement met
         else
         p = "# CVFIND # cpPerPairMin = " + to_string( cpPerPairMin ) + " # ";                     

         // Save the proposed like to the proposal
         pairs[idx1][idx2].newCPlist.push_back( p + s ); // add to the list
         if( loglevel > 6 ) sout << p << s << std::endl;
  
       } // c
       
       // newcplist_guard.unlock();

  if( loglevel > 3 && pp.size() < cpPerPairMin ) IMGLOG << " INFO: Hugin PTO Edits Proposed for image pair ( " 
                                                          << idx1 << ", " << idx2 << " ) : " 
                                                          << pp.size() << " less than " << cpPerPairMin << std::endl;             


  if( loglevel > 3 && newzoneSprawlRatio > 0.2 ) IMGLOG << " INFO: Hugin PTO Edits Proposed for image pair ( " 
                                                          << idx1 << ", " << idx2 << " ) : Sprawl Ratio "
                                                          << newzoneSprawlRatio << " > 0.2" << std::endl;             



// - - - - - - - - - - - - - - - - - - - - - - -
//
// Check that size of point clouds are approximately the same.  Method 2.
//
// - - - - - - - - - - - - - - - - - - - - - - - 

  if( loglevel > 3 ) IMGLOG << " INFO: Check Post Homography Point Cloud Areas for " 
                     << pp.size() << " matches using Method 2" << std::endl;

  pp.refresh();
  if( pp.size() > 0 )
  {
      Rect overlap = pp.rect1 & pp.rect2;
      if( loglevel > 3 ) IMGLOG << " INFO: Post Homography Point Clouds: rect1: " << RECT2STR(pp.rect1) << " rect2: " << RECT2STR(pp.rect2) << std::endl;
      if( loglevel > 3 ) IMGLOG << " INFO: Post Homography Point Clouds: Overlap: " << RECT2STR(overlap) << std::endl;    
  }

  if( cpadded > 0 )
  {

     if( loglevel > 8 ) cout << "TRACE: idx1:" << idx1 << " idx2:" << idx2 << " : " << idx2 - idx1 << std::endl;
     if( loglevel > 8 ) cout.flush();
    

     if( loglevel > 8 ) cout << "TRACE: Pre Expand === " << statsScalars.size() << " " << statsWeights.size() << " " << statsScalarHint.size() << std::endl;
     if( loglevel > 8 ) cout.flush();

     // dynamically grow the stats structures.  --> Like WTF.  since .size() returns an unsigned number it
     // casts everything as a uint and hoses the boundary condition. <--  
     while( (int)( (int)statsScalarHint.size() - (int)2) < (int)(idx2 - idx1) )
     {
     
     if( loglevel > 8 ) cout << "TRACE: Expand --> " << statsScalars.size() << " " << statsWeights.size() << " " << statsScalarHint.size() << std::endl;
     if( loglevel > 8 ) cout.flush();

        std::vector<float> foo1, foo2;
        statsScalarHint.push_back( (float)0 );
        statsScalars.push_back( foo1 );
        statsWeights.push_back( foo2 );

     if( loglevel > 8 ) cout << "TRACE: Expand <-- " << statsScalars.size() << " " << statsWeights.size() << " " << statsScalarHint.size() << std::endl;
     if( loglevel > 8 ) cout.flush();

     }
     if( loglevel > 8 )
     {
     cout << "TRACE: Post Expand === " << statsScalars.size() << " " << statsWeights.size() << " " << statsScalarHint.size() << std::endl;
     cout.flush();
     
     cout << "TRACE: Trying === " << statsScalars[idx2 - idx1].size();
     cout.flush();
     cout << " " << statsWeights[idx2 - idx1].size();
     cout.flush();
     cout << " " << statsScalarHint.size() << std::endl;
     cout.flush();
     }
      
     statsScalars[ idx2 - idx1 ].push_back( idealScalar );
     statsWeights[ idx2 - idx1 ].push_back( cpadded );
     float scalars = 0;
     float samples = 0;
     for( int i=0; i < statsScalars[ idx2 - idx1 ].size(); i++ )
     {
        scalars += ( (float)statsScalars[ idx2 - idx1 ][i] * (float)statsWeights[ idx2 - idx1 ][i] );
        samples += statsWeights[ idx2 - idx1 ][i];
        // Post the Hint if we have a large enough sample size
        if( samples > 1000 ) statsScalarHint[idx2 - idx1] = scalars / samples;
     }
     if( loglevel > 3 && statsScalarHint[idx2 - idx1] > 0 ) IMGLOG << " INFO: Stats: Distance: " << idx2 - idx1 
            << " Weighted Ideal Scalar: " << statsScalarHint[idx2 - idx1] 
            << " Actual: " << idealScalar 
            << " Diff %: " << (100 * ( statsScalarHint[idx2 - idx1] - idealScalar ) ) / statsScalarHint[idx2 - idx1] << std::endl;

     if( loglevel > 3 )
     {  
        IMGLOG << " INFO: Learned Ideal Scalars: { ";
        for( int i=0; i < statsScalarHint.size(); i++ )
           if( statsScalarHint[i] > 0 )
              sout << i << ":" << statsScalarHint[i] << "  ";
        sout << " }" << std::endl;
     }

  }

  if( loglevel > 3 ) IMGLOG << " INFO: Hugin PTO Edits Proposed for image pair ( " << idx1 << ", " << idx2 << " ) : " << cpadded << std::endl;             
                            
  if( loglevel > 3 || cpadded > 0 ) IMGLOG << " INFO: Homography Validated " << cpadded << " of " << pp.size() << " CP to add to Hugin proposal." << std::endl;      

/*
  if( loglevel > 3 && cpadded < cpPerPairMin ) IMGLOG << " WARN: Homography Found Less Than " 
                                      << cpPerPairMin << " control points to Hugin proposal." << std::endl;      
  if( loglevel > 3 && cpadded > cpPerPairMax ) IMGLOG << " INFO: Homography Found More Than " 
                                      << cpPerPairMin << " control points to Hugin proposal." << std::endl;      
*/

  // unProtect newCPlist from concurrency

     
  }
  else
  {
     if( pp.points1.size() > 0 )
     {
        if( loglevel > 3 ) IMGLOG << " WARN: Not enough control points to estimate Homography: " 
               << pp.points1.size() << " + " << pp.points2.size() << std::endl;
     }
  }

       BENCHMARK( pairs[idx1][idx2].benchmarks, "Dump Homography Filtered CP to HUGIN config elements" );
       
  //
  // Dump benchmarks if needed / wanted
  //
  
  if( benchmarking && pairs[idx1][idx2].benchmarks.size() > 0 )
     IMGLOG << "BENCHMARK RESULTS: ( ms = millisecond )" << std::endl;
  int b = 0;
  double tallied = 0;
  while( benchmarking && b < pairs[idx1][idx2].benchmarks.size() )
  {
     // ( should set diff to 0, since no conversion from int to ... exists ) 
     std::chrono::duration<double> diff = pairs[idx1][idx2].benchmarks[0].time - pairs[idx1][idx2].benchmarks[0].time;
     std::chrono::duration<double> elapsed = pairs[idx1][idx2].benchmarks[pairs[idx1][idx2].benchmarks.size()-1].time - pairs[idx1][idx2].benchmarks[0].time;
     if( b > 0 ) diff = pairs[idx1][idx2].benchmarks[b].time - pairs[idx1][idx2].benchmarks[b-1].time;
     tallied += diff.count(); // tally cumulative time captured between benchmarks
     if( loglevel > 6 || (int)( diff.count() * (double) 100 / ( elapsed.count() + 1 ) ) > 5 )
     IMGLOG << "BENCHMARK: Image Pair: " << left0padint(idx1, 4) << " --> " << left0padint(idx2, 4)
          << " : " << left_padint((int)( diff.count() * (double)1000 ), 10) << "ms " 
          << left_padint((int)( diff.count() * (double) 100 / ( elapsed.count() + 1 ) ), 3) << "% " << pairs[idx1][idx2].benchmarks[b].name << std::endl;
     b++;
  }
  if( benchmarking && pairs[idx1][idx2].benchmarks.size() > 1 )
  {
     std::chrono::duration<double> diff = pairs[idx1][idx2].benchmarks[pairs[idx1][idx2].benchmarks.size()-1].time - pairs[idx1][idx2].benchmarks[0].time;
     std::chrono::duration<double> elapsed = pairs[idx1][idx2].benchmarks[pairs[idx1][idx2].benchmarks.size()-1].time - pairs[idx1][idx2].benchmarks[0].time;
     IMGLOG << "BENCHMARK: Image Pair: " << left0padint(idx1, 4) << " --> " << left0padint(idx2, 4)
          << " : " << left_padint((int)( diff.count() * (double)1000 ), 10) << "ms "
          << left_padint((int)( tallied * (double) 100 / ( elapsed.count() + 1 ) ), 3) << "% " << "ELAPSED TIME" << std::endl;
  }

  if( loglevel > 3 ) IMGLOG << " INFO: Finished" << std::endl;

  if( alignLogging )
  {
     if( sout.coss.is_open() )
     {
        sout.coss << "##### CVFIND TASK LOG ENDS " << GetDateTime() << " FOR RUN ID: " << runID << " ######" << std::endl;
        sout.coss.flush();
        sout.coss.close();
     }   
  } 
  
  return;
  
} // alignImage(...)


//
// alignImages( ) - terse invocation using global config settings
//

void alignImages(int idx1, int idx2, std::string outputName )
{
  // Setup private debug log for each image pair.
     soutstream sout;

   if( alignLogging )
   {
//      sout.coss.open( outputName + "_" + runID + ".log", std::ios::out | std::ios::app );
      sout.coss.open( outputName + "_" + runID + ".log" );
      sout.coss << "##### CVFIND TASK LOG STARTS " << GetDateTime() << " FOR RUN ID: " << runID << " ######" << std::endl;
   }
  
   if( loglevel > 3 ) IMGLOG << " INFO: alignImages() using global defaults." << std::endl;


//   cout << "cout << 1.Fuck!!!!!" << std::endl;
//   sout << "sout << 1.Fuck!!!!!" << std::endl;
//   cerr << "cerr << 1.Fuck!!!!!" << std::endl;

//   cout.flush();
//   cerr.flush();

  // default mask is entire image
  int maskx1 = 0;
  int masky1 = 0;
  int maskx2 = 0;
  int masky2 = 0;

  if( detectorMasks )
  {
  // Mask off the center of the image forbidden by overlap
     maskx1 = images[idx1].imgGray.cols * (float)( overlapRatio + overlapMargin );
     masky1 = images[idx1].imgGray.rows * (float)( overlapRatio + overlapMargin );
     maskx2 = images[idx1].imgGray.cols * (float)( (float)1 - ( overlapRatio + overlapMargin ) );
     masky2 = images[idx1].imgGray.rows * (float)( (float)1 - ( overlapRatio + overlapMargin ) );
     // if masked   
     if( maskx1 > maskx2 || masky1 > masky2 ) 
     {
        maskx1 = masky1 = maskx2 = masky2 = 0;
     }

  }

     if( loglevel > 3 ) IMGLOG << " INFO: Detector Mask: (" << maskx1 << "," << masky1 << ") - (" << maskx2 << "," << masky2 << ")" 
                       << " LR: " << shootingLR << " TB: " << shootingTB << " Axis: " << shootingAxis << std::endl;
                   //  << " Rotate: " << rotate_mask 

     // Construct masks for this image pair
     Mat overlapMask(images[idx1].imgGray.size(), CV_8U, Scalar(255));  // type of mask is CV_8U
     // roi is a sub-image of mask specified by cv::Rect object
     Mat roi(overlapMask, cv::Rect( Point2f(maskx1,masky1), Point2f(maskx2,masky2) ) );
     // we set elements in roi region of the mask to 255 
     roi = Scalar(0);
  
     BENCHMARK( pairs[idx1][idx2].benchmarks, "Mask Creation ..." );

     if( loglevel > 8 )
     {
        Mat maskedImg1 = overlapMask;
        // Copy the image data into the masked area of the image
        images[idx1].imgGray.copyTo(maskedImg1, maskedImg1);
        if(true)imwrite(outputName + "_dmask1.png", maskedImg1);
        Mat maskedImg2 = overlapMask;
        // Copy the image data into the masked area of the image
        images[idx2].imgGray.copyTo(maskedImg2, maskedImg2);
        if(true)imwrite(outputName + "_dmask2.png", maskedImg2);
     }

     alignImages(idx1, idx2, outputName, overlapMask );

     if( alignLogging )
     {
        if( sout.coss.is_open() )
        {
           sout.coss << "##### CVFIND TASK LOG ENDS " << GetDateTime() << " FOR RUN ID: " << runID << " ######" << std::endl;
           sout.coss.flush();
           sout.coss.close();
        }   
     } 

     return;
}


 
//
// loopAnalyze() - Analyze transitive homography for loops of possibly connected image pairs,
//                  identify patently good pairs where the transitive error for the loop is low
//                  and mark all images in such loops as "valid".   Final homography is used for
//                  this test and so is computationally trivial.
//
//                  And pairs not marked as "valid" are considered bad.  For loops containg bad pairs,
//                  a search of all trial homographies for each bad pair is conducted.   Each good pair's
//                  final homography is used for assessing the error of the loop.  Each bad pair contributes
//                  all of its trial homographies, possibly 60 trials.   the number of permutations is
//                  ~60^(number of bad pairs in the loop)
//
//                  There can be many loops containg a given bad pair.  Loops curently considered are those
//                  containg the pair of images and the neighbor images they share
//                  [ IMG 1 ] <--> { List of immediate neighbor images in between } <--> [ IMG 2 ]
//                  forming 3 neighbor chains, 1<-->C<-->2  Other chains contain 4 images:
//                  A<-->1<-->2<-->B.  Loop homography is accomplished on a chain by adding its 1st
//                  element to its last:  1-->C-->2-->1    If a high confidence loop can be found
//                  via transitive homography, this implies that all of these pairs are good.
//
//                  if a high confidence permutation ( combination of known valid homographies and trials
//                  from bad pairs ) is found, it is added to a list.   THis list is sorted by error distance
//                  and the best one is used.   The ideal scalar for that combination of trial homographies is
//                  pushed into the ptoImagePair structure and we again call alignImage for that pair to
//                  to force it to rework the homography.   Possibly after enabling additional detectors or imposing
//                  a filter mask. 

std::string loopAnalyze( bool i_repair )
{

  std::stringstream sout;

  int pairsValid = 0;
  int pairsBad = 0;
  int pairsRepaired = 0;
  int pairsReRepaired = 0;
  
  //
  // Clear votes from prior runs
  //
  for( int i=0; i < images.size(); i++ )
  {
     for( int j=i+1; j < images.size(); j++ )
     {
        pairs[i][j].votesLB = 0; // Patently bad
        pairs[i][j].votesL0 = 0;
        pairs[i][j].votesL1 = 0;
        pairs[i][j].votesL2 = 0; // Highest confidence
     }
  }       

  // dotrace = true;

  if( loglevel > 0 ) sout << " INFO: Loop Homography Computation ..." << std::endl;

  for( int i = 0; i + 1 < images.size(); i++ )
  {
     if( loglevel > 8 ) { cout.flush(); cout << "LHA: i=" << i << std::endl; cout.flush(); }
      
     if( loglevel > 8 ) sout << " INFO: LHA: calling transitiveChains( " << i << ", " << i+1 << ", 1 )" << std::endl;
     if( loglevel > 8 ) cout.flush();

     std::vector<std::vector<int>> chains = transitiveChains( i, i+1, 1 );

     if( loglevel > 8 ) sout << " INFO: LHA: transitiveChains( ) returned " << chains.size() << std::endl;
     if( loglevel > 8 ) cout.flush();

     for( int c = 0; c < chains.size(); c++ )
     {
        if( chains[c].size() > 1 )
        {
           vector<int> loop = chains[c];
           loop.push_back( loop[0] ); // take chain and make it a loop.   e.g. A, B, C, A

           if( loglevel > 8 )  sout << " INFO: LHA: Checking loop " << c << " : { " << loop << " } " << std::endl;

           bool isValid = validChain( loop, false ); // Verify minimum CP count is met ( already invalid loop )

           if( loglevel > 8 ) sout << " INFO: LHA: validChain( { " << loop << " } ) = " << isValid << std::endl;

           if( isValid )
           {
              Point2f error = transitiveErrorSub( loop );
              float scalarError = sqrt( ( error.x * error.x ) + ( error.y * error.y ) );

              // Save results in each pairs objects ( except for the trivial case of { A-->B-->A }
              if( loop.size() > 3 )
                 for( int n = 0; n + 1 < loop.size(); n++ )
                 {
                     pairs[min(loop[n], loop[n+1])][max(loop[n], loop[n+1])].loops.push_back( loop );
                     pairs[min(loop[n], loop[n+1])][max(loop[n], loop[n+1])].loopError.push_back( scalarError );
                 }
    
           } // isValid
           else
           {
              if( loglevel > 8 ) sout << " INFO: LHA: i=" << i << " c=" << c << " loop = { " << loop << " } ) not valid. " << std::endl;
           } 
        }
     }
  }

  if( loglevel > 4 ) sout << " INFO: Loop Homography Raw Results ..." << std::endl;
  cout.flush();

  //
  //   Go through each pair, sort and report on the loop homography results
  //   sets pairs[][].loops and pairs[][].loops.loopError
  //   
  for( int i=0; i < images.size(); i++ )
  {
     for( int j=i+1; j < images.size(); j++ )
     {
        // sort loops and errors by error
        if( pairs[i][j].loops.size() > 1 )
        {

            for( int l = 0; loglevel > 8 && l < pairs[i][j].loops.size(); l++ )
               sout << " INFO: Pair ( " << i << ", " << j << " )  Loop = " << pairs[i][j].loops[l] 
                    << " Error = " << pairs[i][j].loopError[l] << " (Unsorted) " <<  std::endl;

            // what a fucking stupid shitshow, Excel has been able to do this since 1992
            std::vector<std::vector<int>> loops = pairs[i][j].loops;
            std::vector<float> errors = pairs[i][j].loopError;
            std::vector<float> index = pairs[i][j].loopError;
            pairs[i][j].loops = sortVecAByVecB( loops, index );
            pairs[i][j].loopError = sortVecAByVecB( errors, index );

            for( int l = 0; ( ( loglevel > 4 && l == 0 ) || loglevel > 6 ) && l < pairs[i][j].loops.size(); l++ )
               sout << " INFO: Pair ( " << i << ", " << j << " )  Loop = " << pairs[i][j].loops[l] 
                    << " Error = " << pairs[i][j].loopError[l] << std::endl;

        } 
     } // For each image ( outer loop )  
  } // For each image ( outer loop )  
  
  if( loglevel > 0 ) sout << " INFO: Loop Homography Vote Tabulation ..." << std::endl;
  cout.flush();
  
  //
  //   Go through each pair, check the "best" result and report if the image pair is bad or not
  //   based on the loop homiograph results in pairs[][].loops and pairs[][].loops.loopError
  //   
  for( int i=0; i < images.size(); i++ )
  {
     for( int j=i+1; j < images.size(); j++ )
     {
        // sort loops and errors by error
        if( pairs[i][j].loops.size() > 0 )
        {
            if( loglevel > 4 ) sout << " INFO: Pair ( " << i << ", " << j << " )  Loop = " << pairs[i][j].loops[0] 
                                    << " Error = " << pairs[i][j].loopError[0];
                 
            if( pairs[i][j].loopError[0] < loopL2Threshold * (float)pairs[i][j].loops[0].size() )
            {
               if( loglevel > 4 ) sout << " < " << loopL2Threshold * (float)pairs[i][j].loops[0].size() 
                                       << " HIGH CONFIDENCE " <<  std::endl;
               pairs[i][j].votesL2++;
            }
            else
            if( pairs[i][j].loopError[0] < loopL1Threshold * (float)pairs[i][j].loops[0].size() )
            {
               if( loglevel > 4 ) sout << " < " << loopL1Threshold * (float)pairs[i][j].loops[0].size() 
                                       << " MEDIUM CONFIDENCE " <<  std::endl;
               pairs[i][j].votesL1++;
            }
            else
            if( pairs[i][j].loopError[0] < loopL0Threshold * (float)pairs[i][j].loops[0].size() )
            {
               if( loglevel > 4 ) sout << " < " << loopL0Threshold * (float)pairs[i][j].loops[0].size() 
                                       << " LOW CONFIDENCE " <<  std::endl;
               pairs[i][j].votesL0++;
            }
            else
            {
               if( loglevel > 4 ) sout << " > " << loopL0Threshold * (float)pairs[i][j].loops[0].size() 
                                       << " BAD PAIR IN LOOP " <<  std::endl;
               pairs[i][j].votesLB++;
            }
        }
     } // For each image ( outer loop )  
  } // For each image ( outer loop )  
  

  if( loglevel > 0 ) sout << " INFO: Loop Homography Final Decision ..." << std::endl;
  cout.flush();
  
  //
  //   Go through each pair, check the "best" result and report if the image pair is bad or not
  //   based on the loop homiograph results.  Reporting only.
  //   
  for( int i=0; i < images.size(); i++ )
  {
     for( int j=i+1; j < images.size(); j++ )
     {
        // sort loops and errors by error
        if( pairs[i][j].loops.size() > 0 )
        {
            if( loglevel > 4 ) sout << " INFO: Pair ( " << i << ", " << j << " )  Votes: Bad = " << pairs[i][j].votesLB
                          << " L2 = " << pairs[i][j].votesL2
                          << " L1 = " << pairs[i][j].votesL1
                          << " L0 = " << pairs[i][j].votesL0 << " Status:" ;

            if( pairs[i][j].votesL0 > 0 || pairs[i][j].votesL1 > 0 || pairs[i][j].votesL2 > 0 )
            {
               if( loglevel > 4 ) sout << " PAIR VALID " << std::endl;
               pairsValid++;
            }
            else
            {
               if( loglevel > 4 ) sout << " PAIR BAD " << std::endl;
               pairsBad++;
            }
        }
     } // For each image ( outer loop )  
  } // For each image ( outer loop )  
  
  // dotrace = false;

  if( loglevel > 0 ) sout << " INFO: Loop Homography Repair Analysis ..." << std::endl;
  cout.flush();
  
  //
  //   Go through each pair, check the "best" result and report if the image pair is bad or not
  //   based on the loop homiograph results
  //   
  for( int i=0; i < images.size(); i++ )
  {
     for( int j=i+1; j < images.size(); j++ )
     {
        // sort loops and errors by error
        if( pairs[i][j].loops.size() > 0 )
        {

            if( pairs[i][j].votesL0 > 0 || pairs[i][j].votesL1 > 0 || pairs[i][j].votesL2 > 0 )
            {

               if( loglevel > 6 ) sout << " INFO: Pair ( " << i << ", " << j << " )  Votes: Bad = " << pairs[i][j].votesLB
                          << " L2 = " << pairs[i][j].votesL2
                          << " L1 = " << pairs[i][j].votesL1
                          << " L0 = " << pairs[i][j].votesL0 << " Status:" ;

               if( loglevel > 6 ) sout << " PAIR VALID " << std::endl;
            }   
            else
            {
                  std::vector<ptoTrialPermutation> goodperms;
                  std::vector<float> gooderrors;
                  std::vector<float> gooderrorsidx;
            
               if( loglevel > 4 ) sout << " INFO: Pair ( " << i << ", " << j << " )  Votes: Bad = " << pairs[i][j].votesLB
                          << " L2 = " << pairs[i][j].votesL2
                          << " L1 = " << pairs[i][j].votesL1
                          << " L0 = " << pairs[i][j].votesL0 << " Status:" ;
            
               if( loglevel > 4 ) sout << " PAIR BAD " << std::endl;
               if( loglevel > 4 ) sout << " INFO: Finding Repair Options for pair[" << i << "][" << j << "]:" << std::endl;
               
               // For each loop contating this pair...
               for( int k = 0; k < pairs[i][j].loops.size(); k++ )
               {
                  if( loglevel > 4 ) sout << " INFO: Finding Repair Options for pair[" << i << "][" << j << "].loops[" << k 
                       << "] = " << pairs[i][j].loops[k] << std::endl;

                  // dotrace = true;
                  std::vector<ptoTrialPermutation> perms = transitiveTrialError( pairs[i][j].loops[k], true, 1 );
                  // dotrace = false;

                  if( loglevel > 4 ) sout << " INFO: Repair Options for pair[" << i << "][" << j << "].loops[" << k 
                       << "] = " << pairs[i][j].loops[k] << " Returned " << perms.size() << " permutations." << std::endl;

                  bool repairable = false;

                  // filter permutations for only the "good" ones with high confidence loop homography                
                  for( int r = 0; r < perms.size(); r++ )
                  {
                     if( perms[r].error / perms[r].chain.size() < loopL0Threshold )
                     {

                         if( loglevel > 3 )
                         {
                            sout << " INFO: Repair solution for pair[" << i << "][" << j << "] Permutation: " << r 
                                 << " Chain: " << perms[r].chain  
                                 << " Trial: " << perms[r].trialChain
                                 << " Error: " << perms[r].errorPoint << " Scalar: " << perms[r].error;
                                 
                            if( perms[r].error / perms[r].chain.size() < loopL2Threshold )       
                                sout << " Threshold: " << loopL2Threshold * perms[r].chain.size() << " High";
                            else if( perms[r].error / perms[r].chain.size() < loopL2Threshold )       
                                sout << " Threshold: " << loopL1Threshold * perms[r].chain.size() << " Medium";
                            else if( perms[r].error / perms[r].chain.size() < loopL2Threshold )       
                                sout << " Threshold: " << loopL0Threshold * perms[r].chain.size() << " Low";
                            else
                                sout << " Threshold: " << loopL0Threshold * perms[r].chain.size() << " Bad";
                            
                            sout << " Confidence " << std::endl;
                         } // If logging     

                         goodperms.push_back( perms[r] );
                         gooderrors.push_back( perms[r].error );
                         
                     } // If "good" permutation
                         
                  } // for each 

               }  // For each loop
               
               // At this point goodperms contains all the, uh, good perms across all loops.

               std::vector<ptoTrialPermutation> sortperms;
                  
               // Sort "good" permutations by error and report out the best one(s)

               sortperms = sortVecAByVecB( goodperms, gooderrors ); 

               // Dump unsorted table
               for( int r = 0; r < goodperms.size(); r++ )
               {
                   if( loglevel > 6 )
                      sout << " INFO: Repair solution for pair[" << i << "][" << j << "] Permutation: " << r 
                           << " Chain: " << goodperms[r].chain  
                           << " Trial: " << goodperms[r].trialChain
                           << " Error: " << goodperms[r].errorPoint << " Scalar: " << goodperms[r].error
                           << " Threshold: " << loopL2Threshold * goodperms[r].chain.size() << " (unsorted) " << std::endl;
               }

               // Dump sorted table
               for( int r = 0; r < sortperms.size(); r++ )
               {
                  if( loglevel > 6 )
                      sout << " INFO: Repair solution for pair[" << i << "][" << j << "] Permutation: " << r 
                           << " Chain: " << sortperms[r].chain  
                           << " Trial: " << sortperms[r].trialChain
                           << " Error: " << sortperms[r].errorPoint << " Scalar: " << sortperms[r].error
                           << " Threshold: " << loopL2Threshold * sortperms[r].chain.size() << " (sorted) " << std::endl;
               }
                  
               // For each permutation CHAIN
               for( int r = 0; r < sortperms.size() && ( r == 0 || loglevel > 6 ); r++ )
               {
                    
                  if( loglevel > 3 )
                     sout << " INFO: Repair proposal for pair[" << i << "][" << j << "] Permutation: " << r 
                          << " Chain: " << sortperms[r].chain  
                          << " Trial: " << sortperms[r].trialChain
                          << " Error: " << sortperms[r].errorPoint << " Scalar: " << sortperms[r].error
                          << " Threshold: " << loopL2Threshold * sortperms[r].chain.size() << std::endl;

                  // For each LINK in the current permutation 
                  for( int q = 0; q + 1 < sortperms[r].chain.size(); q++ )
                  {
                            
                     int il = min(sortperms[r].chain[q], sortperms[r].chain[q+1]);
                     int ih = max(sortperms[r].chain[q], sortperms[r].chain[q+1]);
                     ptoPointPad pp;
                              
                     // If this is a validated pair, use the final homography / point pad
                     if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
                     {
                         int ppidx = pairs[il][ih].currentPointPad();
                         pp = pairs[il][ih].pointPad[ppidx];
                     }
                     else
                     {    
                        pp = pairs[il][ih].trialPointPad[sortperms[r].trialChain[q]]; // I guess? 
                     }
                                
                     pp.refresh(); // recalculate the pp stuff

     
                     // There are 3 genders          
                     if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
                     {
                           if( loglevel > 3 ) sout << "         --       (valid) ";
                     }
                     else
                     {
                        if( pairs[il][ih].repairScalar > 0 )
                        {
                           if( loglevel > 3 ) sout << "         --    (repaired) ";
                        }
                        else
                        {
                           if( loglevel > 3 ) sout << "         --  (repairable) ";
                        }
                     }
                     //
                     if( loglevel > 3 )
                        sout << " pair[" << sortperms[r].chain[q] << "][" << sortperms[r].chain[q+1] 
                             << "]-->" << " Matches: " << pp.size()
                             << " COM Scalar= " << pp.comscalar << " Slope= " << pp.comslope 
                             << " Area= " << pp.rect1.area() << std::endl;
                                    
                  } // For each LINK in a GOOD permutation              

               } // For each good PERMutation

               // At this point sortperms[0] contains the highest confidence repair option

               //
               // Actually repair the broken pairs.   For the best repair option, itterate through
               // each link and for those which are not validated, try running image processing
               // again using the "very special" ( cough, cough ) detector masked off for the overlap area
               // of the repair solution.  This breaks the keypoints and features for the image since they are central. 
               //               
               // For the first permutation only ( keeping as a loop anyway, don't judge )

               if( i_repair ) if( loglevel > 1 ) sout << " INFO: Loop Homography Performing Repairs ..." << std::endl;

               for( int r = 0; i_repair && r == 0 && r < sortperms.size(); r++ )
               {
                    
                  if( loglevel > 3 )
                     sout << " INFO: Repairing pair[" << i << "][" << j << "] Permutation: " << r 
                          << " Chain: " << sortperms[r].chain  
                          << " Trial: " << sortperms[r].trialChain
                          << " Error: " << sortperms[r].errorPoint << " Scalar: " << sortperms[r].error
                          << " Threshold: " << loopL2Threshold * sortperms[r].chain.size() << std::endl;

                  // For each LINK in the current permutation 
                  for( int q = 0; q + 1 < sortperms[r].chain.size(); q++ )
                  {
                     int il = min(sortperms[r].chain[q], sortperms[r].chain[q+1]);
                     int ih = max(sortperms[r].chain[q], sortperms[r].chain[q+1]);
                     ptoPointPad pp;
                              
                     // If this is a validated pair, use the final homography / point pad
                     if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
                     {
                         int ppidx = pairs[il][ih].currentPointPad();
                         pp = pairs[il][ih].pointPad[ppidx];
                     }
                     else
                     {    
                        pp = pairs[il][ih].trialPointPad[sortperms[r].trialChain[q]]; // I guess? 
                     }
                                
                     pp.refresh(); // recalculate the pp stuff
                              
                     // There are 3 genders          
                     if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
                     {
                           if( loglevel > 3 ) sout << "         --      (valid) ";
                     }
                     else
                     {
                        if( pairs[il][ih].repairScalar > 0 )
                        {
                           if( loglevel > 3 ) sout << "         --   (repaired) ";
                        }
                        else
                        {
                           if( loglevel > 3 ) sout << "         --  (REPAIRING) ";
                        }
                     }
                     //
                     if( loglevel > 3 )
                        sout << " pair[" << sortperms[r].chain[q] << "][" << sortperms[r].chain[q+1] 
                             << "]-->" << " Matches: " << pp.size()
                             << " COM Scalar= " << pp.comscalar << " Slope= " << pp.comslope 
                             << " Area= " << pp.rect1.area() << std::endl; 
                                
                     if( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
                     {
 
                     }
                     else
                     {
                        // Did we already try to repair it? 
                        if( pairs[il][ih].repairScalar == 0 )
                        {
                           pairs[il][ih].repairScalar = pp.comscalar; // This tags the pair for rework.
                           std::string prefix;
                           prefix = prefix + "cvfind_repair_" + left0padint(il, 4) + "_" + left0padint(ih, 4);          
                           if( loglevel > 3 ) sout << "            -- alignImages( " 
                                               << il << ", " << ih << ", '" << prefix << "' ) ... " << std::endl;
                           
                           // Enable additional detectors ( Probably akaze is doing the work ) 
                           // detectorDenseSift = true;
                           detectorAkaze     = true;
                           // detectorLine      = true;
                           // detectorBrisk     = true;
                                               
                           // blow away the prior proposed hugin edits                    
                           pairs[il][ih].newCPlist.clear();
                                               
                           alignImages( il, ih, prefix );
                           
                           if( loglevel > 3 ) sout << "            -- alignImages( " 
                                               << il << ", " << ih << ", '" << prefix 
                                               << "' ) Found " << pairs[il][ih].newCPlist.size() 
                                               << " Control Points."  << std::endl;
                           pairsRepaired++;
                        }
                        else
                        {
                           pairsReRepaired++;
                        }
                        // Check to see if this fixed anything...
                     }
                                  
                  } // For each LINK in a GOOD permutation              

               } // For each good PERMutation
               
            }  // if...else pair not valid
        }
     } // For each image ( outer loop )  
  } // For each image ( outer loop )  
  
  if( loglevel > 0 ) sout << " INFO: Loop Homography Pairs Summary: Valid=" << pairsValid << " Bad=" << pairsBad;
  if( i_repair ) if( loglevel > 0 ) sout << " Repaired=" << pairsRepaired << " Ignored Re-Repairs=" << pairsReRepaired;
  if( loglevel > 0 ) sout << std::endl;
  
  // dotrace = false;
  return sout.str();

}

// - - - - - - - - - - - - - - - - - - - - - - - - - 
//
//  main() 
//
// - - - - - - - - - - - - - - - - - - - - - - - - -
int main(int argc, char *argv[], char *envp[])
{

  // Horrific assemblage of global variables

  // bool debug = false;  // changed to a global

	// READ & PARSE

// if(loglevel > 8) sout << "TRACE: " << __FILE__ << ": " << __LINE__ << " argc=" << argc << std::endl;

// Make sure the ENUM and string tables are in sync. if not then there are mysterious segfaults.
if( detectorName.size() - 1 != detectorType::COUNT )
{
   sout << "FATAL: detectorName.size() - 1 != detectorType::COUNT.  Its a bug!" << std::endl; 
   return 1;
}

runID = GetDateTime(); // Set Run ID to a timestamp, used to identify logs from a given run.

for( int i=0; i < argc ; i++ )
{
   cmdline = cmdline + argv[i] + " ";
}

if( loglevel > 6 ) sout << "cmdline=" << cmdline;

// display any effective arguments ( excluding popped ones )
if( loglevel > 6 )
{
  sout << "Effective arguments:" << std::endl;
  // Print arguments because I have not coded in 20 years 	
  int i = 0;
  while ( i < argc )
  {
     sout << "[ " << i << " ]" << argv[i] << std::endl;
     i++;
  }
  sout << std::endl;  
}

// if(loglevel > 8) sout << "TRACE: " << __FILE__ << ": " << __LINE__ << " argc=" << argc << std::endl;

// if first argument id "-d" then turn on debug mode - same as --debug but earlier 
if( true )
{
    debug = false; // an assumption
    if ( argc > 1 && 0 == strcmp(argv[1], "-d") )
    {
        debug = true; // no longer an assumption
        // remove the "-d" argument
        // but do copy the NULL at the end.
        for(int i=1; i<argc; ++i)
            argv[i]  = argv[i+1];
        --argc;
    }
}


// if(loglevel > 8) sout << "TRACE: " << __FILE__ << ": " << __LINE__ << std::endl;

// display any effective arguments ( excluding popped ones )
if( loglevel > 6 )
{
  sout << "Effective arguments:" << std::endl;
  // Print arguments because I have not coded in 20 years 	
  int i = 0;
  while ( i < argc )
  {
     sout << "[ " << i << " ]" << argv[i] << std::endl;
     i++;
  }
  sout << std::endl;  
}

// if(loglevel > 8) sout << "TRACE: " << __FILE__ << ": " << __LINE__ << std::endl;

if( argc > 0 )
{
    std::string current_option;
    int j;
    j = 1;
    while( j < argc )
    {
       bool parsed = false;
       
       if ( ! parsed && j < argc ) current_option = argv[j]; 
       
       OPTIONBOOL(--debug,debug,true);
       if( debug ) loglevel = 7;

       // is it -o output.pto ?
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], "-o") )
       {
           std::string theoption = argv[j];
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           // look for required parameter and ensure it does not begin with a "-" 
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) )
           {
              outputPTO = argv[j];
              if( loglevel > 6 ) sout << "Parsed " << theoption << " " << outputPTO << std::endl;
              // remove the file name
              for(int i=1; i<argc; ++i)
                  argv[i]  = argv[i+1];
              --argc;
           }
           else
           {
              sout << "ERROR: option "<< theoption << " requires a filename." << std::endl;
              return 1;              
           }

           parsed = true;
       }

      OPTIONINT(--mindist,minDist,0.1f,maxDist);
      OPTIONINT(--maxdist,maxDist,minDist,9.0f);  // Note > 9 breaks decimation code!
      
      OPTIONINT(--threads,threadsMax,1,10000);
      OPTIONINT(--ncores,threadsMax,1,10000);
      OPTIONINT(-n,threadsMax,1,10000);
      OPTIONINT(--linearmatchlen,linearMatchLen,1,10000);
      OPTIONINT(--rowsize,rowSize,4,10000);  // rows size below 4 is non-optimizable    
      OPTIONINT(--rowsizetolerance,rowSizeTolerance,0,rowSize);
      OPTIONINT(--peakWidth,peakWidth,1,rowSize);
      OPTIONINT(--centerFilter,centerFilter,0,2);
      OPTIONINT(--trialhomsize,trialHomSize,5,500);
      OPTIONINT(--trialhomstep,trialHomStep,min( 1, trialHomSize / 10 ),trialHomSize);
      OPTIONINT(--cellsize,cellSize,10,100);
      OPTIONINT(--cellmaxcp,cellMaxCP,1,1000);
      OPTIONINT(--cellmincp,cellMinCP,0,cellMaxCP);
      OPTIONINT(--cellcpdupdist,cellCPDupDist,0,cellSize);
      OPTIONINT(--cpperpairmax,cpPerPairMax,cpPerPairMin,10000);
      OPTIONINT(--cpperpairmin,cpPerPairMin,0,cpPerPairMax);
      OPTIONINT(--neighborsmin,neighborsMin,0,neighborsMax);
      OPTIONINT(--neighborsmax,neighborsMax,neighborsMin,10000);
      OPTIONINT(--featuremax,featureMax,10,200000);
      OPTIONINT(--prescale,preScaleFactor,1,100);
      OPTIONINT(--exitdelay,exitDelay,0,10000);
      OPTIONINT(--loglevel,loglevel,0,10);
      OPTIONINT(--overlap,overlapRatioPct,0,100 - overlapMarginPct);
      OPTIONINT(--overlaptolerance,overlapMarginPct,0,100 - overlapRatioPct);
      
      OPTIONINTCSV(--images,mandatoryImages,0,10000);
      // if( true || loglevel > 6 ) sout << " INFO: ( --images ) mandatorImages=" << mandatoryImages << std::endl;
      OPTIONINTCSV(--noimages,forbiddenImages,0,10000);
      // if( true || loglevel > 6 ) sout << " INFO: ( --noimages ) forbiddenImages=" << forbiddenImages << std::endl;

       // is it --detect / --nodetect [ orb, akaze, brisk, blob, corner, line ]  ?
       if ( ! parsed && j < argc && ( 0 == strcmp(argv[j], "--detect") || 0 == strcmp(argv[j], "--nodetect") ) )
       {
           std::string theoption = argv[j];
           bool b_theoption = true;
           if( 0 == strcmp(argv[j], "--nodetect") ) b_theoption = false;

           std::string theparm;
           int thenum = 0;
           bool inbounds = false;
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           // look for required parameter and ensure it does not begin with a "-" 
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) )
           {
              // parse as a comma separated list, e.g. brisk,orb,none,line,all etc.
              std::stringstream ss(argv[j]);
              while (ss.good())
              {
                 inbounds = false;
                 bool b_opt = b_theoption;
                 std::string s_parm;
                 
                 getline(ss, s_parm, ',');
                 theparm = s_parm; 
                               
                 // for "none" we flip the sense of the option and apply "all"
                 if( theparm == "none" ) { b_opt = ! b_theoption; theparm = "all"; }
                 
                 if( theparm == "all" || theparm == "akaze" )   { inbounds=true; detectorAkaze     = b_opt; }
                 if( theparm == "all" || theparm == "orb" )     { inbounds=true; detectorOrb       = b_opt; }
                 if( theparm == "all" || theparm == "brisk" )   { inbounds=true; detectorBrisk     = b_opt; }
                 if( theparm == "all" || theparm == "corner" )  { inbounds=true; detectorCorner    = b_opt; }
                 if( theparm == "all" || theparm == "gftt" )    { inbounds=true; detectorGFTT      = b_opt; }
                 if( theparm == "all" || theparm == "line" )    { inbounds=true; detectorLine      = b_opt; }
                 if( theparm == "all" || theparm == "segment" ) { inbounds=true; detectorLSD       = b_opt; }
                 if( theparm == "all" || theparm == "blob" )    { inbounds=true; detectorBlob      = b_opt; }
                 #if defined(__OPENCV_XFEATURES2D_HPP__)
                 if( theparm == "all" || theparm == "surf" )    { inbounds=true; detectorSurf      = b_opt; }
                 #endif  
                 if( theparm == "all" || theparm == "sift" )    { inbounds=true; detectorSift      = b_opt; }              
                 if( theparm == "all" || theparm == "dsift" )   { inbounds=true; detectorDenseSift = b_opt; } 
                 if( theparm == "all" || theparm == "pto" )     { inbounds=true; detectorPto       = b_opt; }                              
                 if(                     theparm == "masks" )   { inbounds=true; detectorMasks     = b_opt; }              

                 if( inbounds )
                 {
                    if( loglevel > 6 ) sout << "Parsed " << theoption << " " << s_parm << std::endl;
                 }
                 else
                 {
                    sout << "ERROR: option " << theoption << " '" << theparm << "' should be: " << std::endl
                         << "{ all | none | mask | akaze | blob | corner | gftt | line | orb "
                       #if defined(__OPENCV_XFEATURES2D_HPP__)
                         << "| surf"
                       #endif  
                         << " | sift | dsift | ... }" << std::endl
                         << "See --list detectors and or --help for more info on available detectors." << std::endl;
                    return 1;              
                 }

              } // while more values to parse from ss
                 
              // remove the file name
              for(int i=1; i<argc; ++i)
                  argv[i]  = argv[i+1];
              --argc;
           }
           
           parsed = true;  // in this case, just do the next argument
       }

       // is it --filter / --nofilter [ distance | overlap | quadrant | all ]  ?
       if ( ! parsed && j < argc && ( 0 == strcmp(argv[j], "--filter") || 0 == strcmp(argv[j], "--nofilter") ) )
       {
           std::string theoption = argv[j];
           bool b_theoption = true;
           if( 0 == strcmp(argv[j], "--nofilter") ) b_theoption = false;

           std::string theparm;
           int thenum = 0;
           bool inbounds = false;
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           // look for required parameter and ensure it does not begin with a "-" 
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) )
           {
              theparm = argv[j];

              if( theparm == "all" || theparm == "quadrant" )  { inbounds=true; quadrantFilter   = b_theoption; }
              if( theparm == "all" || theparm == "distance" )  { inbounds=true; distanceFilter   = b_theoption; }
              if( theparm == "all" || theparm == "duplicate" ) { inbounds=true; duplicateFilter  = b_theoption; }
              if( theparm == "all" || theparm == "cpoverlap" ) { inbounds=true; cpoverlapFilter  = b_theoption; }
              if( theparm == "all" || theparm == "kpoverlap" ) { inbounds=true; kpoverlapFilter  = b_theoption; }
              if( theparm == "all" || theparm == "original" )  { inbounds=true; originalFilter   = b_theoption; }
              if( inbounds )
              {
                 if( loglevel > 6 ) sout << "Parsed " << theoption << " " << theparm << std::endl;
              }
                 
              // remove the file name
              for(int i=1; i<argc; ++i)
                  argv[i]  = argv[i+1];
              --argc;
           }
           
           if( ! inbounds )
           {
              sout << "ERROR: option " << theoption << " should be one of: " << std::endl
                   << "{ all | quadrant | distance | overlap }" << std::endl;
              return 1;              
           }

           parsed = true;  // in this case, just do the next argument
       }

// = = = = = = = = =

       // is it --savediag / --nosavediag [ masks | trials | pairs | neighbors ]  ?
       if ( ! parsed && j < argc && ( 0 == strcmp(argv[j], "--savediag") || 0 == strcmp(argv[j], "--nosavediag") ) )
       {
           std::string theoption = argv[j];
           bool b_theoption = true;
           if( 0 == strcmp(argv[j], "--nosavediag") ) b_theoption = false;

           std::string theparm;
           int thenum = 0;
           bool inbounds = false;
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           // look for required parameter and ensure it does not begin with a "-" 
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) )
           {
              // parse as a comma separated list, e.g. brisk,orb,none,line,all etc.
              std::stringstream ss(argv[j]);
              while (ss.good())
              {
                 inbounds = false;
                 bool b_opt = b_theoption;
                 std::string s_parm;
                 
                 getline(ss, s_parm, ',');
                 theparm = s_parm; 
                               
                 // for "none" we flip the sense of the option and apply "all"
                 if( theparm == "none" ) { b_opt = ! b_theoption; theparm = "all"; }
                 
                 if( theparm == "all" || theparm == "masks" )      { inbounds=true; saveDiagMasks      = b_opt; }
                 if( theparm == "all" || theparm == "trials" )     { inbounds=true; saveDiagTrials     = b_opt; }
                 if( theparm == "all" || theparm == "pairs" )      { inbounds=true; saveDiagPairs      = b_opt; }
                 if( theparm == "all" || theparm == "aligned" )    { inbounds=true; saveDiagAligned    = b_opt; }
                 if( theparm == "all" || theparm == "neighbors" )  { inbounds=true; saveDiagNeighbors  = b_opt; }
                 if( theparm == "all" || theparm == "decimation" ) { inbounds=true; saveDiagDecimation = b_opt; }


                 if(                     theparm == "good" )      
                    { inbounds=true; saveDiagGood  = b_opt; saveDiagBad  = ! b_opt; }              
                 if(                     theparm == "bad" )       
                    { inbounds=true; saveDiagBad   = b_opt; saveDiagGood = ! b_opt; }              
                 if(                     theparm == "any" )       
                    { inbounds=true; saveDiagBad   = b_opt; saveDiagGood =   b_opt; }              


                 if( inbounds )
                 {
                    if( loglevel > 6 ) sout << "Parsed " << theoption << " " << s_parm << std::endl;
                 }
                 else
                 {
                    sout << "ERROR: option " << theoption << " '" << theparm << "' should be: " << std::endl
                         << "{ all | masks | trials | pairs | aligned | neighbors | decimation }" << std::endl
                         << "See --list detectors and or --help for more info on available detectors." << std::endl;
                    return 1;              
                 }

              } // while more values to parse from ss
                 
              // remove the file name
              for(int i=1; i<argc; ++i)
                  argv[i]  = argv[i+1];
              --argc;
           }
           
           parsed = true;  // in this case, just do the next argument
       }

// = = = = = = = = =


       // is it --pattern [ lr | rl ][ tb | bt ]  ?
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], "--pattern") )
       {
           std::string theoption = argv[j];
           bool b_theoption = true;

           std::string theparm;
           int thenum = 0;
           bool inbounds = false;
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           // look for required parameter and ensure it does not begin with a "-" 
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) )
           {
              theparm = argv[j];

              shootingLR = StringOrder( theparm, "l", "r" );    // if rl +,  or lr - 
              shootingTB = StringOrder( theparm, "t", "b" );    // if tb -,  or bt +
              shootingAxis = StringOrder( theparm, "l", "b" );  // if sequential in rows, i.e. X -, in columns, i.e. Y +

              if( shootingLR != 0 || shootingTB !=0 ) { inbounds=true; }
 
              if( inbounds )
              {
                 if( loglevel > 6 ) sout << "Parsed " << theoption << " " << theparm << " dx=" << shootingLR 
                                  << " dy=" << shootingTB << " axis=" << shootingAxis << std::endl;
              }
                 
              // remove the file name
              for(int i=1; i<argc; ++i)
                  argv[i]  = argv[i+1];
              --argc;
           }
           
           if( ! inbounds )
           {
              sout << "ERROR: option " << theoption << " should include letters: " << std::endl
                   << "{ r,l,t,b } e.g. 'lrbt' or 't-b,r-l' etc. " << std::endl;
              return 1;              
           }

           parsed = true;  // in this case, just do the next argument
       }


       // is it --list arg  ?
       if ( ! parsed && j < argc && 0 == strcmp(argv[j], "--list") )
       {
           std::string theoption = argv[j];
           std::string theparm;
           int thenum = 0;
           bool inbounds = false;
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           // look for required parameter and ensure it does not begin with a "-" 
           if( j < argc && 0 != strncmp(argv[j], "-", (size_t) 1) )
           {
              theparm = argv[j];

              if( theparm == "detectors" )  { inbounds=true; listDetectors = true; }
              if( inbounds )
              {
                 if( loglevel > 6 ) sout << "Parsed " << theoption << " " << theparm << std::endl;
              }
                 
              // remove the file name
              for(int i=1; i<argc; ++i)
                  argv[i]  = argv[i+1];
              --argc;
           }
           
           if( ! inbounds )
           {
              sout << "ERROR: option " << theoption 
                   << " should be { detectors }" << std::endl;
              return 1;              
           }
           
           parsed = true;  // in this case, just do the next argument
       }

       OPTIONBOOL(--verbose,verbose,true);
       OPTIONBOOL(--trace,dotrace,true);
       OPTIONBOOL(--benchmarking,benchmarking,true);
       OPTIONBOOL(--nocpfind,nocpfind,true);
       OPTIONBOOL(--nochippy,nochippy,true);
       OPTIONBOOL(--nochippy,nochippy,true);  // ???
       OPTIONBOOL(--hugin,hugin,true);
       if( ! threaded ) threadsMax = 1;
       OPTIONBOOL(--noquadrantfilter,quadrantFilter,false);
       OPTIONBOOL(--saveallimages,save_all_images,true);
       OPTIONBOOL(--nodistancefilter,distanceFilter,false);
       if( save_all_images ) save_good_images = true;
       OPTIONBOOL(--savegoodimages,save_good_images,true);
       OPTIONBOOL(--dohomography,doHomography,true);
       OPTIONBOOL(--trialhomography,trialHomography,true);
       OPTIONBOOL(--looprepair,loopRepair,true);       
       OPTIONBOOL(--loopfilter,loopFilter,true);
       OPTIONBOOL(--loopanalysis,loopAnalysis,true);       
       OPTIONBOOL(--log,globalLogging,true);
       OPTIONBOOL(--alignlog,alignLogging,true);
       OPTIONBOOL(-h,showhelp,true);
       OPTIONBOOL(-v,showversion,true);
       OPTIONBOOL(--test,dotest,true);              
       OPTIONBOOL(--help,showhelp,true);
       OPTIONBOOL(--huginflipin,huginFlipIn,true);
       OPTIONBOOL(--huginflipout,huginFlipOut,true);
       OPTIONBOOL(--nothreads,threaded,false);

       // is it --  ( i.e. stop parsing options ) 
       if ( ! parsed && j < argc && ( 0 == strcmp(argv[j], "--") || 0 == strcmp(argv[j], "--ignore_rest") ) )
       {
           if( loglevel > 6 ) sout << "Parsed " << argv[j] << " option ( -- )" << std::endl;
           for(int i=1; i<argc; ++i)
               argv[i]  = argv[i+1];
           --argc;
           parsed = true;
           j = argc + 1; // set end condition for command line parsing
       }

       // if an option was not parsed AND there are more items, its likely a bad option.
       if( ! parsed && argc > 2 )
       {
           sout << "ERROR: command line option '" << current_option << "' not valid." << std::endl;
           sout << "       use " << argv[0] << " -h for help." << std::endl;
       }

       if( ! parsed ) j++; // If an option was not found, advance the 
    }
}

       // Display version and other vaguely helpful information
       if( showversion )
       {
        sout << "cvfind 0.0.1 - a Hugin cpfind alternative and panotools project cleanup tool" << std::endl;
        sout << "               (C) 2023 Robert Charles Mahar" << std::endl;
        sout << "               Licensed Under Apache License 2.0" << std::endl;
        sout << "               https://github.com/Bob-O-Rama/cvfind" << std::endl;
        sout << "               OpenCV Version: " << cv::getVersionString() << std::endl;
      #if defined(__OPENCV_XFEATURES2D_HPP__)
        sout << "                  - Built with -DWITH_OPENCV_CONTRIB, SURF available" << std::endl;
      #else
        sout << "                  - Built without -DWITH_OPENCV_CONTRIB, SURF not available" << std::endl;
      #endif
        sout << "               Hard Limits: MAX_IMAGES=" << to_string( MAX_IMAGES ) 
             << " MAX_CPS_PER_PAIR=" << to_string( MAX_CPS_PER_PAIR ) << std::endl;
        sout << "               Memory Usage: ~5.58MB required per megapixel of source images" << std::endl; 
        sout << "               Tested With: Hugin ver. 2020.0.0 generated .pto files" << std::endl;
       }

       // Display full help ...
       if( showhelp )
       {
        sout << "Usage: " << argv[0] << " [-d] [-h] [--option ...] [ -o output.pto] input.pto" << std::endl;
        sout << "  -d                   Enables startrup debugging.  Must be first if used." << std::endl;
        sout << "  -h                   Display (this) help page." << std::endl;
        sout << "  -v                   Display version information." << std::endl;
        sout << "  -o output.pto        Saves modified copy of .pto to specified file" << std::endl;
        sout << "  --benchmarking       Enable reporting of benchmarking info" << std::endl;
        sout << "  --exitdelay N        Sleep for N seconds prior to exiting.  Currently: " << exitDelay << std::endl;        
        sout << "  --hugin              Make cvfind vaguely cpfind compatible by not protecting output .pto files." << std::endl;
        sout << "  --linearmatchlen N   Set maximum image index difference for pairs to N.  Currently: " << linearMatchLen << std::endl;
        sout << "  --rowsize N          Set estimated row size to N.  Currently: " << rowSize << std::endl;
        sout << "  --rowsizetolerance N Effective row size is --rowsize +/- tol.  Currently: " << rowSizeTolerance << std::endl;            
        sout << "  --overlap N          Set estimated overlap percentage.  Currently: " << overlapRatioPct << "%" << std::endl;
        sout << "  --overlaptolerance N Effective overlap is --overlap +/- tol.  Currently: " << overlapMarginPct << "%" << std::endl;            
        sout << "  --noimages range     Exclude pairs with images specified, e.g. all 0-39,44,66,92-"  << std::endl;
        sout << "  --images range       Include pairs with images specified, e.g. all 0- 27-50,60-70,-"  << std::endl;
        sout << "  --centerfilter N     Allow more than N centers ( of two possible ). " << std::endl;            
        sout << "  --peakwidth N        Window width for detecting spurious pairs in adjacent images." << std::endl;            
        sout << "  --nocpfind           Disables loading and examining images ( dry run )" << std::endl;
        sout << "  --nothreads          Disable threading.  Processes each image one at a time" << std::endl;
        sout << "  --ncores -n N        Changes the maximum threads from " << threadsMax << " to N" << std::endl;        
/* ToDo: Deprecated, see --savediag
        sout << "  --saveallimages      Saves all diagnostic images, even for bad pairs" << std::endl;
        sout << "  --savegoodimages     Saves diagnostic images only for good pairs" << std::endl;
*/
        sout << "  --savediag type      Create diagnostic markup images.  Specify multiples types if needed." << std::endl;
        sout << "        where type is: { masks | trials | pairs | aligned | neighbors }  " << std::endl;
         
        sout << "  --dohomography       Computes homography between images in pairs.  ( Default: On )" << std::endl;
        sout << "  --trialhomography    Picks best of several overlaps based on CP distance.  ( Default: On )" << std::endl;
        sout << "  --trialhomsize N     Match distance filter width for homography trials.  Currently: " << trialHomSize << std::endl;
        sout << "  --trialhomstep N     Match distance filter steps of N for homography trials.  Currently: " << trialHomStep << std::endl;
        sout << "  --loopanalysis       Enables loop homography analysis to identify bad pairs." << std::endl;
        sout << "  --loopfilter         ... Plus enables loop homography filter to remove bad pairs. " << std::endl;
        sout << "  --looprepair         ... Plus enabled loop homography repair using trials." << std::endl;

       }
       if( showhelp || listDetectors )
       {
        sout << "  --detect type        Enable a detector.  Specify multiple types if needed." << std::endl;                
        sout << "  --nodetect type      Disable a detector.  Specify multiple type if needed." << std::endl;                
        sout << "        where type is: { akaze | brisk | blob | corner | line | orb "
          #if defined(__OPENCV_XFEATURES2D_HPP__)
             << "| surf"
          #endif  
             << " | sift ... " << std::endl;
        sout << "                       ... segment | rsift | mask | all }.  'all' can be used to disable all." << std::endl;
       }
       if( listDetectors )
       {
        sout << "        detector type: all     - Generally used with --nodetect to clear defaults." << std::endl;
        sout << "                       mask    - Selected detectors mask off non-overlapping central " 
                 << max( 100, 100 - ( 2 * ( overlapRatioPct + overlapMarginPct ) ) ) << "% of image." << std::endl;
        sout << std::endl;
        sout << "                       orb     - best mix of performance and robustness. ( Default )" << std::endl;
        sout << "                                 generally works without futzing." << std::endl;
        sout << "                       sift    - slower, slightly more robust than ORB.  Generally finds" << std::endl;
        sout << "  (generally usable)             more usable features.  Old reliable." << std::endl;
        sout << "                       akaze   - slower, implements a auto tuning to return ~20K key points." << std::endl;
        sout << "                       line    - detects line via binary descriptor, matched keylines." << std::endl;
      #if defined(__OPENCV_XFEATURES2D_HPP__)
        sout << "                       surf    - slower, better for fine patterns?" << std::endl;
      #endif  
        sout << "                       brisk   - slow, generally not as robust as ORB." << std::endl;
        sout << "                       gftt    - Good Features To Track, good, I guess." << std::endl; 
        sout << std::endl;
        sout << "                       dsift   - dense SIFT, force feature detection in a sliding" << std::endl;
        sout << "                                 window.  Useful in traces usually ignored by above. " << std::endl;
        sout << "    (experimental)     blob    - detects blobs, then extracts SIFT features." << std::endl;
        sout << "                       segment - detects line segment via LSD, matches keylines." << std::endl;
        sout << "                       corner  - detects corners." << std::endl;
       }    
       if( showhelp )
       {
        sout << "  --list detectors     Provides additional info for each detector." << std::endl;                                 
        sout << "  --cellsize N         Specified the cell size for decimation.  Currently: " << cellSize << "px" << std::endl;
        sout << "  --cellmincp N        Minimum control points per cell.  Currently: " << cellMinCP << std::endl;        
        sout << "  --cellmaxcp N        Maximum control points per cell.  Currently: " << cellMaxCP << std::endl;
        sout << "  --cellcpdupdist N    Minimum distance CPs in a cell.  Currently: " << cellCPDupDist << std::endl;
        sout << "  --cpperpairmin N     Minimum control points per image pair.  Currently: " << cpPerPairMin << std::endl;        
        sout << "  --cpperpairmax N     Maximum control points per image pair.  Currently: " << cpPerPairMax << std::endl;                
        sout << "  --neighborsnin N     Minimum pairs an image participate in.  Currently: " << neighborsMin << std::endl;        
        sout << "  --neighborsmax N     Maximum pairs an image participate in.  Currently: " << neighborsMax << std::endl;                

        sout << "  --benchmarking       Enable reporting of benchmarking info" << std::endl;
        sout << "  --exitdelay N        Sleep for N seconds prior to exiting.  Currently: " << exitDelay << std::endl;        
        sout << "  --prescale N         (Broken) Downsamples images loaded to 1/N scale.  Currently: " << cellSize << "px" << std::endl;
        sout << "  --mindist  N         minimum RANSAC error distance used for filtering.  Currently: " << minDist << std::endl;
        sout << "  --maxdist  N         maximum RANSAC error distance used for filtering.  Currently: " << maxDist << std::endl;
        sout << "  --prescale N         (Broken) Downsamples images loaded to 1/N scale.  Currently: " << cellSize << "px" << std::endl;
        sout << "  --huginflipin        When reading CPs from a .pto, mirrors their x and y coordinates" << std::endl;
        sout << "  --huginflipout       When adding new CPs to a .pto, mirror their x and y coordinates" << std::endl;
        sout << "  --debug              Enable all debug messages, same as --loglevel 7" << std::endl;
        sout << "  --verbose            Enable some debug messages, same as --loglevel 4" << std::endl;
        sout << "  --loglevel N         0-2: Terse; 3-5: Verbose; 6-8: Debug; 9+: Trace.  Currently: " << loglevel << std::endl;
        sout << "  --log                Logs messages to a file, a mess without --nothreads" << std::endl;
        sout << "  --test               Performs a self test, exits with error level set." << std::endl;
        sout << "  --alignlog           Keeps a separate log for each image pair" << std::endl;
        sout << "  -- --ignore_rest     ignores all further options" << std::endl;        
       }
 
       if( showhelp || listDetectors || showversion ) return 0;
       
       if( dotest )
       {  
          if( testSuccessful() )
          {
             sout << " INFO: Self Test Passed, returning error level 0." << std::endl;
             return 0;
          }
          else
          {
             sout << " INFO: Self Test Failed, returning error level 1." << std::endl;
             return 1;
          }
       }

  //
  // Fixup controls and settings after all of the user specified options have been applied
  //

  // update loglevel from legacy log levels
  if( verbose ) loglevel = 4;
  if( debug )   loglevel = 7;

  // if linear match length not set by user but rowSize is, set linear match length to 2 * row size
  if( linearMatchLen == 100000 && rowSize > 0 ) linearMatchLen = rowSize * 2;

  // --looprepair implies --loopfilter
  if( loopRepair ) loopFilter = true;

  // --loopfilter implies --trialhomography
  if( loopFilter ) loopAnalysis = true;

  // --loopanalysis implies --trialhomography
  if( loopAnalysis ) trialHomography = true;  

  // --trialhomography implies --dohomography
  if( trialHomography ) doHomography = true;

  // update number of available threads
  if( threaded )
  {
     if( threadsMax == 1 )
     {  
        threadsMax = std::thread::hardware_concurrency();
     }
     if( loglevel > 3 ) sout << " INFO: " << argv[0] << " will use " << threadsMax << " threads for image processing." << std::endl;
  }      

  // Fixup overlap values if user cannot do math
  overlapRatio = (float)( (float)overlapRatioPct / (float)100 );
  overlapMargin = (float)( (float)overlapMarginPct / (float)100 );
  
  // Fixup minimum distance between control points if one was not specified.
  if( cellCPDupDist == 0 ) cellCPDupDist = sqrt( ( cellSize * cellSize ) / cpPerPairMax);

  // If logging to a file, open it.
  if( globalLogging && ! sout.coss.is_open() )
  {
     sout.coss.open("cvfind_" + GetDateTime() + ".log", std::ios::out | std::ios::app );
     sout.coss << "##### CVFIND GLOBAL LOG STARTS " << GetDateTime() << " FOR RUN ID: " << runID << " ######" << std::endl;
  }

// display any effective arguments ( excluding popped ones )
if( loglevel > 6 )
{
  sout << " INFO: Remaining Aguments:" << std::endl;
  // Print arguments because I have not coded in 20 years 	
  int i = 0;
  while ( i < argc )
  {
     sout << "[ " << i << " ]" << argv[i] << std::endl;
     i++;
  }
  sout << std::endl;  
}

  
  // At this point argv[0] is the original argv[0] e.g. ./cvfind
  // and argv[1] should be the last and final argument e.g. the input.pto file name

  // Check argument count
  if( argc < 2 )
    {
        sout << "After parsing options, no input.pto file name specified." << std::endl << std::endl;
    }
    else
    {
       inputPTO = argv[1];
    }
       
  // Check argument count
  if( argc > 2 )
    {
        sout << "After parsing options, there were extraneous arguments sprcified." << std::endl << std::endl;
        // display any effective arguments ( excluding popped ones )
        sout << "Extraneous arguments:" << std::endl;
        // Print arguments because I have not coded in 20 years 	
        int i = 2;
        while ( i < argc )
        {
           sout << "[ " << i << " ]" << argv[i] << std::endl;
           i++;
        }
        sout << std::endl;  
    }  
      
  // Check argument count
  if( argc != 2 )
    {
        sout << "Usage: " << argv[0] << " [-d] [-h] [--option ...] [ -o output.pto] input.pto" << std::endl;
        sout << "  -h                   Display full Help." << std::endl;
        sout << "  -o output.pto        Saves modified copy of .pto to specified file" << std::endl;       
        return 1;
    }

        std::string enabled  = "";
        std::string disabled = "";
        detectorEnabledCnt   = 0;
        detectorDisabledCnt  = 0;

       
        if( detectorAkaze    ) { enabled  = enabled  + " AKAZE";     detectorEnabledCnt++; }
        if( detectorOrb      ) { enabled  = enabled  + " ORB";       detectorEnabledCnt++; }
        if( detectorBrisk    ) { enabled  = enabled  + " BRISK";     detectorEnabledCnt++; }
        if( detectorCorner   ) { enabled  = enabled  + " CORNER";    detectorEnabledCnt++; }
        if( detectorGFTT     ) { enabled  = enabled  + " GFTT";      detectorEnabledCnt++; }
        if( detectorLine     ) { enabled  = enabled  + " LINE";      detectorEnabledCnt++; }
        if( detectorLSD      ) { enabled  = enabled  + " SEGMENT";   detectorEnabledCnt++; }
        if( detectorBlob     ) { enabled  = enabled  + " BLOB";      detectorEnabledCnt++; }
        if( detectorSurf     ) { enabled  = enabled  + " SURF";      detectorEnabledCnt++; }
        if( detectorSift     ) { enabled  = enabled  + " SIFT";      detectorEnabledCnt++; }
        if( detectorDenseSift    ) { enabled  = enabled  + " DSIFT";     detectorEnabledCnt++; }
        if( detectorPto      ) { enabled  = enabled  + " PTO";       detectorEnabledCnt++; }
        if( detectorSurf     ) { enabled  = enabled  + " -w- MASK"; }

        if( ! detectorAkaze  ) { disabled = disabled + " AKAZE";     detectorDisabledCnt++; }
        if( ! detectorOrb    ) { disabled = disabled + " ORB";       detectorDisabledCnt++; }
        if( ! detectorBrisk  ) { disabled = disabled + " BRISK";     detectorDisabledCnt++; }
        if( ! detectorCorner ) { disabled = disabled + " CORNER";    detectorDisabledCnt++; }
        if( ! detectorGFTT )   { disabled = disabled + " GFTT";      detectorDisabledCnt++; }
        if( ! detectorLine   ) { disabled = disabled + " LINE";      detectorDisabledCnt++; }
        if( ! detectorLSD    ) { disabled = disabled + " SEGMENT";   detectorDisabledCnt++; }
        if( ! detectorBlob   ) { disabled = disabled + " BLOB";      detectorDisabledCnt++; }
        if( ! detectorSurf   ) { disabled = disabled + " SURF";      detectorDisabledCnt++; }
        if( ! detectorSift   ) { disabled = disabled + " SIFT";      detectorDisabledCnt++; }
        if( ! detectorDenseSift  ) { disabled = disabled + " DSIFT";     detectorDisabledCnt++; }
        if( ! detectorPto    ) { disabled = disabled + " PTO";       detectorDisabledCnt++; }
        if( ! detectorSurf   ) { disabled = disabled + " MASK"; }

        if( enabled.size() > 0 )
        {
           std::replace(enabled.begin(), enabled.end(), ' ', ',');
           enabled.erase(0, 1);
        } 

        if( disabled.size() > 0 )
        {
           std::replace(disabled.begin(), disabled.end(), ' ', ',');
           disabled.erase(0, 1);
        } 

        if( loglevel > 3 ) sout << " INFO: Feature Detectors {" << enabled 
                           << "} Enabled and {" << disabled << "} Disabled." << std::endl;
   
        if( detectorEnabledCnt == 0 ) sout << "ERROR: No detectors selected.  Examine --nodetect and --detect options." << std::endl;
        
  // Can we open the input PTO file?  No?  Then abort. 
   std::ifstream file(inputPTO);
   if(! file)
    {  
	sout << "ERROR: Could not open input PTO file " << inputPTO << std::endl;    
        return 1;
    } 

    inputPTOPath = std::filesystem::path( inputPTO );
    
    if( inputPTOPath.has_relative_path() )
       sout << " INFO: " << inputPTO << " relative_path(): " << inputPTOPath.relative_path() << std::endl;
    if( inputPTOPath.has_root_path() )
       sout << " INFO: " << inputPTO << " root_path(): " << inputPTOPath.root_path() << std::endl;
    if( inputPTOPath.has_parent_path() )
       {
          inputPTOImagePrefix = inputPTOPath.parent_path();
          sout << " INFO: Using '" << inputPTOImagePrefix << "' to prefix relative image file names." << std::endl;
       }


  // Can we open the output PTO file? Yes? Then abort.  ( unless --hugin is specified )
   if( ! hugin && outputPTO.size() > 0 )
   {  
       std::ifstream outfile( outputPTO );
      if(outfile)
       {  
	   sout << "ERROR: Output PTO file " << outputPTO << " already exists.  Use a different output file. " << std::endl;    
           return 1;
       }
   }


   sout << std::endl << " INFO: Chippy - Tile PTO Cleanup for Hugin Panoramas" << std::endl << std::endl   
                     << " INFO: Processing: " << inputPTO << " project file." << std::endl << std::endl;    

  std::string str;
  int line = 0;
  int parsed = 0;
  
  //
  // - - - - - - - - - - - - - - - - - INITIALIZE - - - - - - - - - - - - -
  //

#if defined(__linux__)
  // Non portable use of prctl()
  const std::string tn = "cvfind main()";
  prctl(PR_SET_NAME, tn.c_str(), 0, 0, 0);
#endif
  
  //
  // - - - - - - - - - - - - - - - - - - READ - - - - - - - - - - - - - - -
  //
  line = 0;  
  while (std::getline(file, str))
  {
    line++;
    lines.push_back( str );
  } 

  sout << " INFO: Read " << lines.size() << " lines.  ( " << parsed << " parsed. )" << std::endl; 

  //
  // - - - - - - - - - - - - - - - - - PARSE IMAGES - - - - - - - - - - - -
  //
  /* Example lines we need to parse
  
     i w4800 h3200 f0 v=0 Ra=0 Rb=0 Rc=0 Rd=0 Re=0 Eev4.93663791750474 Er1 Eb1 r0 p-2196.08736648679 y0 T
     rX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a=0 b=0 c=0 d=0 e=0 g=0 t=0 Va=0 Vb=0 Vc=0 Vd=0 Vx=0 Vy=0  Vm5 n"DMap/R03
     C10_R03C10_IMGP6975_IMGP6985_ZS_DMap.tif"
  
     c n0 N1 x178.566137505324 y1173.92032073442 X3561.05125004512 Y1190.1670213889 t0
   */

  line = 0;
  while( line < lines.size() )
  {
   std::string str = lines[line];
   // If the line is not empty, then check the prefix and parse.
   // "i" = image, "c" = control point
   if( str.length() > 0 )
   {
      //
      // parse Image entry
      //
      if( str[0] == 'i' )
      {
	 parsed++;

if( loglevel > 8 )
{     
         sout << "TRACE: w=" << findToken( str, " w" )
              << " h=" << findToken( str, " h" ) 
              << " n=" << findToken( str, " n" ) << std::endl;
}
	 ptoImage img;

         img.mutex = &imageMutex[images.size()];
         img.w = stoi( findToken( str, " w" ) );
	 img.h = stoi( findToken( str, " h" ) );
         img.filename = findToken( str, " n" );
         img.groupRoot = images.size(); // set group root to image index by default.

         // If the image filename from the PTO file is relative, prefix it with the path of the PTO file.
         // ToDo: need a platform independant way to so this.  For the moment assuming Linux.
                   
         if( img.filename.size() && img.filename[0] != '/' )
         {
            img.path = std::filesystem::path( img.filename );
            if( ! img.path.has_root_path() )
            {
               std::string newfilename = inputPTOImagePrefix + "/" + img.filename;
               if( loglevel > 3 ) sout << " INFO: Relative image filename in PTO";
               if( loglevel > 6 )   sout << " '" << img.filename << "'" ;
               if( loglevel > 3 ) sout << " changed to '" << newfilename << "'" << std::endl; 
               img.filename = newfilename;
            }
         }

         images.push_back( img );

         ptoImage foo;

	 foo = images.back();
if( loglevel > 6 )
{
	 sout << "IMAGE: " << images.size() - 1 << " w=" << foo.w << " h=" << foo.w << " name=" << foo.filename << std::endl;
}
      } // 'i'

   }  // If length > 0

  line++;
  }  // While 

  //
  // Give user some statistics and happy feedback
  //
  
  sout << " INFO: " << images.size() << " images referenced in .pto file." << std::endl;

  // At this point we ahve enough info to examine images if the operator wants.
  // Control points are extracted from the input PTO file after that
  // so that new found control points can be tipped into the PTO for
  // eventual output
  
  // if linearMatchLen is more than that prescribed by the number of images, reduce it.
  if( linearMatchLen > images.size() ) linearMatchLen = images.size() - 1;

  //
  // - - - - - - - CREATE PAIR DATA STRUCTURES DYNAMICALLY - - - - - - - - - -
  //
  
  // Initialize pairs table
  // ToDo: This is hugely wasteful for large number of images or where actual number of pairs is limited
  //       the whole pairs[][] should be reworked to be sparse.  But its used everywhere so potentially hard to do.
  //       displays some dots to the user can tell the app is not hung.

  if( true )
  {
     std::vector<ptoImgPair> theRow;
     ptoImgPair thePair;
     sout << " INFO: Creating structures for " << images.size() * images.size() << " pairs ";
     for( int i=0; i < images.size(); i++ ) theRow.push_back( thePair );
     int spinner = 0;
     for( int j=0; j < images.size(); j++ )
     {   
        pairs.push_back( theRow );
        // Initialize the image index values for valid pairs ( where idx1 < idx2 ) 
        for( int k=j+1; k < pairs.back().size(); k++ )
        {
           pairs[j][k].idx1=j;
           pairs[j][k].idx2=k;
        }           
        spinner++;
        if( spinner > images.size() / 50 )
        {
           spinner = 0;
           sout << ".";
        }
     }   
     sout << " done." << std::endl;
  }

  //
  // - - - - - - - BUILD PAIR LIST FOR FEATURE DETECTION ( CVFIND ) - - - - - - 
  //

  sout << " INFO: Find candidate image pairs for " << images.size() << " images. rowSize= " << rowSize << " +/- " << rowSizeTolerance 
       << " linearMatchLen= " << linearMatchLen << std::endl;
  if( mandatoryImages.size() != 0 ) sout << " INFO: Mandatory Images: " << vi2csv( mandatoryImages ) << std::endl;
  if( forbiddenImages.size() != 0 ) sout << " INFO: Forbidden Images: " << vi2csv( forbiddenImages ) << std::endl;
  if( mandatoryPairs.size() != 0 )  sout << " INFO: Mandatory Pairs: " << mandatoryPairs << std::endl;
  if( forbiddenPairs.size() != 0 )  sout << " INFO: Forbidden Pairs: " << forbiddenPairs << std::endl;
  
  // For each interval between images
  for( int i = 1; i < images.size(); i++ )
     // for each image pair ( j, j+i )
     for( int j = 0; i + j < images.size(); j++ )
     {     
        bool requested = false;
        bool included = false;
        bool excluded = false;
        
        // check rowsize and linear matching
        if(    (     rowSize == 0 
                 || i < 2 
                 || ( i >= ( rowSize - rowSizeTolerance ) && i <= ( rowSize + rowSizeTolerance ) ) )
               
            &&  i < linearMatchLen )
           requested = true;   
        
        // ToDo:  Learn boolean algebra and then fix this???
        
        if( false )
        {
            sout << " INFO: Pair forbidden / mandatory search results: "
                 << j     << ": " << binary_search(forbiddenImages.begin(), forbiddenImages.end(), j )     << " "
                 << j + i << ": " << binary_search(forbiddenImages.begin(), forbiddenImages.end(), j + i ) << " "        
                 << j     << ": " << binary_search(mandatoryImages.begin(), mandatoryImages.end(), j )     << " "
                 << j + i << ": " << binary_search(mandatoryImages.begin(), mandatoryImages.end(), j + i ) << std::endl;
        }

        // If image pair includes a forbidden image, the pair is excluded.  So "2,4,7" reject all pairs referencing all 3 images
        if( forbiddenImages.size() > 0 )
        { 
           if( binary_search(forbiddenImages.begin(), forbiddenImages.end(), j ) )     excluded = true;        
           if( binary_search(forbiddenImages.begin(), forbiddenImages.end(), j + i ) ) excluded = true;        
        }

        // but if both images are mandatory, include the pair.  So "{2,4,7}" results in (2,4) (2,7) and (4,7)  
        if(    mandatoryImages.size() > 0
            && binary_search(mandatoryImages.begin(), mandatoryImages.end(), j ) 
            && binary_search(mandatoryImages.begin(), mandatoryImages.end(), j + i ) ) included = true;
                
              
          // if ( not excluded AND requested ) OR mandatory
          if( ( ! excluded && requested ) || included )
          {  
             // Add pair to the list of candidate pairs
             imageIndexPair p;
             p.idx1 = j;
             p.idx2 = j + i;
             candidatePairs.push_back( p );

             // Update candidate lists on both images.
             images[p.idx1].candidates.push_back( p.idx2 );
             images[p.idx2].candidates.push_back( p.idx1 );

             // flag the images as needing to be loaded for subsequent image processing
             images[p.idx1].needed = true;
             images[p.idx2].needed = true;
             // if( loglevel > 6 && p.idx1 == 0) sout << " INFO: Image Pair {" << p.idx1 << ", " << p.idx2 << "}"; 
             // if( loglevel > 6 && p.idx2 == rowSize ) sout << " <-- center of window ";
             // if( loglevel > 6 && p.idx2 == linearMatchLen ) sout << " <-- end of match length ";
             // if( loglevel > 6 ) sout << std::endl;
          }
     }  

  if( loglevel > 3 ) sout << " INFO: Added " << candidatePairs.size() << " candidate pairs for image processing. " << std::endl;
  if( loglevel > 6 ) sout << " INFO: Image Pairs Queued For Image Processing: " << candidatePairs << std::endl;

  //
  // - - - - - - - - - - - - - - - - - - LOAD - - - - - - - - - - - - -
  //

  int num_needed = 0;
  
  for( int i = 0; i < images.size(); i++ ) if( images[i].needed ) num_needed++; 
  if( ! nocpfind ) sout << std::endl << "CVFind: Loading " << num_needed
                        << " of " << images.size() << " images using " << threadsMax << " threads." << std::endl;
    
  auto start_imgload = std::chrono::high_resolution_clock::now(); 

  int i = 0;
  while( ! nocpfind && i < images.size() )
  {
     int tc;
     
     std::vector<std::shared_future<bool>> futures;
     tc = 0;
     while( i < images.size() && tc < threadsMax )
     {  
    
       futures.push_back( std::async( std::launch::async, 
            [ &img = images[i], i = i, tc = tc]()
            {

              // - - - - multithreaded code block 
              
              #if defined(__linux__)
                // Non portable use of prctl()
                std::string tn = "cvfind img " + left0padint(i, 4);
                prctl(PR_SET_NAME, tn.c_str(), 0, 0, 0);
              #endif

              std::ifstream file1( img.filename );
              if(! file1)
              {  
	         sout << "Could not open " << img.filename << std::endl;    
              }
              else
              {
                 if( img.needed )
                 {
                     if( loglevel > 0 ) sout << "[img=" << left0padint(i, 4) << "][tc=" << left0padint(tc, 2) 
                          <<"] Reading image " << left0padint(i, 4) << " : " << img.filename << endl; 
                     img.load(); // img.img = imread(img.filename);
                 }
                 else
                 {
                    if( loglevel > 6 ) sout << "[img=" << left0padint(i, 4) << "][tc=" << left0padint(tc, 2) 
                          <<"] Skipping unneeded image " << left0padint(i, 4) << " : " << img.filename << endl;
                     // We could do other stuff here for skipped images.      
                 }   
              }
              return true;
           }
              // - - - - - 
       ));
      
     tc++; // Batches of 5   
     i++;
     }

     // We wait for all dispatched jobs in this batch to complete
     int n = 0;
     while( n < futures.size() )
     {      
        bool foo = futures[n].get();  // wait till done.
        n++;
     }
     
     futures.clear(); // We still have open threads in top so IDK???

  }  // for all images

  //
  // - - - - - - - - - - - - - - - - - CHECK IMAGES - - - - - - - - - - -
  // when --nocpfind no images are read.  so we skip this check and rely on the h and w provided in the .pto
  //

  if ( ! nocpfind && loglevel > 3 ) sout << " INFO: Checking Images ..." << std::endl;
  
  bool images_ok = true;
  int rows0, cols0;
  rows0 = cols0 = 0;
  // Find first "needed" image whioch should have been loaded.
  for( int i = 0; i < images.size(); i++ )
  {
     // capture the actual size of the first image, all the rest should match
     if( images[i].needed && ! images[i].img.empty() )
     {
        cols0 = images[i].img.cols; rows0 = images[i].img.rows;
        sout << " INFO: Image " << i << " used for size reference: " << cols0 << " x " << rows0 << std::endl;
        break;
     }
  }
  
  if( cols0 + rows0 == 0 )
     if( images.size() > 0 )
        sout << "ERROR: No image was loaded, assuming " << images[0].w << " x " << images[0].h << " from PTO." << std::endl;
     else
        sout << "ERROR: No images! images.size()=" << images.size() << " How are there no images?" << std::endl;
        
  for( int i = 0; ! nocpfind && i < images.size(); i++ )
  {
     // check loaded image against first loaded image
     if( images[i].needed && ( images[i].img.cols != cols0 || images[i].img.rows != rows0 ) )
     { 
        // This may well be a fatal error as we don't expect it.
        images_ok = false;
        sout << "ERROR: Image [" << i << "] Size Mismatch.  Image 0: " << cols0 << " rows0 " << images[i].h << "  !=  "
             << " This IMG: " << images[i].img.cols << " x " << images[i].img.rows << std::endl;
     }
     
     if( images[i].needed && ( images[i].w != images[i].img.cols || images[i].h != images[i].img.rows ) )
     { 
        // This likely won't be a fata error, but may result in an unusable output PTO or wierdness
        images_ok = false;
        sout << " WARN: Image [" << i << "] Size Mismatch.  PTO: " << images[i].w << " x " << images[i].h << "  !=  "
             << " IMG: " << images[i].img.cols << " x " << images[i].img.rows << std::endl;
     }

     if ( loglevel > 6 && images[i].needed )
        if( images[i].w == images[i].img.cols || images[i].h == images[i].img.rows ) 
           sout << " INFO: Image [" << i << "] Size Matches.  PTO: " << images[i].w << " x " << images[i].h << "  ==  "
                << " IMG: " << images[i].img.cols << " x " << images[i].img.rows << std::endl;           
  }
   
  if ( ! nocpfind && loglevel > 3 && images_ok ) sout << " INFO: All Needed Images Seem Valid" << std::endl;
  if ( nocpfind  ) sout << " INFO: Image check skipped, trusting .PTO values.  " << std::endl;



// = = = = = = = = = = Pre Read PTO CPs

  /* Example lines we need to parse
  
     i w4800 h3200 f0 v=0 Ra=0 Rb=0 Rc=0 Rd=0 Re=0 Eev4.93663791750474 Er1 Eb1 r0 p-2196.08736648679 y0 T
     rX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a=0 b=0 c=0 d=0 e=0 g=0 t=0 Va=0 Vb=0 Vc=0 Vd=0 Vx=0 Vy=0  Vm5 n"DMap/R03
     C10_R03C10_IMGP6975_IMGP6985_ZS_DMap.tif"
  
     c n0 N1 x178.566137505324 y1173.92032073442 X3561.05125004512 Y1190.1670213889 t0

     New control points added via Homography ( prior to this ) are inserted into the live .pto image and then parsed
     along with the control points from the PTO file.   We remember the line number of the first PTO provided 
     control point.  Any of the Homography generated points are already in their ptoImagePair cps[detectorType][pad]
     so when parsing, the cps[detectorType::PTO][pad] is updated, and used in pace of locals.
   */

  //
  // - - - - - - - - - - - - - - PARSE CONTROL POINTS ( also priming PTO "Detector" ) - - - - - - - - - - - -
  //
   int lastCPline = lines.size() - 1;
   if( true )
   {
    
      sout << " INFO: Parsing Hugin PTO Control points Pass 1 ... " << std::endl; 
      if( detectorPto ) sout << " INFO: Contol Points will be added to detectorType::PTO match pad..." << std::endl; 

      if( detectorPto && ptoRewriteCPs )
         sout << " INFO: Rewrite Enabled: Contol Points will be re-written, originals commented out..." << std::endl; 

      if( ( ! detectorPto ) && ptoRewriteCPs )
         sout << " INFO: Rewrite Disabled: Contol Points from PTO are disabled, originals commented out..." << std::endl; 

      if( originalFilter )
         sout << " INFO: Filtering original Contol Points from PTO, originals commented out..." << std::endl; 


      bool insertednewCPlines = false;
      for( int line = 0; line < lines.size(); line++ )
      {

         std::string str = lines[line];
         // If the line is not empty, then check the prefix and parse.
         // "i" = image, "c" = control point
         if( str.length() > 0 )
         {

            // If the Hugin comment is present, set the "lastCPline" line to it
            // so that if there are no control points, we put outs AFTER the comment. 
            if( str == "# control points" )
               lastCPline = line;
               
            //
            // Parse Control Point entry
            //

            if( str[0] == 'c' )
            {  
	       parsed++;
               lastCPline = line;
	 
               if( loglevel > 8 )
               {	     
                   sout << "TRACE: n=" << findToken( str, " n" )
                        << " N=" << findToken( str, " N" )
                        << " x=" << findToken( str, " x" )
                        << " y=" << findToken( str, " y" )              
                        << " X=" << findToken( str, " X" )
                        << " Y=" << findToken( str, " Y" )
                        << " t=" << findToken( str, " t" ) << std::endl;
               }

               ptoControlPoint cpe;

               cpe.idx1 = stoi( findToken( str, " n" ) );
               cpe.idx2 = stoi( findToken( str, " N" ) );
               cpe.src.x = stof( findToken( str, " x" ) );
               cpe.src.y = stof( findToken( str, " y" ) );
               cpe.dst.x = stof( findToken( str, " X" ) );
               cpe.dst.y = stof( findToken( str, " Y" ) );
               cpe.line = line; //
               cpe.detector = detectorType::PTO; 

               int xSize = images[cpe.idx1].w;
               int ySize = images[cpe.idx1].h;
         
               // if flipping then we flip
               if( huginFlipIn )
               {
                  cpe.src.x = xSize - cpe.src.x;   
                  cpe.src.y = ySize - cpe.src.y;   
                  cpe.dst.x = xSize - cpe.dst.x;   
                  cpe.dst.y = ySize - cpe.dst.y;
               } 

               // if flipping, rewrite the line in the live config too.
               // ToDo: This probabaly should not be happening during the PTO Detector parsing
               if( huginFlipIn )
               {
                  std::string s =   "c n" + to_string(cpe.idx1) + " N" + to_string(cpe.idx2)
                                   + " x" + std::to_string( cpe.src.x )   
                                   + " y" + std::to_string( cpe.src.y )   
                                   + " X" + std::to_string( cpe.dst.x )   
                                   + " Y" + std::to_string( cpe.dst.y )
                                   + " t0" ;  // no idea what t0 does
                  lines[line]= s;
               }

               // Add the control point to the pair's detectorType::PTO match pad

               if( detectorPto )
               {               
                  int pad   = pairs[cpe.idx1][cpe.idx2].currentMatchPad(detectorType::PTO);
                  pairs[cpe.idx1][cpe.idx2].cps[detectorType::PTO][pad].push_back(cpe);

                  if( loglevel > 6 )
                  {
                      sout << " INFO: detectorType::PTO: CP: " << cpe.idx1 << " <--> " << cpe.idx2
	                   << "(" << cpe.src.x << "," << cpe.src.y << ") <--> ("
	                   << cpe.dst.x << "," << cpe.dst.y << ") ";
	                   
                      if( ptoRewriteCPs ) sout << " ( For ReWrite )";
                      
	              sout << std::endl;
                  }
                  
               } // If detectorPto == true
               
               // if cvfind is to filter the CPs in the PTO, comment out the original
               if( ptoRewriteCPs )
               {
                   std::string s = lines[line];
                   lines[line] = "# CVFIND # NO REWRITE # " + s;
               }

               // if cvfind is to filter the CPs in the PTO, comment out the original
               if( ptoRewriteCPs )
               {
                   std::string s = lines[line];
                   lines[line] = "# CVFIND # FILTER ORIGINAL # " + s;
               }

               
           } // 'c'

        }  // If length > 0
  
      }  // For each line 

      //
      // Give user some statistics and happy feedback
      //

      if( loglevel > 6 )
      {
         sout << " INFO: Match Pad Dump ( From PTO CPs )... " << std::endl;
         
         for( int d = 0; d < detectorType::ALL; d++ )  
            for( int i = 0; i + 1 < images.size(); i++ )
               for( int j = i+1; j < images.size(); j++ )
               {
                  if( pairs[i][j].cps[d].size() > 0 )
                  {
                     int pad  = pairs[i][j].currentMatchPad(d);
                     if( pairs[i][j].cps[d][pad].size() > 0 )
                     {
                        sout << " INFO: pairs[" << i << "][" << j << "].cps[" << detectorName[d] << "][pad=" << pad 
                             << "].size() = " << pairs[i][j].cps[d][pad].size() << std::endl;                   
                     }
                  }
                  else
                  {
                     // report when there is no current match pad 
                     if( loglevel > 6 )
                     {
                        sout << " INFO: pairs[" << i << "][" << j << "].cps[" << detectorName[d] 
                             << "].size() == 0" << std::endl;                   
                     }  
                  }
               } // for j ... for i ... for d
      } // if Dump Match Pads...

   } // detectorPto == true



  //
  // - - - - - - - - - - - - - - - - - - CPFIND - - - - - - - - - - - - -
  //

  if ( ! nocpfind ) sout << std::endl << "CVFind: Finding Control Points for " << images.size() << " images"
                    << " up to " << linearMatchLen << " apart" 
                    << " using " << threadsMax << " threads." << std::endl;

  time_t start_secs;
  time_t end_secs;
  start_secs = time(NULL);
  
  int job = 0;
  while( ! nocpfind && job < candidatePairs.size() )
  {
        int tc;
     
        std::vector<std::shared_future<bool>> futures;
        tc = 0;
        while( job < candidatePairs.size() && tc < threadsMax )
        {  
           int idx1 = candidatePairs[job].idx1;
           int idx2 = candidatePairs[job].idx2;
           
           futures.push_back( std::async(std::launch::async, 
              [ idx1 = idx1, idx2 = idx2, job = job, tc = tc]() {
              // multi-streaded Lambda function - - - - >

              #if defined(__linux__)
                // Non portable use of prctl()
                std::string tn = "cvfind " + left0padint(idx1, 3) + ":" + left0padint(idx2, 3);
                prctl(PR_SET_NAME, tn.c_str(), 0, 0, 0);
              #endif
                
              if( loglevel > 6 ) sout << " INFO: Job: " << left0padint(job, 6)
                               << " Image Pair " << left0padint(idx1, 4) << " ( " << images[idx1].filename << " ) --> "
                        << left0padint(idx2, 4) << " ( " << images[idx2].filename << " )" << std::endl;
          
              std::string prefix;
              prefix = prefix + "cvfind_" + left0padint(idx1, 4) + "_" + left0padint(idx2, 4);          
              if( ! nocpfind ) alignImages( idx1, idx2, prefix );
              // < - - - - - ends 
              return true;
              }
            ));  // futures.push_back( ... )
      
           tc++;   
           job++;
        }

     // each call to .get() blocks until that worker thread completes, so we are doing "batches" of jobs
     int n = 0;
     while( n < futures.size() )
     {      
        bool foo = futures[n].get();  // wait till done.
        n++;
     }
     
     futures.clear();

     end_secs = time(NULL); // time "now"
     int elapsed_secs = 1 + ( end_secs - start_secs );  // prevent divide by zero
     int rem_secs = ( ( candidatePairs.size() - job ) * elapsed_secs ) / job;

     if( loglevel > 3 ) sout << " INFO: Progress: " << ( job * 100 ) / candidatePairs.size() 
                             << "% ( " << job << " of " << candidatePairs.size() 
                             << " Complete.  " << ( 60 * job ) / elapsed_secs 
                             << " jobs / minute. " << rem_secs << " seconds estimated remaining" << std::endl;  
         
  }  // While ... more jobs
 



  //
  // - - - - - - - - - - - - - - - - - PERFORM LOOP HOMOGRAPHY & REPAIR - - - - - - - - - - - -
  //
  
  if( loopRepair )
  {
     sout << " INFO: Loop Homography Analysis & Repair" << std::endl;
     sout << loopAnalyze( true ); // analyze loop homography ( mark good / bad pairs ) and repair bad ones
  }
  
  if( loopFilter )
  {
     sout << " INFO: Loop Homography Final Analysis" << std::endl;
     sout << loopAnalyze( false ); // analyze loop homography ( mark good / bad pairs ) and repair bad ones
  }

  sout << " INFO: Hugin Proposal Cleanup ... " << std::endl;
  
  //
  // - - - - - - - - - - - - - - - - - REPORT OUT LOOP HOMOGRAPHY INFO - - - - - - - - - - - -
  //

  int pairsBad = 0;
  int pairsGood = 0; 
  int pairsRepaired = 0;

  int linesBad = 0;
  int linesGood = 0; 
  int linesRepaired = 0;

  sout << " INFO: newCPlist.size() = " << newCPlist.size() << std::endl;

  for( int i = 0; i + 1 < images.size(); i++ )  
     for( int j = i + 1; j < images.size(); j++ )  
        {
            if( pairs[i][j].votesL0 > 0 || pairs[i][j].votesL1 > 0 || pairs[i][j].votesL2 > 0 )
            {
                if( pairs[i][j].repairScalar == 0 )
                {
                   pairsGood++;
                   linesGood += pairs[i][j].newCPlist.size();
                }
                else
                {
                   pairsRepaired++;
                   linesRepaired += pairs[i][j].newCPlist.size();
                }
                
                for( int k = 0; k < pairs[i][j].newCPlist.size(); k++ )
                {
                   newCPlist.push_back( pairs[i][j].newCPlist[k] );
                }                
            }
            else
            {
                if( loopFilter )
                { 
                   pairsBad++;
                   linesBad += pairs[i][j].newCPlist.size();
                }
                else
                {
                   if( pairs[i][j].newCPlist.size() > 0 )
                   {
                       pairsGood++;
                       linesGood += pairs[i][j].newCPlist.size();
                       for( int k = 0; k < pairs[i][j].newCPlist.size(); k++ )
                       { 
                          newCPlist.push_back( pairs[i][j].newCPlist[k] );
                       }
                   }
                   else
                   {
                      pairsBad++;
                      linesBad += pairs[i][j].newCPlist.size();             
                   }
                } // loopFilter == false
                
            } 

        }

  sout << " INFO: Rejected " << pairsBad      << "      Bad Pairs, " << linesBad      << " Proposed Control Points." << std::endl;        
  sout << " INFO:    Added " << pairsGood     << "     Good Pairs, " << linesGood     << " Proposed Control Points." << std::endl;        
  sout << " INFO:    Added " << pairsRepaired << " Repaired Pairs, " << linesRepaired << " Proposed Control Points." << std::endl;        
  sout << " INFO: newCPlist.size() = " << newCPlist.size() << std::endl;



  //
  // - - - - - - - - - - - - - - - - - AGGREGATE HUGIN PROPOSAL - - - - - - - - - - - -
  //



  /* Example lines we need to parse
  
     i w4800 h3200 f0 v=0 Ra=0 Rb=0 Rc=0 Rd=0 Re=0 Eev4.93663791750474 Er1 Eb1 r0 p-2196.08736648679 y0 T
     rX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a=0 b=0 c=0 d=0 e=0 g=0 t=0 Va=0 Vb=0 Vc=0 Vd=0 Vx=0 Vy=0  Vm5 n"DMap/R03
     C10_R03C10_IMGP6975_IMGP6985_ZS_DMap.tif"
  
     c n0 N1 x178.566137505324 y1173.92032073442 X3561.05125004512 Y1190.1670213889 t0

     New control points added via Homography ( prior to this ) are inserted into the live .pto image and then parsed
     along with the control points from the PTO file.   We remember the line number of the first PTO provided 
     control point.  Any of the Homography generated points are already in their ptoImagePair cps[detectorType][pad]
     so when parsing, the cps[detectorType::PTO][pad] is updated, and used in pace of locals.
   */

  // lastCPline is set in the prior CP parsing block ( to import PTO procided CPs into the detectorType::PTO structures )

/*
  int original_pto_cp_line = 0;               // Starting line index of the original PTO control point section
  int original_pto_line_count = lines.size(); // The current count
  int doctored_pto_line_count = lines.size(); // set to the same in case we don't add anything.
  int doctored_pto_cp_line = 0;
*/


  //
  // - - - - - - - - - - - - - - - - MERGE EDITS - - - - - - - - - - -
  //

  //
  // Insert proposed edits to the control point last AFTER the last control point related entry in the PTO
  // ( either an actual control point, or the standard Hugin control point table comment line
  //
  if( newCPlist.size() > 0 )
  {   
      sout << " INFO: Merging " << newCPlist.size() << " proposed edits at line " << lastCPline << " of live PTO." << std::endl;
      if( ! lastCPline == 0 )
      {  
         //
	 // Insert generated cp lines discovered via homography
	 //
         lines.insert(lines.begin() + lastCPline, "# cvfind added control points -->" );
         lines.insert(lines.begin() + lastCPline + 1, "# <-- cvfind added control points" );
         // ToDo: I think this inserts the lines in reverse order ... which is confusing AF
         int k = newCPlist.size() - 1; // ordinal to cardinal
         while( k >= 0 )
	 {
	    lines.insert(lines.begin() + lastCPline + 1, newCPlist[k] );
	    if( loglevel > 6 ) sout << " INFO: Inserted new CP line '" << lines[lastCPline+1] << "' into into live PTO" << std::endl;
	    k--;
	 }
	 sout << " INFO: Inserted " << newCPlist.size() << " new CP entries at line " << lastCPline << " of the live PTO" << std::endl;
      }
      else
      {
         sout << "ERROR: lastCPline was not set -- proposed edits to Hugin PTO will not be merged." << std::endl;
      }

   } // If newCPlist.size() > 0
   else
   {
      sout << " INFO: No proposed edits to Hugin PTO." << std::endl;
   }


  //
  // - - - - - - - - - - - - - - PARSE CONTROL POINTS FROM LIVE PTO ( WITH EDITS ) - - - - - - - - - - - -
  //

  bool insertednewCPlines = false;
  for( int line = 0; line < lines.size(); line++ )
  {

   std::string str = lines[line];
   // If the line is not empty, then check the prefix and parse.
   // "i" = image, "c" = control point
   if( str.length() > 0 )
   {

      //
      // Parse Control Point entry
      //
      if( str[0] == 'c' )
      {  
	 parsed++;
	 
         if( loglevel > 8 )
         {	     
            sout << "TRACE: n=" << findToken( str, " n" )
                 << " N=" << findToken( str, " N" )
	         << " x=" << findToken( str, " x" )
                 << " y=" << findToken( str, " y" )              
                 << " X=" << findToken( str, " X" )
                 << " Y=" << findToken( str, " Y" )
                 << " t=" << findToken( str, " t" ) << std::endl;
         }

         ptoControlPoint cpe;

         cpe.idx1 = stoi( findToken( str, " n" ) );
         cpe.idx2 = stoi( findToken( str, " N" ) );
         cpe.src.x = stof( findToken( str, " x" ) );
         cpe.src.y = stof( findToken( str, " y" ) );
         cpe.dst.x = stof( findToken( str, " X" ) );
         cpe.dst.y = stof( findToken( str, " Y" ) );
         cpe.line = line; // 

         int xSize = images[cpe.idx1].w;
         int ySize = images[cpe.idx1].h;
         
         // if flipping then we flip
         if( huginFlipIn )
         {
            cpe.src.x = xSize - cpe.src.x;   
            cpe.src.y = ySize - cpe.src.y;   
            cpe.dst.x = xSize - cpe.dst.x;   
            cpe.dst.y = ySize - cpe.dst.y;
         } 

         // if flipping, rewrite the line in the live config too.
         if( huginFlipIn )
         {
            std::string s =   "c n" + to_string(cpe.idx1) + " N" + to_string(cpe.idx2)
                            + " x" + std::to_string( cpe.src.x )   
                            + " y" + std::to_string( cpe.src.y )   
                            + " X" + std::to_string( cpe.dst.x )   
                            + " Y" + std::to_string( cpe.dst.y )
                            + " t0" ;  // no idea what t0 does
            lines[line]= s;
         }
         
         cplist.push_back( cpe );

         ptoControlPoint foo2;
         foo2 = cplist.back();

         if( loglevel > 6 )
         {
             sout << "CP: " << foo2.idx1 << " <--> " << foo2.idx2 \
	          << "(" << foo2.src.x << "," << foo2.src.y << ") <--> (" \
	          << foo2.dst.x << "," << foo2.dst.y << ") " << std::endl;
         }
      } // 'c'

   }  // If length > 0

  
  }  // While each line 

  //
  // Give user some statistics and happy feedback
  //
  sout << " INFO: " << cplist.size() << " control points referenced in the .pto file." << std::endl;
  sout << " INFO: " << lines.size() << " lines in-memory .pto.  ( " << parsed << " parsed. )" << std::endl;

      //
      // Dump current match pad for each detector
      //
      if( loglevel > 6 )
      {
         sout << " INFO: Match Pad Dump ( Post Homography )... " << std::endl;
         
         for( int d = 0; d < detectorType::ALL; d++ )  
            for( int i = 0; i + 1 < images.size(); i++ )
               for( int j = i+1; j < images.size(); j++ )
               {
                  if( pairs[i][j].cps[d].size() > 0 )
                  {
                     int pad  = pairs[i][j].currentMatchPad(d);
                     if( pairs[i][j].cps[d][pad].size() > 0 )
                     {
                        sout << " INFO: pairs[" << i << "][" << j << "].cps[" << detectorName[d] << "][pad=" << pad 
                             << "].size() = " << pairs[i][j].cps[d][pad].size() << std::endl;                   
                     }
                  }
                  else
                  {
                     // report when there is no current match pad 
                     if( loglevel > 6 )
                     {
                        sout << " INFO: pairs[" << i << "][" << j << "].cps[" << detectorName[d] 
                             << "].size() == 0" << std::endl;                   
                     }  
                  }
               } // for j ... for i ... for d
      }

  
  

  //
  // - - - - - - - - - - - - - - - - - ANALYSIS - - - - - - - - - - - -
  //
  
  // std::vector<std::vector<ptoImgPair>> pairs( images.size() , std::vector<ptoImgPair> (images.size())); 
  ptoControlPoint cp;	  

  //
  // Calculate slope and scalar displacement for each control point and update the image pair's control point list
  //
  i = 0;
  while( i < cplist.size() )
  {
     // Build the image pairs table
     cp.src = cplist[i].src;
     cp.dst = cplist[i].dst;
     cp.vector.x = cp.dst.x - cp.src.x;
     cp.vector.y = cp.dst.y - cp.src.y;
     cp.line = cplist[i].line; // carry over the line number from the original pto file.

     if( cp.vector.x == 0 )
     {
        cp.slope=cp.vector.y * 1000000; // instead of divide by zero, multiply by 1000000
     }
     else
     {
	cp.slope=cp.vector.y / cp.vector.x;
     }

     cp.scalar=sqrt( ( cp.vector.x * cp.vector.x ) + ( cp.vector.y * cp.vector.y ) );

     pairs[cplist[i].idx1][cplist[i].idx2].idx1 = cplist[i].idx1;
     pairs[cplist[i].idx1][cplist[i].idx2].idx2 = cplist[i].idx2;
     
     pairs[cplist[i].idx1][cplist[i].idx2].points.push_back( cp );

/*
     // If this one of the OG control points from the PTO, add it to the matchPad for the PTO detectorType
     if( cp.line > doctored_pto_cp_line )
     {
        int pad   = pairs[cplist[i].idx1][cplist[i].idx2].currentMatchPad(detectorType::PTO);
        pairs[cplist[i].idx1][cplist[i].idx2].cps[detectorType::PTO][pad].push_back(cp);
     }
*/     
     i++;
  } // for each CP list entry


  //
  // Pass -1 - Check some shit
  //
  // For each source image

/*  This is destructive

  if( true ) sout << " INFO: Consolidated Control Point List For All Detectors: " << std::endl;

  // For each pair, execute refreshPadPoint() to rebuild it from the detector level match / cps pads
  // dump the consolidated list for debug purposes
  for( int src=0; src < images.size(); src++ )
  {  
     // For each destination image ( always a higher index than source ) 
     for( int dst = src + 1; dst < images.size(); dst++ )
     {
        // get the current point pad for this src, dst
        pairs[src][dst].refreshPointPad();
        int pad = pairs[src][dst].currentPointPad();
        // Dump control point table from the point pad.
        if( loglevel > 6 )
        {
           std::vector<std::string> foo = pairs[src][dst].pointPad[pad].dump();
           for( int i = 0; i < foo.size(); i++ ) sout << foo[i] << std::endl;
        }
     }
  }

*/

 
  //
  // Pass 0 -Analyze control points / per image pair to find outliers and poorly connected images
  //

  int cpcountmin = 10000;  // Minimum number of control points per pair
  int cpcountsd = 0;	   // standard deviation of cp per pair ( for non-zero pairs )
  int cpcountmean = 0;  
  int cpcountmax = 0;      // Maximum number of control points per pair
  int cpcountpairs = 0;    // Number of pair with control points
  int cpfrequency[10000];  // keep a count of the number of image pairs for each population count
  std::vector<float> cpcountdata;

  for( int i=0; i < 10000; i++ ) cpfrequency[i]=0;   // Hose out the stalls
  
  // For each source image
  for( int src=0; src < images.size(); src++ )
  {  
     // For each destination image ( always a higher index than source ) 
     for( int dst = src + 1; dst < images.size(); dst++ )
     {
         if( pairs[src][dst].points.size() > 0 )
         {
            cpcountpairs++;
            cpfrequency[pairs[src][dst].points.size()] ++;
            cpcountdata.push_back( (float)pairs[src][dst].points.size() );
            if( pairs[src][dst].points.size() > cpcountmax ) cpcountmax=pairs[src][dst].points.size();
            if( pairs[src][dst].points.size() < cpcountmin ) cpcountmin=pairs[src][dst].points.size();
            if( loglevel > 6 ) sout << " INFO: PAIR [" << src << "] --> [" << dst << "] CP Count: " << pairs[src][dst].points.size() << std::endl;
         }
     }
  }

  cpcountsd   =   calculateSD( cpcountdata );
  cpcountmean = calculateMean( cpcountdata );


  //
  // Update neighbors list for each image based on number of control points per image pair
  //
   

  for( int src=0; src < images.size(); src++ )
  {  
     // For each destination image ( always a higher index than source ) 
     for( int dst=0; dst < images.size(); dst++ )
     {
     
         if( loglevel > 6 && pairs[src][dst].points.size() > 0 ) sout << " INFO: Image [" << src << "] Connect to [" << dst << "] via "
                                << pairs[src][dst].points.size() << " CPs. ";

         if( pairs[src][dst].points.size() > cpPerPairMin )
         {
             images[src].neighbors.push_back( dst );
             images[dst].neighbors.push_back( src );
             if( loglevel > 6 && pairs[src][dst].points.size() > 0 ) sout << "  Pair Good. " << std::endl;
         }
         else
             if( loglevel > 6 && pairs[src][dst].points.size() > 0 ) sout << "  Pair Bad. " << std::endl;    
     }
  }

  // For each source image
  for( int src=0; loglevel > 3 && src < images.size(); src++ )
  {  
     if( images[src].neighbors.size() == 0 )
        sout << "ERROR: " ;
     else
        if( images[src].neighbors.size() < neighborsMin || images[src].neighbors.size() > neighborsMax )
           sout << " WARN: "; 
        else sout << " INFO: ";
 
     sout << "Image [" << src << "] Connects to " << images[src].neighbors.size() << " images: {";
  
     // For each destination image ( always a higher index than source ) 
     for( int i = 0; i < images[src].neighbors.size(); i++ )
     {
         int s, d;

         if( src < images[src].neighbors[i] ) { s = src; d = images[src].neighbors[i]; } 
         else { s = images[src].neighbors[i]; d = src; }  // make sure s < d 

         sout << images[src].neighbors[i]; 
         if( loglevel > 6 ) sout << ":(" << s << "," << d << ")"; 
         if( loglevel > 6 ) sout << pairs[s][d].points.size() ;
         if ( images[src].neighbors.size() > i + 1 )  sout << ", " ;
     }

     sout << "}";

     if( ( loglevel > 6 || images[src].neighbors.size() > 0 ) && images[src].neighbors.size() < neighborsMin ) sout << " Too Few Neighbors ";
     if( images[src].neighbors.size() > neighborsMax ) sout << " Too Many Neighbors ";
     
     sout << std::endl;
  }

  int cpperpair_dist[ 101 ];
  bool zoutput = false;
  for( int pct = 0; pct <= 100; pct += 1 )
  {
     int cpcount_pct = 0;
     int pairs_pct = 0;
     float Z = 0;

     for( int i = cpcountmax; pairs_pct < ( cpcountpairs * pct ) / 100  && i > cpcountmin; i-- )
     {
        cpcount_pct = i; // the current bucket
        pairs_pct = pairs_pct + cpfrequency[i]; 
        if( loglevel > 8 || ( loglevel > 6 && pct == 90 ) ) sout << " INFO: CP per Pair Distribution: CPs per PAIR: " << i << " Num Pairs " << cpfrequency[i] << std::endl;
     }

     if( cpcountsd > 0 ) Z = ( ( (float)pairs_pct - (float)cpcountmean ) / (float)cpcountsd );
     
     if( false && pct == 95 )
     sout << " INFO: " << images.size() << " Images in " << cpcountpairs 
          << " Pairs. CPs/pair: min=" << cpcountmin << ", mean=" << cpcountmean 
          << ", max=" << cpcountmax << ", sd=" << cpcountsd << std::endl;
     
     if( loglevel > 6 || ( pct == 90 || ( ! zoutput && Z >= 3 ) ) )
     {
        sout << " INFO: Control points per pair @ " << pct 
             << "%: " << cpcount_pct << ", mean=" << cpcountmean << ", sd=" << cpcountsd << ", Z=" << Z  
             << ".  " << pairs_pct << " Inlier Pairs. " << cpcountpairs - pairs_pct << " Outlier Pairs." << std::endl;
        zoutput = true;
     }
          
     cpperpair_dist[pct] = pairs_pct;
  }

  //
  // Pass 1 - slope and displacement calculations
  // 
  // Go through each image pair, compute an average slope and displacement for the group of control
  // points linking them.   
  //
  int src, dst, cpidx;
  src = 0;

  if( loglevel > 3 ) sout << std::endl << "Pass 1: Computing displacements and slopes" << std::endl;

  while( src < pairs.size() )
  {

     dst = 0;	  
     while( dst < pairs[src].size() )
     {
if( loglevel > 6 && ! loglevel > 6 )
{	
	if( pairs[src][dst].points.size() > 0 )
	{
        sout << "IMAGE: " << src << " <--> " << dst \
//                  << " GROUP: " << images[src].groupRoot 
                  << " Points: " << pairs[src][dst].points.size() << std::endl;
	}
}        
        pairs[src][dst].cpAve.src.x = 0;
        pairs[src][dst].cpAve.src.y = 0;
        pairs[src][dst].cpAve.dst.x = 0;
        pairs[src][dst].cpAve.dst.y = 0;

        //
        // Compute average delta for the entire control point cloud
        //
	cpidx=0;
	while( cpidx < pairs[src][dst].points.size() )
	{

           pairs[src][dst].cpAve.src.x += pairs[src][dst].points[cpidx].src.x / pairs[src][dst].points.size();
           pairs[src][dst].cpAve.src.y += pairs[src][dst].points[cpidx].src.y / pairs[src][dst].points.size();
           pairs[src][dst].cpAve.dst.x += pairs[src][dst].points[cpidx].dst.x / pairs[src][dst].points.size();
           pairs[src][dst].cpAve.dst.y += pairs[src][dst].points[cpidx].dst.y / pairs[src][dst].points.size();

	   cpidx++;
	}

	// Update vector data for average point
        pairs[src][dst].cpAve.vector.x = pairs[src][dst].cpAve.dst.x - pairs[src][dst].cpAve.src.x;
        pairs[src][dst].cpAve.vector.y = pairs[src][dst].cpAve.dst.y - pairs[src][dst].cpAve.src.y;

        // update slope for the average of all control points between this pair if slope is undefined make it huge
        if( pairs[src][dst].cpAve.vector.x == 0 )
        {
           pairs[src][dst].cpAve.slope=pairs[src][dst].cpAve.vector.y * 1000000; // instead of divide by zero, multiply by 1000000
        }
        else
        {
	   pairs[src][dst].cpAve.slope=pairs[src][dst].cpAve.vector.y / pairs[src][dst].cpAve.vector.x;
        }
        
        // calculate average displacement
        pairs[src][dst].cpAve.scalar=sqrt( ( pairs[src][dst].cpAve.vector.x * pairs[src][dst].cpAve.vector.x ) \
                                    + ( pairs[src][dst].cpAve.vector.y * pairs[src][dst].cpAve.vector.y ) );


if( loglevel > 6 && pairs[src][dst].points.size() > 0 )
{  
         sout << " PAIR: " << pairs[src][dst].idx1 << " <--> " << pairs[src][dst].idx2 \
                   << " Pts: " << pairs[src][dst].points.size() \
                   << " Ave: (" << pairs[src][dst].cpAve.src.x << "," << pairs[src][dst].cpAve.src.y << ") <--> (" \
                   << pairs[src][dst].cpAve.dst.x << "," << pairs[src][dst].cpAve.dst.y << ") "
                   << " Delta: (" << pairs[src][dst].cpAve.dst.x - pairs[src][dst].cpAve.src.x << "," \
		   << pairs[src][dst].cpAve.dst.y - pairs[src][dst].cpAve.src.y << ") "
		   << " Slope: " << pairs[src][dst].cpAve.slope 
		   << " Scalar: " << pairs[src][dst].cpAve.scalar 
		   << std::endl;
}

	      i++;  // do we use i ??

	dst++;
     }

     src++;
  }



  if( loglevel > 3 ) sout << std::endl << "Pass 1a: Compute connected image groups" << std::endl << std::endl;
  bool groupChanged = true;
  i = 0;
  while( groupChanged || i <= images.size() )
  {  
    
//     sout << "Group derivation pass " << i + 1 << " ... " << std::endl;
     src = 0;
     groupChanged = false;
     while( src < pairs.size() )
     {
        dst = 0;	  
        while( dst < pairs[src].size() )
        {
	   if( pairs[src][dst].points.size() > 0 )
           {
	      if( images[dst].groupRoot > images[src].groupRoot )
	      {
	         images[dst].groupRoot = images[src].groupRoot;
if( loglevel > 6 )	         sout << "Image " << dst << " changed to " << images[src].groupRoot << " via Image " << src << std::endl;
	         groupChanged = true;
	      }	      
	   }
	   	   
	   dst++;
        }
        src++;
      }
      if( ! groupChanged ) i++;
   }

/*
                 if( (    loglevel > 3 
                       && forbiddenImages.size() > 1 
                       && binary_search(forbiddenImages.begin(), forbiddenImages.end(), src ) ) || loglevel > 6 ) 
                      sout << "Group " << src << " members: [ ";
*/

     int totalGroups = 0;
     src = 0;
     while( src < images.size() )
     {
        bool dolog = ( loglevel > 6 ) || (    loglevel > 3 && forbiddenImages.size() > 1 
                                && ! binary_search(forbiddenImages.begin(), forbiddenImages.end(), src ) ); 

        dst = 0;
        int grpCnt = 0;	  
        while( dst < images.size() )
        {   	   
           if( images[dst].groupRoot == src )
           {
              grpCnt++;
              if( grpCnt == 1 )
              {  
                 if( dolog ) sout << "Group " << src << " members: [ ";
                 totalGroups++;
              }
              if( dolog && grpCnt > 1 ) sout << ", ";              
              if( dolog ) sout << dst;
           }
	   dst++;
        }
        if( dolog ) if ( grpCnt > 0 ) sout << " ] " << std::endl;
     src++;
     }


//
//  Pass 2 - Try to determine row size automatically. 
//

  if( false && loglevel > 3 ) sout << std::endl << "Pass 2: Row Size Analysis - Method 1" << std::endl << std::endl;

int rowSizeM1 = 300;
int rowSizeCandidate = 12;  // use 12 as a default
int scoreAverage = 0;
int priorAverage = 0;
// This method is trash, disabling
while( false && rowSizeM1 > 3 )
{
  //
  // Go through each image pair, compute an average slope and displacement for the group of control
  // points linking them.  This only looks forward, so no useful for determining total connected-ness.
  //
  
  std::int64_t scoreTotal = 0;
  std::int64_t scoreCount = 0;
  
  src = 0;

  while( src < pairs.size() )
  {
     dst = 0;	  
     while( dst < pairs[src].size() )
     {
	
        //
        // Compute average delta for the entire control point cloud
        //
	cpidx=0;
	while( cpidx < pairs[src][dst].points.size() )
	{
	   cpidx++;
	}

        //
        // Scoring method 1
        //
        int score = 0;
        
        // Bias score based on the number of control points
        score = pairs[src][dst].points.size() * pairs[src][dst].points.size();
        // Up score 2x if next image is horizontally adjacent
        if( src == (dst - 1) && pairs[src][dst].cpAve.slope < 0.1 ) score = score * 2;
        // If destination image is on next row offset by -1, 0, +1, and the slope is more than  
        if( ( dst - src ) >= rowSizeM1 - 1 && ( dst - src ) <= rowSizeM1 + 1 
            && abs( pairs[src][dst].cpAve.slope ) > 1 ) score = score * 1 * abs( pairs[src][dst].cpAve.slope ) ;
        // if the images are too close, add huge downward bias
        if( ( dst - src ) >= 2 && ( dst - src ) <= rowSizeM1 - 1 )
            score = -100 * abs( score );
        // If distance between pairs is >> row size, then multiply by that negative score.
        score = score * 10 * ( src + rowSizeM1 + 2 - dst );
	
	if( score > 0 )
	{
	    scoreTotal = scoreTotal + score;
	    scoreCount++;
	}
	
	dst++;
     }

     src++;
  }
  
  
  if( scoreCount > 0 )
  {
     scoreAverage = scoreTotal / scoreCount;
  }
  else
  {
     scoreAverage = 0;
     if( loglevel > 6 ) sout << "ERROR: scoreCount= " << scoreCount << std::endl;
  }
  
  
  if( loglevel > 6 )
  {
     sout << " INFO: Row Size Guess: " << rowSizeM1 << " Scores: " << scoreCount << " Total: " << scoreTotal \
          << " Average Score: " << scoreAverage << std::endl;
  }
  
  if( priorAverage != 0 && scoreAverage > 2 * priorAverage )
  {
     if( loglevel > 3 ) sout << " INFO: Row Size Guess: " << rowSizeM1 << " Scores: " << scoreCount << " Total: " << scoreTotal \
                        << " Average Score: " << scoreAverage;
     if( loglevel > 3 ) sout << " <--- Candidate " << std::endl;
     rowSizeCandidate = rowSizeM1;
  }     
            
//  sout << std::endl;  
  priorAverage = scoreAverage; 
  rowSizeM1 --;
}



//
//  Pass 2 - Try to determine row size automatically.  Method 2. 
//

  if( loglevel > 1 ) sout << std::endl << "Pass 2: Row Size Analysis - Method 2" << std::endl << std::endl;

  if( loglevel > 3 ) sout << std::endl << "Pass 2: Row Size Analysis - Method 2 - LR=" << shootingLR << " TB=" << shootingTB << std::endl << std::endl;

   std::vector< vector< int >> rows; // a list of rows and row members based on sequentially adjacent pairs
   std::vector< int > row;
   for( int i = 0; i < images.size(); i++ )
   {
      int il = i;
      int ih = i + 1;
      if( row.size() == 0 && ( il == 0 || ih == images.size() ) )
      {
         row.push_back( il );
         if( loglevel > 6 ) sout << "il == " << il << " ih == " << ih 
                                 << ": Added Initial Image " << il << ".  New row: " << row << std::endl;
      }
      else
      {
      
         // if both images exist, and they are a good pair ...
         if(   ! pairs[il][ih].h.empty() && ! images[il].img.empty() && ! images[ih].img.empty() 
              && pairs[il][ih].pointPad.back().size() > cpPerPairMin
              && ( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
           )
         {
            row.push_back( il );
            if( loglevel > 6 ) sout << "il == " << il << " ih == " << ih << ": Added Interior Image " << il 
                                    << ".  New row: " << row 
                                    << " pp.size() = " << pairs[il][ih].pointPad.back().size() << std::endl;
         }
         else
         {
             // Row ended, post row to row list, make a new row, and add image to that row.
             // If shooting pattern specified reverse the order if needed
             row.push_back( il );
             if( loglevel > 6 ) sout << "il == " << il << " ih == " << ih << ": Added Final Image " << il << ".  New row: " << row << std::endl;
             if( shootingLR < 0 ) std::reverse(row.begin(), row.end());
             rows.push_back( row );
             if( loglevel > 6 ) sout << "il == " << il << " ih == " << ih << ": Final New Row: " << row << " added as row " << rows.size() - 1 << std::endl;
             row.clear();
             if( loglevel > 6 ) sout << "il == " << il << " ih == " << ih << ": Cleared New Row " << row << std::endl;
//             row.push_back( il );
//             sout << "il == " << il << " ih == " << ih << ": Added Image " << il << ".  New row: " << row << std::endl;
         }
      }
   }
   
   if( shootingLR > 0 ) std::reverse(row.begin(), row.end());
   if( row.size() > 0 ) rows.push_back( row );
   if( shootingTB > 0 ) std::reverse(rows.begin(), rows.end());
   
   int rowSizeGoodCount = 0;
   int rowSizeBadCount = 0;
   
   sout << " INFO: Row Analysis based on connected images" << std::endl;
   for( int r = 0; r < rows.size(); r++ )
   {
      sout << "       " << rows[r] << " Row ";
      if( rows[r].size() < rowSize - rowSizeTolerance )
      {   
          sout << " Too Short";
          rowSizeBadCount++;
      } 
      else if( rows[r].size() > rowSize + rowSizeTolerance )
      {   
          sout << " Too Long";
          rowSizeBadCount++;
      } 
      else
      {
         sout << " Plausible";
         rowSizeGoodCount++;
      }
      
      sout << std::endl;   
   } 
   sout << " INFO: " << (float)((float)images.size() / (float)rows.size()) << " Images Per Row." << std::endl;   
   if( rowSizeGoodCount == rows.size() )
   {
      rowSizeCandidate = (int)(float)((float)images.size() / (float)rows.size());
      sout << " INFO: All rows are plausible.  Setting rowSizeCandidate to "
           << rowSizeCandidate << std::endl;      
   }
   else
   {
      sout << " WARN: " << rowSizeBadCount << " Bad Rows.  Check sequential images and proper overlap." << std::endl;
   }

//
//  Pass 3 - Scoring and editing 
//

  //
  // Go through each image pair, compute an average slope and displacement for the group of control
  // points linking them.  This only looks forward, so no useful for determining total connected-ness.
  //
  int countBadPair = 0;     // Number with bad score or too few control points
  int countIgnoredPair = 0; // Number of pairs with no control points at all
  int countBadCP = 0;       // number of bad control points redacted

  if( loglevel > 3 ) sout << std::endl << "Pass 3: Tasting and Judgement, Row Size Candidate = " << rowSizeCandidate << std::endl;

  if( loglevel > 3 ) sout << std::endl << " INFO: Apply Peak Width Filter for Adjacent Matches.  Width = +/1 " 
                                  << peakWidth << " Images" << std::endl;

  int PWFpass = 0;
  int PWFfail = 0; 
  for( int src = 0; src < pairs.size(); src ++ )
  {
     
     //
     // Peak Filter - look for spurious matches on either side of a strong match, ignoring row adjancencies
     //
     
     // For each image, scan its image pairs looking for sequential matches.
     for( int dst = 0; dst < pairs[src].size(); dst++ )
     {  
        int center = 0;    // number of matches for center
        int ctrcount = 0;  // number of center samples ( should be 1 )        
        int sideband = 0;  // number of matches for sidebands
        int sbcount = 0;   // number of sideband samples ( should be peakWidth * 2 )        
        int sblo = 0;      // number of matches for lo sidebands
        int sblocount = 0; // number of lo sideband samples ( should be peakWidth )        
        int sbhi = 0;      // number of matches for hi sidebands
        int sbhicount = 0; // number of hi sideband samples ( should be peakWidth )        
        int sbave = 0;
        int sbloave = 0;
        int sbhiave = 0;
        int ctrave = 0;
        float ratio = 0;

        // try narrower and narrower peak widths - handles boundary conditions for src near start or end of the series.
        for( int w = peakWidth; w > 0; w-- )
        { 
           // If we have data to examine for this dst, width combination
           if( dst - w >= 0 && dst + w < pairs[src].size() )
           {
              if( pairs[src][dst - w].pointPad.size() > 0 )
              {
                  sideband += pairs[src][dst - w].pointPad.back().size();
                  sbcount ++;
                  sblo += pairs[src][dst - w].pointPad.back().size();
                  sblocount ++;
              }
              if( pairs[src][dst + w].pointPad.size() > 0 )
              {
                  sideband += pairs[src][dst + w].pointPad.back().size();
                  sbcount ++;
                  sbhi += pairs[src][dst + w].pointPad.back().size();
                  sbhicount ++;
              }
           } // If range of samples is within bounds         
           if( pairs[src][dst].pointPad.size() > 0 )
           {
              center += pairs[src][dst].pointPad.back().size();
              ctrcount ++;
           }
        
        // If a center was present, and there were sidebands, do an analysis 
        }

        bool tdebug = false;

        // If a center was present, and there were sidebands, do an analysis
        // protected from divide by zero requiring all samples 
        if(    ctrcount + sbcount == 1 + ( peakWidth * 2 ) 
            && ctrcount * sbcount * sbhicount * sblocount * sideband > 0 ) // protect from div by zero
        {
           sbave = sideband / sbcount;
           sbloave = sblo / sblocount;
           sbhiave = sbhi / sbhicount;
           ctrave = center / ctrcount;
           ratio = (float)( (float)ctrave / (float)sbave );
           
           if(    sblocount * ctrcount * sbhicount > 0
               && sblo * center * sbhi > 0 ) // we need 3 consecutive matches
           {
              if( tdebug ) sout << " INFO: CP Match Peak [" << src << "][" << dst << "]:"
                                << " Averages = { " << sbloave << " | " << ctrave << " | " << sbhiave << " } "
                                << " Counts = { " << sblocount << " | " << ctrcount << " | " << sbhicount << " } "
                                << " Peak = { " << sbloave << " | " << ctrave << " | " << sbhiave << " } "
                                << " Ratio: " << ratio;                                                  
              if( ratio > 2 )
              {
                 if( tdebug ) sout << " > 2, Match likely good.  Would remove matches from [" << src << "] to "
                                   << " {"  << dst - 2 - (( rowSize - rowSizeTolerance ) / 2) 
                                   << " - " << dst - 2 << " } and"
                                   << " {"  << dst + 2  
                                   << " - " << dst + 2 + (( rowSize - rowSizeTolerance ) / 2) << " }";
                                                     
                 PWFpass++;
              }
              if( ratio < 1 )
              {
                 if( tdebug ) sout << " < 1, Match likely bad.  Would remove center match [" << src << "] --> [" << dst << "]";
                 PWFfail++;
              }   
              if( tdebug ) sout << std::endl;
           }         
        } // if( we won't divide by zero )
     } // for( each dst ... )

     dst = 0;	  
     while( dst < pairs[src].size() )
     {

        if( loglevel > 6 )
        {	
	    if( pairs[src][dst].points.size() > 0 && src == (dst - 1) )
	    {
               sout << "IMAGE: " << src << " <--> " << dst \
                    << " Points: " << pairs[src][dst].points.size() << std::endl;
	    }
        }

        //
        // Compute average delta for the entire control point cloud
        //
	cpidx=0;
	while( cpidx < pairs[src][dst].points.size() )
	{
	   cpidx++;
	}

        //
        // Scoring method 1
        //
        int score = 0;
        
        // Bias score based on the number of control points
        score = pairs[src][dst].points.size() * pairs[src][dst].points.size();
        // Up score 2x if next image is horizontally adjacent
        if( src == (dst - 1) && pairs[src][dst].cpAve.slope < 0.1 ) score = score * 2;
        // If destination image is on next row offset by -1, 0, +1, and the slope is more than  
        if( ( dst - src ) >= rowSizeCandidate - 1 && ( dst - src ) <= rowSizeCandidate + 1 
            && abs( pairs[src][dst].cpAve.slope ) > 1 ) score = score * 1 * abs( pairs[src][dst].cpAve.slope ) ;
        // if the images are too close, add huge downward bias
        if( ( dst - src ) >= 2 && ( dst - src ) <= rowSizeCandidate - 1 )
            score = -100 * abs( score );
        // If distance between pairs is >> row size, then multiply by that negative score.

        // Display the average delta info
	if (pairs[src][dst].points.size() > 0 )
	{  

            if( loglevel > 6 )
            {  
                sout << "   CP: " << pairs[src][dst].idx1 << " <--> " << pairs[src][dst].idx2 \
                     << " Points: " << pairs[src][dst].points.size() \
                     << " cpPerPairMin: " << cpPerPairMin
//                   << " Average: (" << pairs[src][dst].cpAve.src.x << "," << pairs[src][dst].cpAve.src.y << ") <--> (" \
//                   << pairs[src][dst].cpAve.dst.x << "," << pairs[src][dst].cpAve.dst.y << ") "
                     << " Delta: ( " << pairs[src][dst].cpAve.dst.x - pairs[src][dst].cpAve.src.x << "," \
                     << pairs[src][dst].cpAve.dst.y - pairs[src][dst].cpAve.src.y << ") "
		     << " Slope: " << pairs[src][dst].cpAve.slope 
		     << " Scalar: " << pairs[src][dst].cpAve.scalar << " Score: " << score << std::endl;
            }
	}

	
	if( score < 0 || pairs[src][dst].points.size() < cpPerPairMin )
	{
	   if( pairs[src][dst].points.size() > 0 ) countBadPair++;
	   // If pairs seem unmatched based on average scoring, then comment out ALL control points for the pair.
	   int i = 0;
	   while( i < pairs[src][dst].points.size() )
	   {
	      if( ! nochippy ) lines[pairs[src][dst].points[i].line] = "# CHIPPY # BAD PAIR #" + lines[pairs[src][dst].points[i].line];
              if( loglevel > 6 ) sout << "  EDIT: " << lines[ pairs[src][dst].points[i].line ] << std::endl;
	      i++;
	   }	         
	}
	else
	{
           // If pair seems viable check each point for weird stuff
           int i = 0;
           int cp_removed_count = 0;
	   while( i < pairs[src][dst].points.size() )
	   {
              if( loglevel > 6 ) sout << "   CP: " << pairs[src][dst].idx1 << " <--> " << pairs[src][dst].idx2 \
                     << " # " << i << " Range: " << ( pairs[src][dst].cpAve.slope * 0.8 ) \
                     << " --> " << ( pairs[src][dst].cpAve.slope * 1.2 ) \
                     << " Actual: " << pairs[src][dst].points[i].slope ;
               
              if(      ( ( ( pairs[src][dst].cpAve.slope * 0.8 ) < pairs[src][dst].points[i].slope )
                      && ( ( pairs[src][dst].cpAve.slope * 1.2 ) > pairs[src][dst].points[i].slope ) )
                  ||   ( ( ( pairs[src][dst].cpAve.slope * 0.8 ) > pairs[src][dst].points[i].slope )
                      && ( ( pairs[src][dst].cpAve.slope * 1.2 ) < pairs[src][dst].points[i].slope ) )             
                )
                  {
                      if(loglevel > 6) sout << std::endl;                    
                  }
                  else
                  {
                      countBadCP++; // cumulative count
                      cp_removed_count++; // number removed for this image pair
                      if(loglevel > 6) sout << " <-- BAD";
                      if(loglevel > 6) sout << std::endl; 
                      if( ! nochippy ) lines[pairs[src][dst].points[i].line] \
                                          = "# CHIPPY # BAD CP   #" + lines[pairs[src][dst].points[i].line];
                      if(loglevel > 6) sout << "  EDIT: " << lines[ pairs[src][dst].points[i].line ] << std::endl;
                  }
	      i++;
	   }
	   
	   // At this point we may have removed enough control points that we are below the threshold.
	   
	   if( pairs[src][dst].points.size() - cp_removed_count < cpPerPairMin )
	   {
	       countBadPair++;
               if( loglevel > 6 ) sout << " INFO: IMAGE: " << src << " <--> " << dst \
                  << " Removed " << cp_removed_count << " of " << pairs[src][dst].points.size() 
                  << " CPs.  Pair is bad." << std::endl;
               int i = 0;
	       while( i < pairs[src][dst].points.size() )
	       {
	          if( ! nochippy )
	             // if the line is not already commented out
	             if( lines[pairs[src][dst].points[i].line].at(0) != '#' )
	                 lines[pairs[src][dst].points[i].line] = "# CHIPPY # BAD PAIR #" + lines[pairs[src][dst].points[i].line];
                  if( loglevel > 6 ) sout << "  EDIT: " << lines[ pairs[src][dst].points[i].line ] << std::endl;
	          i++;
	       }	         
	   }
	   
	}
	dst++;
     }

  } // For( each src image ) 

  //
  // For each image, do a markup with all neighboring images and their overlap.
  //
  if( saveDiagNeighbors )
  {
     if( loglevel > 3 ) sout << " INFO: Saving Diagnostic Neignborhood Images ..." << std::endl;
     for( int i = 0; i < images.size(); i++ )
     {

         std::vector< cv::Rect> tiles;
         std::vector< int > indices;
         std::vector< std::string > labels;

         if( ! images[i].img.empty() )
         {
             if( loglevel > 3 ) sout << " INFO: Saving Diagnostic Neignborhood Image [" << i << "] ..." << std::endl;

             cv::Mat output = cv::Mat::zeros(cv::Size(3 * images[i].img.cols , 3 * images[i].img.rows), CV_8UC3);
              
             for( int j = 0; j < images.size(); j++ )
             {

                int il = min(i,j);
                int ih = max(i,j);

                if( i == j )
                {
                   // Place reference image[i].img in middle of the canvas

                      
                   cv::Rect jrect = cv::Rect( images[i].img.cols,
                                              images[i].img.rows,
                                              images[i].img.cols,
                                              images[i].img.rows );
                   tiles.push_back( jrect );
                   indices.push_back( j );
                   labels.push_back( "[" + to_string( j ) + "]" );                            

                   images[i].img.copyTo(output(cv::Rect(images[i].img.cols,images[i].img.rows,images[i].img.cols,images[i].img.rows)));
                }
                else
                {
                   // If the pair has a homography solution ( and an image )
                   if(    ! pairs[il][ih].h.empty() && ! images[j].img.empty() && pairs[il][ih].pointPad.back().size() > cpPerPairMin
                       && ( pairs[il][ih].votesL0 > 0 || pairs[il][ih].votesL1 > 0 || pairs[il][ih].votesL2 > 0 )
                     )
                   {
                      // Refresh latest point pad for this pair.
                      // pairs[il][ih].refreshPointPad();
                      
                      // Get a PP with only PTO points ( e.g. those in the PTO or added to it by homography )
                      ptoPointPad pp = pairs[il][ih].pointPad.back();
                      
                      // cout << " INFO: pairs[" << il << "][" << ih << "] pointPad.back().size()=" << pairs[il][ih].pointPad.back().size()
                      //     << " Filtered: " << pp.size() << std::endl; 
                      
                      float dx, dy;

                      // Get latest (control) pointPad                        
                      pp.refresh();
                      // reverse the direction of the delta depending on order of the image pair
                      if( i < j )
                      {
                         dx = pp.com1.x - pp.com2.x;
                         dy = pp.com1.y - pp.com2.y;
                      }
                      else
                      {
                         dx = pp.com2.x - pp.com1.x;
                         dy = pp.com2.y - pp.com1.y;
                      }
                      
                      cv::Rect jrect = cv::Rect( images[i].img.cols + dx,
                                                 images[i].img.rows + dy,
                                                 images[i].img.cols,
                                                 images[i].img.rows );
                      tiles.push_back( jrect );                           
                      indices.push_back( j );                           
                      labels.push_back( "[" + to_string( j ) + "]:" + to_string(pp.size()) );                            
                                                            
                      images[j].img.copyTo(output(jrect));
                   }
                }

             } // For each heighboring image

             for( int t = 0; t < tiles.size(); t++ )
             {
                int r = 127 + ( ( rand() % 4 ) * 32 );
                int g = 127 + ( ( rand() % 4 ) * 32 );
                int b = 127 + ( ( rand() % 4 ) * 32 );
                
                cv::rectangle(output, tiles[t], cv::Scalar(r, g, b), 40 );

//    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
//    cv::rectangle(im, or + cv::Point(0, baseline), or + cv::Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);

//                cv::putText(output, to_string( indices[t] ), Point2f(tiles[t].br() + tiles[t].tl())*0.5, \
                            FONT_HERSHEY_COMPLEX, (int)10, Scalar(r,g,b), (int)32 );

                setLabelCenter(output, labels[t], Point2f(tiles[t].br() + tiles[t].tl())*0.5, \
                            FONT_HERSHEY_COMPLEX, (int)10, Scalar(r,g,b), (int)32 );
             
             }
             
             std::string path = "cvfind_neighbors_" + left0padint(i, 4) + "_" + runID;
             cv::resize(output, output, images[i].img.size());
             imwrite(path + ".png", output);
             
         } // If image is not empty
     } // For each image 
  } // If saving diagnostic images


  if( true )
  {
  
   if( loglevel > 3 )
   {  
      sout << "Learned Ideal Scalars: { ";
      for( int i=0; i < statsScalarHint.size(); i++ )
         if( statsScalarHint[i] > 0 )
            sout << i << ":" << statsScalarHint[i] << "  ";
       sout << " }" << std::endl;
   }
  
   sout << std::endl << "Summary:" << std::endl << std::endl;
   sout << images.size() << " Images in " << totalGroups << " Group(s)" << std::endl;
   sout << "~" << rowSizeCandidate << " Images per row" << std::endl;
   sout << countBadPair << " Bad pairs" << std::endl; 
   sout << countBadCP << " Bad control points" << std::endl;  
  }
   //
   // Write edited PTO file
   //
   if( outputPTO != "" )
   {  
      std::ofstream outfile(outputPTO);
      if(outfile)
       {  
	   sout << "Saving edited PTO to " << outputPTO << " ... " << std::endl;    
           int i = 0;
           while( i < lines.size() )
           {
              outfile << lines[i] << std::endl;
              i++;
           }
           
           // Ouput this stanza because then I can tell what the hell is going on
           outfile << "# Edited by cvfind " << GetDateTime() << " using command line: " << std::endl;
           outfile << "# " << cmdline << std::endl;;
           
           outfile.flush();
           outfile.close();
        }
        else
        {
           sout << "Error writing to " << outputPTO << std::endl;
        }    
    }
     
    if( sout.coss.is_open() )
    {
       sout.coss << "##### CVFIND GLOBAL LOG ENDS " << GetDateTime() << " FOR RUN ID: " << runID << " ######" << std::endl;
       sout.coss.flush();
       sout.coss.close();
    }
    
    if( exitDelay > 0 )
    {
       cout << " INFO: >>> waiting for " << exitDelay << " seconds <<< " << std::endl;
       sleep(exitDelay);
    }
  
  // Welp, we didn't error out, so indicate success  
  return 0;
}

