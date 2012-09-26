/*============================================================================*/
/*                        IMAQ Vision Machine Vision                          */
/*----------------------------------------------------------------------------*/
/*    Copyright (c) National Instruments 2004.  All Rights Reserved.          */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Title:       NIMachineVision.h                                             */
/* Purpose:     Provides open source functional extensions for IMAQ Vision.   */
/* Note:        If you need to make changes to this file, create a separate   */
/*              copy in another directory and edit that file. Otherwise, you  */
/*              may lose your changes when upgrading to newer versions of     */
/*              IMAQ Vision.                                                  */
/*                                                                            */
/*============================================================================*/
#if !defined(NiMacVis_h)
#define NiMacVis_h


//============================================================================
//  Includes
//============================================================================
#ifdef _CVI_
    #include <ansi_c.h>
#else
    #include <math.h>
    #include <float.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <stdarg.h>
	#ifndef __GNUC__   
		#include <malloc.h>
	#endif
#endif
#include "nivision.h"


//============================================================================
//	Static strings
//============================================================================
#pragma const_seg("MachineVisionStringData")
//Add read only strings to the "MachineVisionStringData" data segment
static const char defaultAnnulusWindowTitle[]           = "Select an Annulus";
static const char defaultLineWindowTitle[]              = "Select a Line";
static const char defaultPointWindowTitle[]             = "Select a Point";
static const char defaultRectWindowTitle[]              = "Select a Rect";
static const char imaqClampMaxErrorString[]             = "imaqClampMax";
static const char imaqClampMinErrorString[]             = "imaqClampMin";
static const char imaqCountObjectsErrorString[]         = "imaqCountObjects";
static const char imaqFindCircularEdgeErrorString[] 	= "imaqFindCircularEdge";
static const char imaqFindConcentricEdgeErrorString[]	= "imaqFindConcentricEdge";
static const char imaqFindEdgeErrorString[]             = "imaqFindEdge";
static const char imaqFindPatternErrorString[]          = "imaqFindPattern";
static const char imaqFindTransformPatternErrorString[] = "imaqFindTransformPattern";
static const char imaqFindTransformRectErrorString[] 	= "imaqFindTransformRect";
static const char imaqFindTransformRectsErrorString[] 	= "imaqFindTransformRects";
static const char imaqLightMeterLineErrorString[]       = "imaqLightMeterLine";
static const char imaqLightMeterPointErrorString[]      = "imaqLightMeterPoint";
static const char imaqLightMeterRectErrorString[]       = "imaqLightMeterRect";
static const char imaqSelectAnnulusErrorString[]        = "imaqSelectAnnulus";
static const char imaqSelectLineErrorString[]           = "imaqSelectLine";
static const char imaqSelectPointErrorString[]          = "imaqSelectPoint";
static const char imaqSelectRectErrorString[]           = "imaqSelectRect";
//Stop adding strings into the "MachineVisionStringData" data segment
#pragma const_seg()


//============================================================================
//	Constants
//============================================================================
#define DEFAULT_ANGLE_RANGES        NULL
#define DEFAULT_BORDER_SIZE         3
#define DEFAULT_FILL_HOLES          FALSE
#define DEFAULT_MAIN_AXIS_DIR       IMAQ_BOTTOM_TO_TOP
#define DEFAULT_MATCH_FACTOR        0
#define DEFAULT_MATCH_MODE          IMAQ_MATCH_SHIFT_INVARIANT
#define DEFAULT_MAX_SIZE            0
#define DEFAULT_MIN_CONTRAST        10
#define DEFAULT_MIN_LINE_SCORE      990
#define DEFAULT_MIN_MATCH_SCORE     800
#define DEFAULT_MIN_SIZE            0
#define DEFAULT_NUM_MATCHES         1
#define DEFAULT_NUM_RANGES          0
#define DEFAULT_OBJECT_THRESHOLD    128
#define DEFAULT_PIXEL_RADIUS        3
#define DEFAULT_REFINEMENTS         0
#define DEFAULT_REJECT_BORDER       FALSE
#define DEFAULT_SEC_AXIS_DIR        IMAQ_LEFT_TO_RIGHT
#define DEFAULT_SHOW_EDGES_FOUND    FALSE
#define DEFAULT_SHOW_FEATURE_FOUND  FALSE
#define DEFAULT_SHOW_RESULTS        TRUE
#define DEFAULT_SHOW_SEARCH_AREA    FALSE
#define DEFAULT_SHOW_SEARCH_LINES   FALSE
#define DEFAULT_STEEPNESS           2
#define DEFAULT_SUBPIXEL            FALSE
#define DEFAULT_SUBPIXEL_DIVISIONS  10
#define DEFAULT_SUBSAMPLING_RATIO   5
#define DEFAULT_THRESHOLD           40
#define DEFAULT_TYPE                IMAQ_BRIGHT_OBJECTS
#define DEFAULT_USE_MAX_SIZE        FALSE
#define DEFAULT_USE_MIN_SIZE        FALSE
#define DEFAULT_WIDTH               4
#define FIND_TRAN_MIN_MATCH_SCORE   500
#define MEASURE_POINT_WIDTH         3
#define MEASURE_POINT_HEIGHT        3 
#define OVERLAY_FEATURE_SIZE        3
#define OVERLAY_PATTERN_SIZE        9
#define OVERLAY_RESULT_SIZE         5
#define IMAQ_PI                     3.1415926535897932384626433832795028841971


//============================================================================
//	Macros
//============================================================================
#define ROUND_TO_INT(x) ((x > 0) ? (int)(x + 0.5) : (int)(x - 0.5))
#define S_OFFSET(x)     (int)(x - 0.5)
#define M_OFFSET(x)     (int)(x - 1.5)
#define L_OFFSET(x)     (int)(x - 3.5)

//============================================================================
//  Enumerated Types
//============================================================================

//============================================================================
//  Forward Declare Data Structures
//============================================================================
typedef struct FindEdgeOptions_struct FindEdgeOptions;
typedef struct FindPatternOptions_struct FindPatternOptions;
typedef struct FindTransformPatternOptions_struct FindTransformPatternOptions;
typedef struct FindTransformRectOptions_struct FindTransformRectOptions;
typedef struct FindTransformRectsOptions_struct FindTransformRectsOptions;
typedef struct CircularEdgeReport_struct CircularEdgeReport;
typedef struct ObjectReport_struct ObjectReport;
typedef struct CountObjectsOptions_struct CountObjectsOptions;
typedef struct StraightEdgeReport_struct StraightEdgeReport;

//============================================================================
// Data Structures
//============================================================================
#pragma pack(push,1)
typedef struct FindEdgeOptions_struct {
    int    threshold;        //Specifies the threshold for the contrast of an edge.
    int    width;            //The number of pixels that the function averages to find the contrast at either side of the edge.
    int    steepness;        //The span, in pixels, of the slope of the edge projected along the path specified by the input points.
    double subsamplingRatio; //The number of pixels that separates two consecutive search lines.
    int    showSearchArea;   //If TRUE, the function overlays the search area on the image.
    int    showSearchLines;  //If TRUE, the function overlays the search lines used to locate the edges on the image.
    int    showEdgesFound;   //If TRUE, the function overlays the locations of the edges found on the image.
    int    showResult;       //If TRUE, the function overlays the hit lines to the object and the edge used to generate the hit line on the result image.
} FindEdgeOptions;

typedef struct FindPatternOptions_struct {
    MatchingMode        mode;                //Specifies the method to use when looking for the pattern in the image.
    int                 numMatchesRequested; //Number of valid matches expected.
    int                 minMatchScore;       //The minimum score a match can have in order for the function to consider the match valid.
    int                 subpixelAccuracy;    //Set this parameter to TRUE to return areas in the image that match the pattern area with subpixel accuracy.
    RotationAngleRange* angleRanges;         //An array of angle ranges, in degrees, where each range specifies how much you expect the pattern to be rotated in the image.
    int                 numRanges;           //Number of angle ranges in the angleRanges array.
    int                 showSearchArea;      //If TRUE, the function overlays the search area on the image.
    int                 showResult;          //If TRUE, the function overlays the centers and bounding boxes of the patterns it locates on the result image.
} FindPatternOptions;

typedef struct FindTransformPatternOptions_struct {
    MatchingMode        matchMode;        //Specifies the technique to use when looking for the template pattern in the image.
    int                 minMatchScore;    //The minimum score a match can have for the function to consider the match valid.
    int                 subpixelAccuracy; //Set this element to TRUE to return areas in the image that match the pattern with subpixel accuracy.
    RotationAngleRange* angleRanges;      //An array of angle ranges, in degrees, where each range specifies how much you expect the pattern to be rotated in the image.
    int                 numRanges;        //Number of angle ranges in the angleRanges array.
    int                 showSearchArea;   //If TRUE, the function overlays the search area on the image.
    int                 showFeatureFound; //If TRUE, the function overlays the locations of the center of the pattern and the bounding box of the pattern on the image.
    int                 showResult;       //If TRUE, the function overlays the position and orientation of the coordinate system on the result image.
} FindTransformPatternOptions;

typedef struct FindTransformRectOptions_struct {
    int           threshold;              //Specifies the threshold for the contrast of the edge.
    int           width;                  //Specifies the number of pixels that are averaged to find the contrast at either side of the edge.
    int           steepness;              //Specifies the slope of the edge.
    int           subsamplingRatio;       //Specifies the number of pixels that separates two consecutive search lines of the rake.
    RakeDirection mainAxisDirection;      //Specifies the order and direction in which the function searches the edge along the main axis.
    RakeDirection secondaryAxisDirection; //Specifies the order and direction in which the function searches the edge along the secondary axis.
    int           showSearchArea;         //If TRUE, the function overlays the search area on the image.
    int           showSearchLines;        //If TRUE, the function overlays the search lines used to locate the edges on the image.
    int           showEdgesFound;         //If TRUE, the function overlays the locations of the edges found on the image.
    int           showResult;             //If TRUE, the function overlays the hit lines to the object on the result image.
} FindTransformRectOptions;

typedef struct FindTransformRectsOptions_struct {
    int           primaryThreshold;          //Specifies the threshold for the contrast of the edge in the primary rectangle.
    int           primaryWidth;              //Specifies the number of pixels that are averaged to find the contrast at either side of the edge in the primary rectangle.
    int           primarySteepness;          //Specifies the slope of the edge in the primary rectangle.
    int           primarySubsamplingRatio;   //Specifies the number of pixels that separate two consecutive search lines of the rake in the primary rectangle.
    int           secondaryThreshold;        //Specifies the threshold for the contrast of the edge in the secondary rectangle.
    int           secondaryWidth;            //Specifies the number of pixels that are averaged to find the contrast at either side of the edge in the secondary rectangle.
    int           secondarySteepness;        //Specifies the slope of the edge in the secondary rectangle.
    int           secondarySubsamplingRatio; //Specifies the number of pixels that separate two consecutive search lines of the rake in the secondary rectangle.
    RakeDirection mainAxisDirection;         //Specifies the order and direction in which the function searches the edge along the main axis.
    RakeDirection secondaryAxisDirection;    //Specifies the order and direction in which the function searches the edge along the secondary axis.
    int           showSearchArea;            //If TRUE, the function overlays the search area on the image.
    int           showSearchLines;           //If TRUE, the function overlays the search lines used to locate the edges on the image.
    int           showEdgesFound;            //If TRUE, the function overlays the locations of the edges found on the image.
    int           showResult;                //If TRUE, the function overlays the hit lines to the object on the result image.
} FindTransformRectsOptions;

typedef struct CircularEdgeReport_struct {
    PointFloat  center;         //The center of the circle that best fits the circular edge.
    double      radius;         //The radius of the circle that best fits the circular edge.
    double      roundness;      //The roundness of the calculated circular edge.
    PointFloat* coordinates;    //An array of points indicating the location of the detected edge.
    int         numCoordinates; //The number of detected edge coordinates.
} CircularEdgeReport;

typedef struct ObjectReport_struct {
    PointFloat center;       //Specifies the location of the center of mass of the binary object.
    Rect       boundingRect; //Specifies the location of the bounding rectangle of the binary object.
    float      area;         //Specifies the area of the binary object.
    float      orientation;  //Specifies the orientation of the longest segment in the binary object.
    float      aspectRatio;  //Specifies the ratio between the width and the height of the binary object.
    int        numHoles;     //Specifies the number of holes in the binary object.
} ObjectReport;

typedef struct CountObjectsOptions_struct {
    ObjectType type;             //Specifies the types of objects the function detects.
    float      threshold;        //Specifies the grayscale intensity that is used as threshold level.
    int        rejectBorder;     //If TRUE, the function ignores objects touching the boarder of the search area.
    int        fillHoles;        //If TRUE, the function fills the holes in the objects.
    int        useMinSize;       //If TRUE, the function ignores objects the same size or smaller than minSize.
    int        minSize;          //The function ignores objects this size and smaller when useMinSize is TRUE.
    int        useMaxSize;       //If TRUE, the function ignores objects the same size or larger than maxSize.
    int        maxSize;          //The function ignores objects this size and larger when useMaxSize is TRUE.
    int        showSearchArea;   //If TRUE, the function overlays the search area on the image.
    int        showObjectCenter; //If TRUE, the function overlays the location of the center of mass of the objects on the result image.
    int        showBoundingBox;  //If TRUE, the function overlays the bounding boxes of the objects on the result image.
} CountObjectsOptions;

typedef struct StraightEdgeReport_struct {
    PointFloat  start;          //The coordinates location of the start of the calculated edge.
    PointFloat  end;            //The coordinates location of the end of the calculated edge.
    double      straightness;   //The straightness of the calculated edge, which is equal to the least-square error of the fitted line to th entire set of coordinates.
    PointFloat* coordinates;    //An array of detected edge coordinates the function used to calculate the location of the straight edge.
    int         numCoordinates; //The number of detected edge coordinates.
} StraightEdgeReport;

#pragma pack(pop)

#ifdef __cplusplus
    extern "C" {
#endif

//============================================================================
//	Utility functions declaration
//============================================================================
int   imaqFindExtremeEdge( const RakeReport* report, int findClosestEdge, PointFloat* edge );
int   imaqFitAxisReportToRect( Rect rect, RakeDirection mainAxisDirection, RakeDirection secondaryAxisDirection, PointFloat origin, PointFloat* mainAxisEnd, PointFloat* secondaryAxisEnd );
Point imaqMakePointFromPointFloat( PointFloat pointFloat );
int   imaqMeasureMaxDistance( const RakeReport* reportToProcess, LineFloat* firstPerpendicularLine, LineFloat* lastPerpendicularLine, float* distance, PointFloat* firstEdge, PointFloat* lastEdge );	
int   imaqMeasureMinDistance( const RakeReport* firstReport, const RakeReport* secondReport, LineFloat* firstPerpendicularLine, LineFloat* lastPerpendicularLine, float* distance, PointFloat* firstEdge, PointFloat* lastEdge );
int   imaqOverlayArrow( Image* image, Point pointAtArrow, Point pointOnLine, const RGBValue* color, void* reserved );
int   imaqOverlayArcWithArrow( Image* image, ArcInfo* arc, const RGBValue* color, ConcentricRakeDirection direction, void* reserved );
int   imaqOverlayClampResults( Image* image, const ROI* searchArea, const RakeReport* firstReport, const RakeReport* secondReport, const LineFloat* firstPerpendicularLine, 
                              const LineFloat* lastPerpendicularLine, PointFloat firstEdge, PointFloat lastEdge, int forMax, const FindEdgeOptions* options );
int   imaqOverlayCountObjectsResults( Image* image, const ROI* roi, const ObjectReport* reports, int reportCount, const CountObjectsOptions* options );
int   imaqOverlayFindCircularEdgeResults( Image* image, const ROI* roi, const SpokeReport* spokeReport, const CircularEdgeReport* edgeReport, const FindEdgeOptions* options );
int   imaqOverlayFindConcentricEdgeResults( Image* image, const ROI* roi, const ConcentricRakeReport* rakeReport, const StraightEdgeReport* edgeReport, ConcentricRakeDirection direction, const FindEdgeOptions* options );
int   imaqOverlayFindEdgeResults( Image* image, const ROI* roi, const RakeReport* rakeReport, const StraightEdgeReport* edgeReport, const FindEdgeOptions* options );
int   imaqOverlayFindPatternResults( Image* image, const ROI* roi, const PatternMatch* matches, int numMatches, const FindPatternOptions* options );
int   imaqOverlayFindTransformPattern (Image* image, RotatedRect searchRect, const PatternMatch* match, const FindTransformPatternOptions* options );
int   imaqOverlayFindTransformRects( Image* image, const ROI* mainROI, const RakeReport* mainRakeReport, const RakeReport* secondaryRakeReport, PointFloat origin, PointFloat mainAxisEnd, 
                                    PointFloat secondaryAxisEnd, const FindTransformRectOptions* optionsRect, const FindTransformRectsOptions* optionsRects );
int   imaqOverlayLineWithArrow (Image* image, Point start, Point end, const RGBValue* color, int startArrow, int endArrow, void* reserved);
int   imaqOverlayPatternMatch( Image* image, const PatternMatch* match, const RGBValue* color, void* reserved );
void  imaqPrepareForExit( int error, const char* functionName, ... );
int   imaqSetAxisOrienation( RakeDirection mainAxisDirection, RakeDirection secondaryAxisDirection, AxisOrientation* orientation );
int   imaqSplitRotatedRectangle( RotatedRect rectToSplit, RakeDirection direction, RotatedRect* firstRect, RotatedRect* secondRect );

//============================================================================
//  Measure Distances functions
//============================================================================
int __stdcall imaqClampMax(Image* image, RotatedRect searchRect, RakeDirection direction, float* distance, const FindEdgeOptions* options, const CoordinateTransform2* transform, PointFloat* firstEdge, PointFloat* lastEdge);
int __stdcall imaqClampMin(Image* image, RotatedRect searchRect, RakeDirection direction, float* distance, const FindEdgeOptions* options, const CoordinateTransform2* transform, PointFloat* firstEdge, PointFloat* lastEdge);

//============================================================================
//  Coordinate Transform functions
//============================================================================
int __stdcall imaqFindTransformPattern(Image* image, Image* pattern, CoordinateTransform2* transform, RotatedRect searchRect, FindTransformMode mode, const FindTransformPatternOptions* options, AxisReport* report);

//============================================================================
//  Measure Intensities functions
//============================================================================
LineProfile*     __stdcall imaqLightMeterLine(Image* image, Point start, Point end, int showMeasurement, const CoordinateTransform2* transform);
int              __stdcall imaqLightMeterPoint(Image* image, Point point, int showMeasurement, float* intensity, const CoordinateTransform2* transform);
HistogramReport* __stdcall imaqLightMeterRect(Image* image, RotatedRect rect, int showMeasurement, const CoordinateTransform2* transform);

//============================================================================
//  Select Region of Interest functions
//============================================================================
int __stdcall imaqSelectAnnulus(const Image* image, Annulus* annulus, const ConstructROIOptions* options, int* okay);
int __stdcall imaqSelectLine(const Image* image, Point* start, Point* end, const ConstructROIOptions* options, int* okay);
int __stdcall imaqSelectPoint(const Image* image, Point* point, const ConstructROIOptions* options, int* okay);
int __stdcall imaqSelectRect(const Image* image, RotatedRect* rect, const ConstructROIOptions* options, int* okay);

//============================================================================
//  Find Patterns functions
//============================================================================
PatternMatch* __stdcall imaqFindPattern(Image* image, Image* pattern, RotatedRect searchRect, const FindPatternOptions* options, const CoordinateTransform2* transform, int* numMatches);

//============================================================================
//  Count and Measure Objects functions
//============================================================================
ObjectReport* __stdcall imaqCountObjects(Image* image, RotatedRect searchRect, const CountObjectsOptions* options, const CoordinateTransform2* transform, int* numObjects);
int           __stdcall imaqDisposeObjectReport(ObjectReport* report);

//============================================================================
//  Locate Edges functions
//============================================================================
int                 __stdcall imaqDisposeCircularEdgeReport(CircularEdgeReport* report);
int                 __stdcall imaqDisposeStraightEdgeReport(StraightEdgeReport* report);
CircularEdgeReport* __stdcall imaqFindCircularEdge(Image* image, Annulus searchArea, SpokeDirection direction, const FindEdgeOptions* options, const CoordinateTransform2* transform);
StraightEdgeReport* __stdcall imaqFindConcentricEdge(Image* image, Annulus searchArea, ConcentricRakeDirection direction, const FindEdgeOptions* options, const CoordinateTransform2* transform);


#ifdef __cplusplus
    }
#endif

#endif

