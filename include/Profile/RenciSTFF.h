//////////////////////////////////////////////////////////////////////
//
// RENCI STFF Layer
//
// This layer is used for capturing signtaures using the RENCI
// Scalable Time-series Filter Framework. Data collected by the
// TAU profiler can be passed to this layer, and signatures are
// made per function/callsite/counter.
// 
// Scalable Traces compress runtime trace data using least 
// squares and other fitting methods.
// 
// For more information about STFF contact:
//
// Todd Gamblin tgamblin@cs.unc.edu
// Rob Fowler   rjf@renci.org
//
// Or, visit http://www.renci.org
// 
//////////////////////////////////////////////////////////////////////

#ifndef TAU_RENCI_STFF_H
#define TAU_RENCI_STFF_H

#include <vector>

#include "stff/ApplicationSignature.h"
using renci_stff::ApplicationSignature;
using renci_stff::Point;
using renci_stff::SigId;

#include "Profile/Profiler.h"
class FunctionInfo;

/**
 * This class is entirely static, in the vein of TAU's other layers.
 * The STFF layer will initialize itself automatically when recordValues
 * is called. recordValues should be called only after the multiple
 * counter layer is inited, as the STFF layer's init function needs 
 * to know the names of the counters.
 * 
 * NOTE: if, in a future release of TAU, counter setup changes dynamically,
 *       this layer will also need to change.
 */
class RenciSTFF {
private:
  
  /** 
   * Whether stff lib has been inited. We make sure this happens
   * after MPI_Init() if MPI is enable for the stff library.
   */
  static bool inited;
  
  /** Names of counters retrieved from counter layer at init time. */
  static const char **counterList;
  
  /** Local copy of number of TAU counters, obtained at init time. */
  static int numCounters;
  
  /** Vector of all signatures instantiated, so we can stop them easily at the end. */
  static std::vector<ApplicationSignature*> signatures;
  
  /**
   * Creates an application signature and inits it with proper metadata 
   * for the provided metric.
   */
  static ApplicationSignature *createSignature(const Point &point, 
					       const FunctionInfo *info, 
					       int tid, int metricId = 0);
  
  /** Custom handle to string function, for signature filenames. */
  static string function_info_to_string(const void *id);
  
  /** Custom id to string function, for signature filenames. */
  static string handle_to_string(int handle);
  
  /**
   * Initializes the signature layer by creating a signature for each metric monitored.
   * Called by recordValues().
   */
  static bool init(); 
  
public:
  /**
   * Multiple counter version of routine.
   * 
   * Adds measurements to all signatures, from the array of provided values.
   * @param function FunctionInfo describing the function being profiled
   * @param timestamp time the observation is taken
   * @param metricValue double value of counter, taken from FunctionInfo
   */
  static void recordValues(FunctionInfo *function, double timestamp, const double *metricValue, int tid);
  
  /**
   * Shuts down the signature layer and writes all signatures out to specially 
   * named files.
   */
  static void cleanup();		
  
};


#endif //TAU_RENCI_STFF_H


