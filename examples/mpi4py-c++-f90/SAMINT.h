//
// File:        SAMINT.h
// Description: Defines interfaces to SAMARC code 
//

#ifndef included_SAMINT
#define included_SAMINT

// system classes
#ifndef included_string
#include <string>
using namespace std;
#define included_string
#endif

/*!
 * This class defines interfaces in raw C to SAMRAI grids and data.  It
 * was built to facilitate Python wrapping for the SAMARC Cartesian 
 * high-order adaptive cartesian code that uses SAMRAI & ARC3D.
 *
 * Example Usage:
 *
 *    \verbatim
 *
 *    // Read SAMARC input file
 *    SAMINT::initialize(input_file):
 *
 *    // Build initial adaptive grid system
 *    SAMINT::buildGrids(time);
 *
 *    // Take a timestep
 *    SAMINT::timestep(time,dt);
 *
 *    // Apply boundary conditions
 *    SAMINT::setFarfieldBCs(time);
 *
 *    // Regrid
 *    SAMINT::regrid(time);
 *
 *    // Write data to output  (plot3d or visit, set in samarc input file)
 *    SAMINT::writePlotData(time,step);
 *   
 *    // Access global (all procs) off-body grid system
 *    int nglobal = SAMINT::getGlobalNumberPatches(); 
 *    SAMINT::patchgrid* obgrids = new SAMINT::patchgrid[nglobal];
 *    SAMINT::getPatchGrids(nglobal,obgrids);
 *
 *    // Access local patch data (pdata) held by grids assigned this processor
 *    int nlocal = SAMINT::getLocalNumberPatches(); 
 *    SAMINT::patchdata* pdata = new SAMINT::patchdata[nlocal];
 *    SAMINT::getPatchData(nlocal,pdata);
 *
 *    \endverbatim
 */

struct SAMINT
{

   /*!
    * @brief Initialize SAMARC
    *
    * Read input file, start SAMRAI timers, and initialize the various
    * "manager" classes used in SAMARC.  This method must be called before
    * any of the other methods are.
    *
    * @param samrai_input_file  SAMRAI input file
    */
   static void initialize(string& samrai_input_file);

   /*!
    * @brief Build initial adaptive Cartesian grid system.
    *
    * Build the initial off-body Cartesian grid hierarchy based on 
    * parameters specified in input and on geometry and flow solution
    * quantities at the initial time.
    *
    * @param time  simulation time
    */
   static void buildGrids(const double time);

   /*!
    * @brief Re-build the off-body grid system at a later timestep.
    *
    * Re-build the off-body Cartesian grid hierarchy from marked
    * geometry and flow solution quantities that define where refinement
    * should take place.  Transfer data from old to new configuration.
    *
    * @param time  simulation time
    */
   static void regrid(const double time);

   /*!
    * @brief Take single flow solver timestep.
    *
    * Perform a timestep in the flow solver.  The solution is advanced
    * from "time" to "time + dt".
    *
    * @param time  simulation time
    * @param dt    size of timestep
    */
   static void timestep(const double time,
                        const double dt);

   /*!
    * @brief Write data to output.
    *
    * Write viz data to Plot3D or VisIt format (format specified in input
    * file).
    *
    * @param time  simulation time
    * @param step  step number 
    */
   static void writePlotData(const double time,
                             const int step);

   /*!
    * @brief Returns the far-field domain extents:
    *
    *    @param farfield_lo[3] lower corner
    *    @param farfield_hi[3] - upper corner
    */
   static void getFarfieldDomainExtents(double* farfield_lo,
                                        double* farfield_hi);

   /*!
    * @brief Returns the total number of patches on all levels.
    */
   static int getGlobalNumberPatches();

   /*!
    * @brief Returns the number of patches that reside on this processor.
    */
   static int getLocalNumberPatches();

   /*!
    * @brief Grid information for each patch
    *
    * @param global_id id in global array of patches
    * @param level_num level number
    * @param level_id  id in level array of patches
    * @param proc_id   processor that patch is assigned to
    * @param ilo[3]    lower corner indices
    * @param ihi[3]    upper corner indices
    * @param xlo[3]    lower corner extent
    * @param dx        grid spacing
    */
   struct patchgrid {
      int global_id;    // id in global array of patches
      int level_num;    // level number
      int level_id;     // id in level array of patches
      int proc_id;      // processor that patch is assigned to
      int ilo[3];      // lower corner indices
      int ihi[3];      // upper corner indices
      double xlo[3];    // lower corner extent
      double dx;        // grid spacing
   };

   /*!
    * @brief Returns global grid information for all patches.  
    *
    * This method returns patch grid information (lower/upper indices, 
    * processor assignment, etc.) for ALL patches in the hierarchy.
    * See 'patchgrid' struct below for information on what is contained 
    * in the struct 'grids'. It assumes the 'grids' array has already 
    * been allocated to size grids[nglobal] before it gets here.  
    *
    * @param ngrids total number of grids in the hierarchy
    * @param grids[ngrids] lower/upper indices of each grid, processor 
    *        assignment,etc.
    */
   static void getPatchGrids(const int ngrids,
                             patchgrid* grids);

   /*!
    * @brief Size (JD,KD,LD) and Data Pointers (IBL,X,Q) pointers for 
    * each local patch.
    *
    * @param global_id id in global array of patches
    * @param jd        number of nodes in X
    * @param kd        number of nodes in Y
    * @param ld        number of nodes in Z
    * @param ibl       pointer to iblank
    * @param q         pointer to q
    */
   struct patchdata {
      int global_id;    // id in global array of patches
      int jd;           // dimensions in X
      int kd;           // dimensions in Y
      int ld;           // dimensions in Z
      int* ibl;         // pointer to IBLANK data
      double* q;        // pointer to Q data
   };

   /*!
    * @brief Returns pointers to data for all local patches 
    *
    * This method returns pointers to data for those patches that
    * reside on this processor. See 'patchdata' struct
    * below to see the information contained in the struct 'pdata'.
    * It assumes the 'pdata' array has already been allocated 
    * to size pdata[nlocal] before it gets here. 
    *
    * @param nlocal number of grids in the hierarchy local to this proc
    * @param pdata[nlocal] size (jd,kd,ld) and pointers to the data
    */
   static void getPatchData(const int nlocal,
                            patchdata* pdata);

   /*!
    * @brief Shuts down interface.
    *
    * This method shuts down the SAMRAI timers and statistics gathering,
    * and deletes all opened objects.  
    */
   static void finalize();

};


#endif
