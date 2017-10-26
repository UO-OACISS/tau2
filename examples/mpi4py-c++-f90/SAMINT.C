//
// File:        SAMINT.C
// Description: Defines interfaces to SAMRAI data 
//
//#include "SAMRAI_config.h"

// system classes
#include <assert.h>
#include <stdlib.h>
#include <iostream>

// Application classes
#include "SAMINT.h"

// SAMRAI classes
// none

// Fortran interface defns
extern "C" {
   void buildgrids_(const double&);
   void regrid_(const double&);
   void timestep_(const double&, const double&, const double*, const int&);
}



/*
************************************************************************
*
* Initialize SAMRAI
*
************************************************************************
*/
void SAMINT::initialize(string& samarc_input_file)
{
   cout << "SAMINT::initialize()" << endl;
}

/*
************************************************************************
*
* Build initial off-body grid hierarchy
*
************************************************************************
*/
void SAMINT::buildGrids(const double time)
{
   cout << "SAMINT::buildGrids()" << endl;
   buildgrids_(time);
}

/*
************************************************************************
*
* Regrid - re-build off-body grid hierarchy at later timestep
*
************************************************************************
*/
void SAMINT::regrid(const double time)
{
   cout << "SAMINT::regrid()" << endl;
   regrid_(time);
}

/*
************************************************************************
*
* Take a timestep - advance solution from "time" to "time + dt"
*
************************************************************************
*/
void SAMINT::timestep(const double time,
                      const double dt)
{
   int len = 100;

   // Allocate memory
   double * x = (double*)malloc(len*sizeof(double));
   cout << "SAMINT::timestep()" << endl;

   // Call fortran function
   timestep_(time,dt,x,len);

   // Free memory
   free((void*)x);
}

/*
************************************************************************
*
* Write data to output 
* (visit, fieldview, or overgrid - set in samarc input file)
*
************************************************************************
*/
void SAMINT::writePlotData(const double time,
                           const int step)
{
   cout << "SAMINT::writePlotData()" << endl;
}

/*
************************************************************************
*
* Return far-field domain extents
*
************************************************************************
*/
void SAMINT::getFarfieldDomainExtents(double* farfield_lo,
                                      double* farfield_hi)
{
   cout << "SAMINT::getFarfieldDomainExtents()" << endl;
   for (int i = 0; i < 3; i++) {
      farfield_lo[i] = 10.;
      farfield_hi[i] = 10.;
   }
}

/*
************************************************************************
*
* Return global number of patches, all processors all levels
*
************************************************************************
*/
int SAMINT::getGlobalNumberPatches()
{
   cout << "SAMINT::getGlobalNumberPatches()" << endl;
   int nid = 10;
   return(nid);
}

/*
************************************************************************
*
* Return local number of patches
*
************************************************************************
*/
int SAMINT::getLocalNumberPatches()
{
   cout << "SAMINT::getLocalNumberPatches()" << endl;
   int lid = 10;
   return(lid);
}

/*
************************************************************************
*
* Return patch grids - index information only
*
************************************************************************
*/
void SAMINT::getPatchGrids(const int ngrids,
                           patchgrid* grids)
{
   cout << "SAMINT::getPatchGrids()" << endl;
   int n,i,nid;
   
   int nboxes = 10;
   int ln = 0;
   int ip = 0;
   for (n = 0; n < nboxes; n++) {
      grids[n].global_id = n;
      grids[n].level_num = ln;
      grids[n].level_id  = n;
      grids[n].proc_id   = ip;
      for (i = 0; i < 3; i++) {
         grids[nid].ilo[i] = 0;
         grids[nid].ihi[i] = 10;
         grids[nid].xlo[i] = 0.0;
         grids[nid].dx     = 0.1;
      }
   }

}

/*
************************************************************************
*
* Return Q,IBLANK data for local patches
*
************************************************************************
*/
void SAMINT::getPatchData(const int nlocal,
                          patchdata* pdata)
{
   cout << "SAMINT::getPatchData()" << endl;
   int nboxes = 10;
   for (int n = 0; n < nboxes; n++) {
      pdata[n].global_id = n;
      pdata[n].jd  = 10;
      pdata[n].kd  = 10;
      pdata[n].ld  = 10;
      pdata[n].ibl = NULL;
      pdata[n].q   = NULL;
   }
}

/*
************************************************************************
*
* Finalize - shut down SAMRAI
*
************************************************************************
*/
void SAMINT::finalize()
{
   cout << "SAMINT::getPatchData()" << endl;
}



