//
// File:        pycintfc.C 
// Description: C functions used by samint.py
//

#ifndef included_pycintfc
#define included_pycintfc

#include <iostream>
#define MPICH_SKIP_MPICXX
#include <mpi.h>
#define extern_t
#ifndef included_pyGlobals
#include "pyGlobals.h"
#endif

void printGrids(int ngrids, SAMINT::patchgrid* obgrids);
void printLocalData(int nlocal, SAMINT::patchdata* obdata);
int samarcInit(const char *fileName);
int samarcStep(double simtime, double dt);
int samarcRegrid(double simtime);
int samarcWritePlotData(double simtime, int step);
int samarcFinish();
int samarcTest();

int samarcTest()
{
  printf("testing interface\n");
  return(0);
}

int samarcInit(const char *fileName)
{
   printf("initializing SAMARC\n");
   string input_file=fileName;
   SAMINT::initialize(input_file);

   double time=0.;	
 
   printf("building SAMARC initial AMR mesh\n");
   SAMINT::buildGrids(time);

   printf("printing output\n");
   ngrids = SAMINT::getGlobalNumberPatches();
   obgrids = new SAMINT::patchgrid[ngrids];
//   SAMINT::getPatchGrids(ngrids,obgrids);

   nlocal = SAMINT::getLocalNumberPatches();
   obdata = new SAMINT::patchdata[nlocal]; 
//   SAMINT::getPatchData(nlocal,obdata);
   
   return(0);
}

int samarcStep(double simtime, double dt)
{
   SAMINT::timestep(simtime,dt);   
   return(0);
}

int samarcRegrid(double simtime)
{
   SAMINT::regrid(simtime);
   
   // Reset obgrids,obdata to new grid state
   delete [] obgrids;
   delete [] obdata;

   ngrids = SAMINT::getGlobalNumberPatches(); 
   obgrids = new SAMINT::patchgrid[ngrids];
//   SAMINT::getPatchGrids(ngrids,obgrids);

   nlocal = SAMINT::getLocalNumberPatches(); 
   obdata = new SAMINT::patchdata[nlocal];
//   SAMINT::getPatchData(nlocal,obdata);

   return(0); 
}

int samarcWritePlotData(double simtime, int step)
{
   SAMINT::writePlotData(simtime,step);
   return(0);
}

int samarcFinish()
{
   SAMINT::finalize();
   delete [] obgrids;
   delete [] obdata;

   return(0); 
}

void printGrids(int ngrids,
                SAMINT::patchgrid* obgrids)
{
   cout << "\n\nADAPTIVE GRID HIERARCHY:" 
        << "\n\tngrids = " << ngrids << endl;
   
   for (int n = 0; n < ngrids; n++) {
      cout << "\tpatch: " << n 
           << "\tglobal_id: " << obgrids[n].global_id
           << "\tlevel_num: " << obgrids[n].level_num
           << "\tlevel_id: " << obgrids[n].level_id
           << "\tproc_id: " << obgrids[n].proc_id
           << endl;
      for (int i = 0; i < 3; i++) {
         cout << "\t\tilo[" << i << "]: " << obgrids[n].ilo[i]
              << "\tihi[" << i << "]: " << obgrids[n].ihi[i]
              << "\txlo[" << i << "]: " << obgrids[n].xlo[i]
              << endl;
      }
      cout << "\t\tdx: " << obgrids[n].dx << endl;
   }
}

void printLocalData(int nlocal,
                    SAMINT::patchdata* pdata)
{
   cout << "LOCAL PATCH DATA:" 
        << "\n\tnlocal = " << nlocal << endl;
   
   for (int n = 0; n < nlocal; n++) {
      cout << "\tpatch: " << n 
           << "\tglobal_id: " << pdata[n].global_id
           << "\tjd: " << pdata[n].jd
           << "\tkd: " << pdata[n].kd
           << "\tld: " << pdata[n].ld
           << "\tibl: " << pdata[n].ibl
           << "\tq: " << pdata[n].q
           << endl;
   }
}

#endif

