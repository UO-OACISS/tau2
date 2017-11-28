%module samint
%include "typemaps.i"
//%include mpi4py/mpi4py.i

%header
%{
#include <numpy/arrayobject.h>
#include <mpi.h>
#include "SAMINT.h"
#include "pyGlobals.h"
extern int samarcInit(const char *fileName);
extern int samarcStep(double simtime, double dt);
extern int samarcRegrid(double simtime);
extern int samarcWritePlotData(double simtime, int step);
extern int samarcFinish();
%}

%inline %{

SAMINT::patchdata *getPdataPy(int i)
{
// return &(SAMINT::pdata[i]);
 return &(obdata[i]);
}

SAMINT::patchgrid *getObgridsPy(int i)
{
// return &(SAMINT::obgrids[i]);
 return &(obgrids[i]);
}

void setPatchDataPy( SAMINT::patchdata* p )
{
 datacountPy=p->jd*p->kd*p->ld;
 dataPy=p->q;
 iblPy=p->ibl;
 global_idPy=p->global_id;
 jdPy=p->jd;
 kdPy=p->kd;
 ldPy=p->ld;
}

void setObgridDataPy(SAMINT::patchgrid* o)
{
 global_idPy=o->global_id;
 level_numPy=o->level_num;
 level_idPy=o->level_id;
 proc_idPy=o->proc_id;
 piloPy=o->ilo;
 pihiPy=o->ihi;
 xloPy=o->xlo;
 dxPy=o->dx;
}

%}

%include "pyGlobals.h"
int samarcInit(const char *fileName);
int samarcStep(double simtime, double dt);
int samarcRegrid(double simtime);
int samarcWritePlotData(double simtime, int step);
int samarcFinish();

%init
%{
 import_array();
%}
