#ifndef included_pyGlobals
#define included_pyGlobals

#ifdef SWIG
%import "global.i"
#endif /* SWIG */

#include "SAMINT.h"

#ifndef extern_t
#define extern_t extern
#endif

extern_t SAMINT::patchdata *obdata;
extern_t SAMINT::patchgrid *obgrids;

extern_t int ngrids;
extern_t int nlocal;
extern_t double timeSAM;
extern_t int stepSAM;

extern_t int datacountPy;
extern_t double *dataPy;
extern_t int *iblPy;
extern_t int *piloPy;
extern_t int *pihiPy;
extern_t double *xloPy;
extern_t double dxPy;

// patch grid integers

extern_t int level_numPy;
extern_t int level_idPy;
extern_t int global_idPy;
extern_t int proc_idPy;

// patchdata integers

extern_t int jdPy;
extern_t int kdPy;
extern_t int ldPy;

#endif
