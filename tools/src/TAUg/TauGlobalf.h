/****************************************************************************
**			TAU Portable Profiling Package			                       **
**			http://www.cs.uoregon.edu/research/tau	                       **
*****************************************************************************
**    Copyright 1997-2008                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/

/***************************************************************************
**  File            : globalf.h                                           **
**  Description     : TAU Global (TAUg) implementation header file.       **
**  Author          : Kevin Huck                                          **
**  Contact         : khuck@cs.uoregon.edu                                **
**  Documentation   : See http://tau.uoregon.edu                          **
***************************************************************************/

#ifndef _GLOBALF_H
#define _GLOBALF_H

#include "global.h"

/* external macro declarations */

#define TAU_REGISTER_VIEW(a,b) TauGlobal::tau_register_view(a,b);
#define TAU_REGISTER_COMMUNICATOR(a,b,c) TauGlobal::tau_register_communicator(a,b,c);
#define TAU_GET_GLOBAL_DATA(a,b,c,d,e,f) TauGlobal::tau_get_global_data(a,b,c,d,e,f);

/* class declarations */

extern "C" void tau_register_view_(void);
extern "C" void tau_register_communicator_(void);
extern "C" void tau_get_global_data_(void);

#endif /* _GLOBALF_H */

/* EOF globalf.h */

