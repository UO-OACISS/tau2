/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.cs.uoregon.edu/research/tau             **
 * *****************************************************************************
 * **    Copyright 1997-2017                                                  **
 * **    Department of Computer and Information Science, University of Oregon **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauCaliperTypes.h                                **
 * **      Description     : Type definitions for TAU-CALIPER integration     **
 * **      Contact         : sramesh@cs.uoregon.edu                           **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 * ***************************************************************************/

#ifndef _TAU_CALIPER_TYPES_H_
#define _TAU_CALIPER_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif
enum Type {INTEGER, DOUBLE, STRING};
union Data {
  int as_integer;
  double as_double;
  char str[100];
};

struct StackValue {
  Type type;
  Data data;
};

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_CALIPER_TYPES_H_ */

