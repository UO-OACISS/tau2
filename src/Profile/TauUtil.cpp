/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauUtil.cpp                                      **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : This file contains utility routines              **
**                                                                         **
****************************************************************************/

#include <TauUtil.h>
#include <stdarg.h>
#include <string.h>

/*********************************************************************
 * Write to output device
 ********************************************************************/
int Tau_util_output(Tau_util_outputDevice *out, const char *format, ...) {
  int rs;
  va_list args;
  if (out->type == TAU_UTIL_OUTPUT_BUFFER) {
    va_start(args, format);
    rs = vsprintf(out->buffer+out->bufidx, format, args);
    va_end(args);
    out->bufidx+=rs;
    if (out->bufidx+TAU_UTIL_OUTPUT_THRESHOLD > out->buflen) {
      out->buflen = out->buflen * 2;
      out->buffer = (char*) realloc (out->buffer, out->buflen);
    }
  } else {
    va_start(args, format);
    rs = vfprintf(out->fp, format, args);
    va_end(args);
  }
  return rs;
}

/*********************************************************************
 * Read an entire line from a file
 ********************************************************************/
int Tau_util_readFullLine(char *line, FILE *fp) {
  int ch;
  int i = 0; 
  while ( (ch = fgetc(fp)) && ch != EOF && ch != (int) '\n') {
    line[i++] = (unsigned char) ch;
  }
  line[i] = '\0'; 
  return i; 
}

/*********************************************************************
 * Replaces all the runs of spaces with a single space in a string.
 * This modifies the string, but the user should use the return string
 * because the pointer may change while removing leading whitespace.
 ********************************************************************/
char *Tau_util_removeRuns(char *str) {
  int i, idx;
  int len; 

  if (!str) {
    return str; /* do nothing with a null string */
  }

  // also remove leading whitespace
  while (*str && *str == ' ') {
    str++;
  }

  len = strlen(str);
  for (i=0; i<len; i++) {
    if (str[i] == ' ') {
      idx = i+1;
      while (idx < len && str[idx] == ' ') {
	idx++;
      }
      int skip = idx - i - 1;
      for (int j=i+1; j<=len-skip; j++) {
	str[j] = str[j+skip];
      }
    }
  }
  return str;
}
