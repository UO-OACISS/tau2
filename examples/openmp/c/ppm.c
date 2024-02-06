#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <Profile/Profiler.h>

#include "ppm.h"

#define BUFLEN 2048
#define NUMCOLORS 64
#define NUMSHADES 3

void ppmwrite(char *fname, field iterations, int maxiter) {
  double factor;
  char buf[BUFLEN];
  int b, ix, iy, iz, idx, total;
  char table[NUMCOLORS][NUMSHADES];
  int out;
  TAU_PROFILE_TIMER(pt, "ppmwrite()", "void (char *, field, int)", TAU_DEFAULT);

  TAU_PROFILE_START(pt);
  out = creat(fname, 0666);
 
  if ( out == -1 ) {
    perror(fname);
    TAU_PROFILE_STOP(pt);
    TAU_PROFILE_EXIT("exit on error");
    exit(1);
  }

  idx = NUMCOLORS;
  for (iy=0; iy<=NUMSHADES; ++iy) {
    for (ix=0; ix<=NUMSHADES; ++ix) {
      for (iz=0; iz<=NUMSHADES; ++iz) {
	 --idx;
	 table[idx][0] = ix;
	 table[idx][1] = iy;
	 table[idx][2] = iz;
      }
    }
  }

  total = 0;
  factor = (double)maxiter / NUMCOLORS;
  b = snprintf(buf, sizeof(buf),  "P6 %5d %5d %d\n", width, height, NUMSHADES);

  for (iy=0; iy<height; ++iy) {
    for (ix=0; ix<width; ++ix) {
      idx = iterations[ix][iy];
      total += idx;
      if ( idx == maxiter ) {
        for (iz=0; iz<3; ++iz) {
          buf[b++] = 0;
          if ( b >= BUFLEN ) {
            write(out, buf, BUFLEN);
            b = 0;
          }
        }
         } else {
        idx = (int) (idx / factor);
        for (iz=0; iz<3; ++iz) {
          buf[b++] = table[idx][iz];
          if ( b >= BUFLEN ) {
            write(out, buf, BUFLEN);
            b = 0;
          }
        }
      }
    }
  }
  write(out, buf, b);
  close(out);
  printf("%d total iterations\n", total);
  TAU_PROFILE_STOP(pt);
}
