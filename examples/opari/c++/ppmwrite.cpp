#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include "ppmwrite.h"

/*
 * ColorTable
 */
char ColorTable::operator()(int i, base b) const {
  return tab[i].c[b];
}

ColorTable::~ColorTable() {
  delete [] tab;
}

int ColorTable::numColors() const {
  return num;
}

int ColorTable::shades() const {
  return sds; 
}

ColorTable::ColorTable(int shades, int numColors)
                      : sds(shades), num(numColors) {
    tab = new Color[num];
}

/*
 * PermutationColorTable
 */
PermutationColorTable::
PermutationColorTable(int shades, direction d, base b1, base b2, base b3)
                     : ColorTable(shades, shades*shades*shades) {
  int count = d==fwd ? 0 : numColors()-1;
  for (int ix=0; ix<shades; ++ix) {
    for (int iy=0; iy<shades; ++iy) {
      for (int iz=0; iz<shades; ++iz) {
        tab[count].c[b1] = ix;
        tab[count].c[b2] = iy;
        tab[count].c[b3] = iz;
        count += d;
      }
    }
  }
}

/*
 * SmoothColorTable
 */
SmoothColorTable::
SmoothColorTable(int shades, direction d, base b1, base b2, base b3)
                : ColorTable(shades, 7*(shades-1)+1) {
  int S = shades - 1;
  int i;

  int count = d==fwd ? 0 : numColors()-1;
    tab[count].c[b1] = 0;
    tab[count].c[b2] = 0;
    tab[count].c[b3] = 0;
    count += d;

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = 0;
    tab[count].c[b2] = 0;
    tab[count].c[b3] = i;
    count += d;
  }

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = 0;
    tab[count].c[b2] = i;
    tab[count].c[b3] = S;
    count += d;
  }

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = 0;
    tab[count].c[b2] = S;
    tab[count].c[b3] = S-i;
    count += d;
  }

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = i;
    tab[count].c[b2] = S;
    tab[count].c[b3] = 0;
    count += d;
  }

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = S;
    tab[count].c[b2] = S-i;
    tab[count].c[b3] = 0;
    count += d;
  }

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = S;
    tab[count].c[b2] = 0;
    tab[count].c[b3] = i;
    count += d;
  }

  for (i=1; i<=S; ++i) {
    tab[count].c[b1] = S;
    tab[count].c[b2] = i;
    tab[count].c[b3] = S;
    count += d;
  }
}

/*
 * ppmwrite
 */
void ppmwrite(char *fname, field iterations, int maxiter,
              const ColorTable& table) {
  int out = creat(fname, 0666);
 
  if ( out == -1 ) {
    perror(fname);
    exit(1);
  }

  double factor = double(maxiter) / table.numColors();
  const int buflen = 2048;
  char buf[buflen];
  int b = snprintf(buf, sizeof(buf),  "P6 %5d %5d %d\n", width, height, table.shades()-1);
  int total = 0;

  for (int iy=0; iy<height; ++iy) {
    for (int ix=0; ix<width; ++ix) {
      int idx = iterations[ix][iy];
      total += idx;
      if ( idx == maxiter ) {
        for (int iz=0; iz<3; ++iz) {
          buf[b++] = 0;
          if ( b >= buflen ) {
            write(out, buf, buflen);
            b = 0;
          }
        }
         } else {
        idx = int (idx / factor);
        for (int iz=0; iz<3; ++iz) {
          buf[b++] = table(idx, base(iz));
          if ( b >= buflen ) {
            write(out, buf, buflen);
            b = 0;
          }
        }
      }
    }
  }
  write(out, buf, b);
  close(out);
  printf("%d total iterations\n", total);
}
