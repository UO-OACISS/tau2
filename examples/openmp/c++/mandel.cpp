/* This program was written by Bernd Mohr, FZJ, Germany. It illustrates the
   use of OpenMP in this Mandelbrot program. The output of this program is an 
   image stored in mandel.ppm.  
*/
// #include <cstdio>
//   using std::fprintf;
//   using std::printf;
#include <stdio.h>

// #include <cstdlib>
//   using std::strtod;
//   using std::atoi;
//   using std::exit;
#include <stdlib.h>
#include <Profile/Profiler.h>
#include <iostream.h>


extern "C" void mytimer_(int *);

#ifdef STD_COMPLEX
#  include <complex>
   typedef std::complex<double> dcomplex;
   using std::norm;
#else
#  include "TComplex.h"
   typedef TComplex<double> dcomplex;
#endif

#ifdef _OPENMP
extern "C" {
#  include <omp.h>
}
#endif

#include "ppmwrite.h"

void foo(void)
{
  TAU_PROFILE("foo", " ", TAU_DEFAULT);
}
int main(int argc, char *argv[]) {
  double xmin, xmax, ymin, ymax;
  int maxiter;
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);
  // --- init input parameter
  //     -1.5 0.5 -1.0 1.0
  //     -.59 -.54 -.58 -.53
  //     -.65 -.4 .475 .725
  //     -.65 -.5 .575 .725
  //     -.59 -.56 .47 .5
  //     -.535 -.555 .49 .51
  if ( argc != 6 ) {
    fprintf(stderr, "%s: xmin xmax ymin ymax maxiter\n", argv[0]);
    fprintf(stderr, "Using defaults: -.59 -.56 .47 .5 216\n");
    xmin = -.59; xmax = -.56; ymin = .47; ymax = .5; maxiter = 216; 
  }
  else {
    xmin = strtod(argv[1], 0);
    xmax = strtod(argv[2], 0);
    ymin = strtod(argv[3], 0);
    ymax = strtod(argv[4], 0);
    maxiter =  atoi(argv[5]);
    cout <<"Using xmin = "<<xmin <<" xmax = "<<xmax<<" ymin = "<<ymin
	 <<" ymax = "<<ymax<<" maxiter = "<<maxiter<<endl;
  }

  // --- initialization
  int numpe = 1;
  double dx = (xmax - xmin) / width;
  double dy = (ymax - ymin) / height;

  // --- calculate mandelbrot set
  field iterations;

  mytimer_(0);
#pragma omp parallel
  {
    TAU_PROFILE("Parallel Region", " " , TAU_DEFAULT);
    int ix;
#ifdef _OPENMP
    numpe = omp_get_num_threads();
#endif
#pragma omp for
    for (ix=0; ix<width; ++ix) {
      double x = xmin + ix*dx;
      foo();
      for (int iy=0; iy<height; ++iy) {
        double y = ymin + iy*dy;
        dcomplex c(x, y), z(0.0, 0.0);
        int count = 0;
        while ( norm(z) < 16 && count < maxiter ) {
          z = z*z + c;
          ++count;
        }
        iterations[ix][iy] = count;
      }
    }

  }
  mytimer_(&numpe);

  // --- generate ppm file
  printf("Writing picture ...\n");
  ppmwrite("mandel.ppm", iterations, maxiter);
  return 0;
}

