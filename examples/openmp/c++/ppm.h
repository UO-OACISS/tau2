#define width  800                   /* Resolution */
#define height 800
typedef int field[width][height];    /* Type to store iteration counts */

extern
void ppmwrite(char *fname,           /* Name of PPM file */
              field iterations,      /* Calculated iteration counts */
              int maxiter);          /* Iteration count limit */
