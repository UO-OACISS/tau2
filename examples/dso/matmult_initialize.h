#ifndef _INIT_H_
#define _INIT_H_

#ifdef __cplus_plus 
extern "C" {
#endif /* __cplus_plus */

extern void initialize(double **matrix, int rows, int cols);
extern void compute_lib(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b); 
extern void compute_interchange_lib(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b); 

#ifdef __cplus_plus 
}
#endif /* __cplus_plus */

#endif /* _INIT_H */
