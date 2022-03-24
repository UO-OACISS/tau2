#ifndef _INIT_H_
#define _INIT_H_

#ifdef __cplus_plus 
extern "C" {
#endif /* __cplus_plus */

extern void initialize(double **matrix, int rows, int cols);
extern void initialize(float  **matrix, int rows, int cols);
extern void initialize(int    **matrix, int rows, int cols);

#ifdef __cplus_plus 
}
#endif /* __cplus_plus */

#endif /* _INIT_H */
