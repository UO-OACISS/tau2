/****************************************************************************
** File 	: decl.h
** Author 	: Sameer Shende					
** Contents 	: function prototypes for kl.c to calculate kth largest elt.
*****************************************************************************/
#ifndef _DECL_H
#define _DECL_H


/* function declarations */

int  select_kth_largest(int k, int *S, int n);
void interchange(int *a, int *b);
void setup(int *arr);
void quicksort(int *arr, int m, int n);
void sort_5elements(int *arr);
int  kth_largest_qs(int k, int *arr, int size);
void display_array(int *A, int nelems);
int  floor (int num, int den);
int  ceil (int num, int den);


#endif _DECL_H

/* EOF decl.h */
