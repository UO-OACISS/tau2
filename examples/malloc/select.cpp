#include <Profile/Profiler.h>
/****************************************************************************
** File 	: select.cpp	
** Author 	: Sameer Shende					
** Contents 	: program to calculate the kth largest element in two ways 
** Directory	: $(TAUDIR)/examples/autoinstrument/
** Usage 	: program [<size>] [<k>]
**		  e.g., % klargest 5000 675 
*****************************************************************************/
#include <malloc.h>
#include <string>
#include <vector>
using namespace std;
#include <stdio.h>
#include <sys/time.h>
#include <limits.h> /* for INT_MIN */
#include <stdlib.h>
#include "decl.h"

extern int *M;   /* the median array */
/**************************************************************************
** Function   : select_kth_largest				   
** Parameters : k, array of integers S and n - the size of array.
** Description: The SELECT algorithm to find the kth largest element from 
** 		the set of integers S in O(n) time. Used recursively. 
** Returns    : returns the kth largest element.
***************************************************************************/
int select_kth_largest(int k, int *S, int n)
{
  TAU_PROFILE("int select_kth_largest(int, int *, int)", " ", TAU_USER);
 /* return the kth largest element from the ordered set S of integers */
  /* n is the number of elements in S */
  /* Algorithm : if n < 50 then sort in dec. order S, return kth largest element
	                 in S
		 else
			divide S into ceil(n/5) sequences of 5 elements each 
			with upto four leftover elements
			sort in dec order each 5 element sequence
			let M be the sequence of medians of the 5 element sets
			m = select_kth_largest(ceil(n(M)/2), M, n/5);
			let S1, S2 and S3 be the sequences of elements in S 
			greater than, equal to and less than m respectively.
			if n(S1) >=k then return select_kth_largest(k, S1, n(S1));
			else
				if (n(S1) + n(S2) >= k) then return m
				else 
			return select_kth_largest (k - n(S1) - n(S2), S3);
		
		*/
  int n_M, n_S1, n_S2, n_S3; /* number of elements in M, S1, S2 and S3 resp. */
  int *S1, *S3; /* the other arrays */
  int m; /* the median element */
  int i, leftover, ret;

  if (n < 50)
    return kth_largest_qs(k, S, n);
  /* Use quicksort to sort the array S and return the kth largest element */
    
  leftover = n % 5; /* upto four leftover elements */
  if (leftover != 0)
  { /* then pad the remaining elements with min. negative values */
    for (i=n; i< n + 5 - leftover; i++)
      S[i] = INT_MIN; /* number of leftover elements */
             /* INT_MIN is defined for each system in /usr/include/limits.h */
  }
  

  /* after padding the remaining array, sort the 5 element sequence */
  n_M = ceil(n, 5); /* number of elements in M */

  for(i=0; i< n_M; i++)
  { 
    sort_5elements(&S[5*i]); /* beginning element */
  }
   /* 5 element sequences are sorted */
/*
  display_array(S, ceil(n,5)*5);
*/
   /* size of padded array is ceil(n,5)*5 */
  S1 = (int *) malloc ((3*n/4) * sizeof(int)); /* allocate S1 */
  S3 = (int *) malloc ((3*n/4) * sizeof(int)); /* allocate S3 */
  if((S1 == (int *) NULL) || (S3 == (int *) NULL)) 
  {
    perror("S1 or S3 malloc error");
    exit(1);
  }
  for (i=0; i< n_M; i++)
    M[i] = S[middle(i)]; /* fill up the median array */
  m = select_kth_largest(ceil(n_M,2), M, n_M); /* calculate the median element*/
  /* construct S1, S2, and S3 as sequences of elements in S greater than,  
	equal to and less than m. Don't need a separate S2 array. */
  /* initialize the count of elements in S1, S2, S3 */
  n_S1 = n_S2 = n_S3 = 0;
  /* go through every element in S */
  for (i = 0; i < n; i++)
  {
    if(S[i] > m) S1[n_S1++] = S[i];       /* goes to S1 */
    else if (S[i] < m) S3[n_S3++] = S[i]; /* goes to S3 */
         else n_S2++; /* S2 count incremented goes to S2 */
  } 
  /* now we have S1, and S3 */

  if(n_S1 >= k)  /* S1 has the elements greater than m and there are
	greater than k elements, so its bound to be in this section */
  { /* free memory */
    /* don't need S3 */
    free(S3);
    ret =  select_kth_largest(k, S1, n_S1);
    free(S1); /* before returning clean up S1 */
    return ret;
  }
  else
  {
    if(n_S1 + n_S2 >= k) /* its not in S1, but its in S2 */
    { 
      /* free memory */
      /* don't need S1 and S3 */
      free(S1);
      free(S3);
      return m; /* the median that we had calculated earlier */
    }
    else /* its in S3 */
    {
      /* free memory */
      /* don't need S1 */
      free(S1);
      ret =  select_kth_largest(k - n_S1 - n_S2, S3, n_S3);
      free(S3); /* before returning clean up S3 */ 
      return ret;
    } 
  } 
}

