#include <stdio.h>
#include <math.h>

#define ITERATIONS 1000000000

__attribute__((target(mic))) float compute_pi(int start)
{
	 float pi = 0.0f;
	 float p16 = 1;
	 int i;
#pragma omp parallel for private(i) reduction(+:pi)
	 for (i=start; i<start+ITERATIONS/2; i++)
	 {	
	 		p16 = pow(16,i);
			pi += 1.0/p16 * (4.0/(8*i + 1) - 2.0/(8*i + 4) -
	          1.0/(8*i + 5) - 1.0/(8*i + 6));
	 }
	 return pi;
}

int main()
{
	float pi = 0.0f;

//First offload half the computation to the mic card.
#pragma offload target (mic)
	{
	 	pi = compute_pi(0);
	}
		 
//Compute the second half on the host.
	{
		pi += compute_pi(ITERATIONS/2);
	}
	
	printf("Approximation of PI: %.20f.\n", pi);
		
}
