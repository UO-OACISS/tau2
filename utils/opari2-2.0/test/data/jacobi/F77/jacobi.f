      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,maxit)
******************************************************************
* Subroutine HelmholtzJ
* Solves poisson equation on rectangular grid assuming : 
* (1) Uniform discretization in each direction, and 
* (2) Dirichlect boundary conditions 
* 
* Jacobi method is used in this routine 
*
* Input : n,m   Number of grid points in the X/Y directions 
*         dx,dy Grid spacing in the X/Y directions 
*         alpha Helmholtz eqn. coefficient 
*         omega Relaxation factor 
*         f(n,m) Right hand side function 
*         u(n,m) Dependent variable/Solution
*         tol    Tolerance for iterative solver 
*         maxit  Maximum number of iterations 
*
* Output : u(n,m) - Solution 
*****************************************************************
      implicit none 
      integer n,m,maxit
      double precision dx,dy,f(n,m),u(n,m),alpha, tol,omega
*
* Local variables 
* 
      integer i,j,k,k_local 
      double precision error,resid,rsum,ax,ay,b
      double precision error_local, uold(n,m)

      real ta,tb,tc,td,te,ta1,ta2,tb1,tb2,tc1,tc2,td1,td2
      real te1,te2
*
* Initialize coefficients 
      ax = 1.0/(dx*dx) ! X-direction coef 
      ay = 1.0/(dy*dy) ! Y-direction coef
      b  = -2.0/(dx*dx)-2.0/(dy*dy) - alpha ! Central coeff  

      error = 10.0 * tol 

      k = 1

      do while (k.le.maxit .and. error.gt. tol) 

         error = 0.0    

* Copy new solution into old
C$omp parallel private(j,i)
C$omp do
         do j=1,m
            do i=1,n
               uold(i,j) = u(i,j) 
            enddo
         enddo
C$omp end do
* Compute stencil, residual, & update
C$omp do reduction(+:error)
         do j = 2,m-1
            do i = 2,n-1 

*     Evaluate residual 
               resid = (ax*(uold(i-1,j) + uold(i+1,j)) 
     &                + ay*(uold(i,j-1) + uold(i,j+1))
     &                 + b * uold(i,j) - f(i,j))/b
* Update solution 
               u(i,j) = uold(i,j) - omega * resid
* Accumulate residual error
               error = error + resid*resid 
            end do
         enddo
C$omp end do
C$omp end parallel
* Error check 

         k = k + 1

         error = sqrt(error)/dble(n*m)
*
      enddo                     ! End iteration loop 
*
      print *, 'Total Number of Iterations ', k 
      print *, 'Residual                   ', error 

      return 
      end 
