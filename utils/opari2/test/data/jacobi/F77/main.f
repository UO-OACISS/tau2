      program main 
************************************************************
* program to solve a finite difference 
* discretization of Helmholtz equation :  
* (d2/dx2)u + (d2/dy2)u - alpha u = f 
* using Jacobi iterative method. 
*
* Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
* Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
* 
* Directives are used in this code to achieve paralleism. 
* All do loops are parallized with default 'static' scheduling.
* 
* Input :  n - grid dimension in x direction 
*          m - grid dimension in y direction
*          alpha - Helmholtz constant (always greater than 0.0)
*          tol   - error tolerance for iterative solver
*          relax - Successice over relaxation parameter
*          mits  - Maximum iterations for iterative solver
*
* On output 
*       : u(n,m) - Dependent variable (solutions)
*       : f(n,m) - Right hand side function 
*************************************************************
      implicit none 

      integer n,m,mits 
      double precision tol,relax,alpha 

      common /idat/ n,m,mits
      common /fdat/tol,alpha,relax
* 
* Fix input for test cases 
* 
      n=200
      m=200
      alpha=0.8
      relax=1
      tol=1e-10
      mits=5
*
* Calls a driver routine 
* 
      call driver  

      stop
      end 

      subroutine driver ( ) 
*************************************************************
* Subroutine driver () 
* This is where the arrays are allocated and initialzed. 
*
* Working varaibles/arrays 
*     dx  - grid spacing in x direction 
*     dy  - grid spacing in y direction 
*************************************************************
      implicit none 
      double precision omp_get_wtime
      integer n,m,mits 
      double precision tol,relax,alpha 
      double precision r0
      double precision c0

      common /idat/ n,m,mits
      common /fdat/tol,alpha,relax

      double precision u(n,m),f(n,m),dx,dy

* Initialize data

      call initialize (n,m,alpha,dx,dy,u,f)

* Solve Helmholtz equation

      r0 = omp_get_wtime()
      call jacobi (n,m,dx,dy,alpha,relax,u,f,tol,mits)
      r0 = omp_get_wtime() - r0
      write (*,'(a,f12.6)') ' elapsed time usage     ', r0
      write (*,'(a,f12.6)') ' MFlop/s                ', 
     &   1d-6*mits*(m-2d0)*(n-2d0)*13d0 / r0

* Check error between exact solution

      call  error_check (n,m,alpha,dx,dy,u,f)

      return 
      end 

      subroutine initialize (n,m,alpha,dx,dy,u,f) 
******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************
      implicit none 
     
      integer n,m
      double precision u(n,m),f(n,m),dx,dy,alpha
      
      integer i,j, xx,yy
      double precision PI 
      parameter (PI=3.1415926)

      dx = 2.0 / (n-1)
      dy = 2.0 / (m-1)

* Initilize initial condition and RHS
C$omp parallel do private(i,j)
      do j = 1,m
         do i = 1,n
            xx = -1.0 + dx * dble(i-1)        ! -1 < x < 1
            yy = -1.0 + dy * dble(j-1)        ! -1 < y < 1
            u(i,j) = 0.0 
            f(i,j) = -alpha *(1.0-xx*xx)*(1.0-yy*yy) 
     &           - 2.0*(1.0-xx*xx)-2.0*(1.0-yy*yy)
         enddo
      enddo
C$omp end parallel do

*
      return 
      end 

      subroutine error_check (n,m,alpha,dx,dy,u,f) 
      implicit none 
************************************************************
* Checks error between numerical and exact solution 
*
************************************************************ 
     
      integer n,m
      double precision u(n,m),f(n,m),dx,dy,alpha 
      
      integer i,j
      double precision xx,yy,temp,error 

      dx = 2.0 / (n-1)
      dy = 2.0 / (m-1)
      error = 0.0 

      do j = 1,m
         do i = 1,n
            xx = -1.0d0 + dx * dble(i-1)
            yy = -1.0d0 + dy * dble(j-1)
            temp  = u(i,j) - (1.0-xx*xx)*(1.0-yy*yy)
            error = error + temp*temp 
         enddo
      enddo

      error = sqrt(error)/dble(n*m)

      print *, 'Solution Error : ',error

      return 
      end 
