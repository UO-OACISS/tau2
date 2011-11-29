program MAIN
    !***********************************************************************
    ! program to solve a finite difference                                 * 
    ! discretization of Helmholtz equation :                               *
    ! (d2/dx2)u + (d2/dy2)u - alpha u = f                                  *
    ! using Jacobi iterative method.                                       *
    !                                                                      *
    ! Modified: Abdelali Malih,    Aachen University (RWTH), 2007          *
    ! Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998   *
    ! Author  : Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998   *
    !                                                                      * 
    ! Directives are used in this code to achieve paralleism.              *
    ! All do loops are parallized with default 'static' scheduling.        *
    !                                                                      * 
    ! Input :  n - grid dimension in x direction                           *
    !          m - grid dimension in y direction                           *
    !          alpha - Helmholtz constant (always greater than 0.0)        *
    !          tol   - error tolerance for iterative solver                *
    !          relax - Successice over relaxation parameter                *
    !          mits  - Maximum iterations for iterative solver             *
    !                                                                      *
    ! On output                                                            *
    !       : u(n,m) - Dependent variable (solutions)                      *
    !       : f(n,m) - Right hand side function                            *
    !***********************************************************************
  
    use VariableDef
    use JacobiMod
#ifdef _OPENMP
    use omp_lib
#endif
    implicit none
    double precision get_wtime
  
    TYPE(JacobiData) :: myData

!   sets default values or reads from stdin
!    * inits MPI and OpenMP if needed
!    * distribute MPI data, calculate MPI bounds
!    */
    call Init(mydata)

    if ( allocated(myData%afU) .and. allocated(myData%afF) ) then
!        /* matrix init */
        call InitializeMatrix(myData)

!        /* starting timer */
        mydata%fTimeStart = get_wtime()

!        /* running calculations */
        call Jacobi(myData)

!        /* stopping timer */
        mydata%fTimeStop = get_wtime()

!        /* error checking */
        call CheckError(myData)

!        /* print result summary */
        call PrintResults(myData)
    else
        write (*,*) " Memory allocation failed ...\n"
    end if

!    /* cleanup */
    call Finish(myData)
    
end program MAIN

subroutine Init (myData)
    use VariableDef
    use omp_lib
    implicit none
    type(JacobiData), intent(inout) :: myData
    character(len=8) :: env = ' '
    integer :: ITERATIONS = 5 
    integer :: iErr, i
    call get_environment_variable("ITERATIONS", env)
    if (len_trim(env) > 0) then
        read(env,*,iostat=iErr) i
        if ((iErr == 0) .and. (i > 0)) then
            ITERATIONS=i
         else
            print*, "Ignoring ITERATIONS=", env
        endif
    endif
#ifdef _OPENMP
    write (6,*) 'Jacobi', omp_get_max_threads(), 'OpenMP-', _OPENMP, ' thread(s)'
#else
    write (6,*) 'Jacobi (serial)'
#endif
!/* default medium */
        myData%iCols      = 2000
        myData%iRows      = 2000
        myData%fAlpha     = 0.8
        myData%fRelax     = 1.0
        myData%fTolerance = 1e-10
        myData%iIterMax   = ITERATIONS
        write (*,327) "-> matrix size: ", myData%iCols, myData%iRows
        write (*,329) "-> alpha: " , myData%fAlpha
        write (*,329) "-> relax: ", myData%fRelax
        write (*,329) "-> tolerance: ", myData%fTolerance
        write (*,328) "-> iterations: ", myData%iIterMax
        flush(6)
327     format (A22, I10, ' x ', I10)
328     format (A22, I10)
329     format (A22, F10.6)


!    /* MPI values, set to defaults to avoid data inconsistency */
    myData%iMyRank   = 0
    myData%iNumProcs = 1
    myData%iRowFirst = 0
    myData%iRowLast  = myData%iRows - 1

!    /* memory allocation for serial & omp */
    allocate(myData%afU (0 : myData%iCols -1, 0 : myData%iRows -1))
    allocate(myData%afF (0 : myData%iCols -1, 0 : myData%iRows -1))

!    /* calculate dx and dy */
    myData%fDx = 2.0d0 / DBLE(myData%iCols - 1)
    myData%fDy = 2.0d0 / DBLE(myData%iRows - 1)

    myData%iIterCount = 0

end subroutine Init

subroutine InitializeMatrix (myData)
    !*********************************************************************
    ! Initializes data                                                   *
    ! Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)                 *
    !                                                                    *
    !*********************************************************************
    use VariableDef
    implicit none

    type(JacobiData), intent(inout) :: myData 
    !.. Local Scalars .. 
    integer :: i, j, xx, yy
    !.. Intrinsic Functions .. 
    intrinsic DBLE
   
    ! Initilize initial condition and RHS
  
!$omp parallel do private (j, i, xx, yy)
    do j = myData%iRowFirst, myData%iRowLast
        do i = 0, myData%iCols -1
            xx = INT(-1.0 + myData%fDx*DBLE(i)) ! -1 < x < 1
            yy = INT(-1.0 + myData%fDy*DBLE(j)) ! -1 < y < 1
            myData%afU(i, j) = 0.0d0
            myData%afF(i, j) = - myData%fAlpha * (1.0d0 - DBLE(xx*xx))  &
                * (1.0d0 - DBLE(yy*yy)) - 2.0d0 * (1.0d0 - DBLE(xx*xx)) &
                - 2.0d0 * (1.0d0 - DBLE(yy*yy))
        end do
    end do
!$omp end parallel do
end subroutine InitializeMatrix

subroutine Finish(myData)
    use VariableDef
    implicit none

    integer :: iErr
    type(JacobiData), intent(inout) :: myData

    deallocate (myData%afU)
    deallocate (myData%afF)

end subroutine Finish

subroutine PrintResults(myData)
    use VariableDef
    implicit none

    type(JacobiData), intent(inout) :: myData

    if (myData%iMyRank == 0) then
        write (*,328) " Number of iterations : ", myData%iIterCount
        write (*,329) " Residual             : ", myData%fResidual
        write (*,329) " Solution Error       : ", myData%fError
        write (*,330) " Elapsed Time         : ", &
               myData%fTimeStop - myData%fTimeStart
        write (*,330) " MFlops/s             : ", &
               0.000013 * DBLE (myData%iIterCount) &
               * DBLE((myData%iCols - 2) * (myData%iRows - 2)) &
               / (myData%fTimeStop - myData%fTimeStart)
        flush(6)
328     format (A, I8)
329     format (A, F15.12)
330     format (A, F15.6)
    end if
end subroutine PrintResults


subroutine CheckError(myData)
    use VariableDef
    implicit none
    
    type(JacobiData), intent(inout) :: myData
    !.. Local Scalars .. 
    integer :: i, j, iErr
    double precision :: error, temp, xx, yy
    !.. Intrinsic Functions ..
    intrinsic DBLE, SQRT
    ! ... Executable Statements ...
    error = 0.0d0

    do j = myData%iRowFirst, myData%iRowLast
        do i = 0, myData%iCols -1
            xx = -1.0d0 + myData%fDx * DBLE(i)
            yy = -1.0d0 + myData%fDy * DBLE(j)
            temp = myData%afU(i, j) - (1.0d0-xx*xx)*(1.0d0-yy*yy)
            error = error + temp*temp
        end do
    end do

    myData%fError = sqrt(error) / DBLE(myData%iCols * myData%iRows)
   
end subroutine CheckError

double precision function get_wtime()
#ifdef _OPENMP
    use omp_lib
    get_wtime = omp_get_wtime()
#else
    real, dimension(2) :: tarray
    get_wtime = dtime(tarray)
#endif    
    return
end function get_wtime
