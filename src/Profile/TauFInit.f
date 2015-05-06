       subroutine tau_mpi_fortran_init_predefined_constants()
       include 'mpif.h'
       call tau_mpi_predef_init_in_place(MPI_IN_PLACE)
       call tau_mpi_predef_init_bottom(MPI_BOTTOM)
       call tau_mpi_predef_init_status_ignore(MPI_STATUS_IGNORE)
       call tau_mpi_predef_init_statuses_ignore(MPI_STATUSES_IGNORE)
       call tau_mpi_predef_init_unweighted(MPI_UNWEIGHTED)
       return
       end subroutine tau_mpi_fortran_init_predefined_constants

       subroutine tau_mpi_fortran_init_predefined_constants_()
       include 'mpif.h'
       call tau_mpi_predef_init_in_place(MPI_IN_PLACE)
       call tau_mpi_predef_init_bottom(MPI_BOTTOM)
       call tau_mpi_predef_init_status_ignore(MPI_STATUS_IGNORE)
       call tau_mpi_predef_init_statuses_ignore(MPI_STATUSES_IGNORE)
       call tau_mpi_predef_init_unweighted(MPI_UNWEIGHTED)
       return
       end subroutine tau_mpi_fortran_init_predefined_constants_
