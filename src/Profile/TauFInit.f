       subroutine tau_mpi_fortran_init_predefined_constants()
       include 'mpif.h'
       call tau_mpi_predef_init_in_place(MPI_IN_PLACE)
       call tau_mpi_predef_init_bottom(MPI_BOTTOM)
       return
       end subroutine tau_mpi_fortran_init_predefined_constants

       subroutine tau_mpi_fortran_init_predefined_constants_()
       include 'mpif.h'
       call tau_mpi_predef_init_in_place(MPI_IN_PLACE)
       call tau_mpi_predef_init_bottom(MPI_BOTTOM)
       return
       end subroutine tau_mpi_fortran_init_predefined_constants_
