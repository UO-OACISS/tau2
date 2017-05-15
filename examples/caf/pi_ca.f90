!$Id: pi_ca.f90 25 2016-02-03 15:16:15Z mexas $
!
! Copyright (c) 2016 Anton Shterenlikht, The University of Bristol
!
! See LICENSE for licensing conditions

! calculating pi with coarrays

program pica
implicit none

integer, parameter :: ik = selected_int_kind(10),     &
                      rk = selected_real_kind(15,3)
integer( kind=ik ), parameter :: limit = 2_ik**31_ik
real( kind=rk ), parameter :: one = 1.0_rk,           &
 piref = 3.14159265358979323846264338327950288_rk

integer( kind=ik ) :: i
integer :: img, nimgs
real( kind=rk ) :: pi[*] = 0.0_rk

  img = this_image()
nimgs = num_images()

! Each image sums terms starting from its image number,
! with num_images steps.

do i = img, limit, nimgs
 pi = pi + (-1)**(i+1) / real( 2*i-1, kind=rk )
end do

!write (*,"(a,i0,a,f20.15)") "pi[", img, "]=", pi

sync all

! Image 1 does the reduction

if (img .eq. 1) then
  do i = 2, nimgs
    pi = pi + pi[i] 
  end do
  pi = pi * 4.0_rk
  write (*,*) "coarrays:", nimgs, "images"
  write (*,*) "Series limit", limit
  write (*,*) "Calculated  pi=", pi
  write (*,*) "Reference   pi=", piref
  write (*,*) "Absolute error=", pi-piref
end if

end program pica
