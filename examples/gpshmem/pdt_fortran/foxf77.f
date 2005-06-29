!
! $Id: foxf77.f,v 1.1 2005/06/29 19:15:31 sameer Exp $
!
! program in f77 for fox's code using internal shmem calls
!                should work with g77
!
      program foxf77
      implicit none
      integer rank
      parameter (rank=840)
      integer coresize,corefirst
      parameter (coresize = 1 000 000)
      integer sizeofcore,pcore_C,pcore_Z,
     &    pcore_Atmp,pcore_Btmp
      common /acoreinfo/sizeofcore,corefirst,pcore_C,pcore_Z,
     &    pcore_Atmp,pcore_Btmp
      double precision mycore
      common /acore/mycore(coresize)
      integer size, error, abort
      integer i, j, k, nproc, me
      integer nn, nproctest
      integer local_rank, myrow, mycol
      integer my_low_r, my_high_r, my_low_c, my_high_c
      integer gpmype
      integer gpnumpes
      integer handle_A, index_A
      integer handle_B, index_B
      external gpmype
      external gpnumpes
      logical master
      double precision norm, normall
      include 'gps_arrays.Fh'
      integer gpshmalloci
      external gpshmalloci
!
      sizeofcore = coresize
      corefirst = 1
      pcore_C = 0
      pcore_Z = 0
      pcore_Atmp = 0
      pcore_Btmp = 0
!
! Initialize SHMEM and process IDs, NPROC
!
      call gpshmem_init()
      me = gpmype()
      nproc = gpnumpes()
      master = me.eq.0
      
!
! Check for perfect square
!
      nn = int(sqrt(dble(nproc)))
      nproctest = nn**2
      call gpshmem_barrier_all()
      if (nproc > 1) then
        if (nproc /=  nproctest) then
          if (master) then
            write(6,*)' nproc not a perfect square! '
            write(6,*)' nproctest = ',nproctest
            write(6,*)' nproc     = ',nproc
          endif
          call gpshmem_error('perfect square fatal error  ')
        endif
      endif

!
! Check rank remainder with respect to sqrt(nproc)
!
      if (mod(rank,nn) /= 0) then
        if (master) then
          write(6,*)' rank not divisible by sqrt(nproc)!' 
          write(6,*)' rank = ',rank,' sqrt(nproc) = ',nn,
     &        ' remainder is:',mod(rank,nn)
        endif
        call gpshmem_error('remainder fatal error  ')
      endif
!
! compute process row and column IDs 
!
      call gpshmem_barrier_all()
      myrow = me / nn
      mycol = mod(me,nn)
      write(6,'(1x,a,i4,a,i4,a,i4,a,i4)')' me = ',me,
     &    ' nproc = ',nproc,' row = ',myrow,' col = ',mycol
!
! compute local ranks and ranges
!
      call gpshmem_barrier_all()
      local_rank = rank/nn
      my_low_r  = myrow*local_rank + 1
      my_high_r = my_low_r + local_rank - 1
      my_low_c  = mycol*local_rank + 1
      my_high_c = my_low_c + local_rank - 1
      size = local_rank*local_rank
      write(6,
     & '(2x,a,i4,1x,a,i4,1x,a,i4,3x,a,i4,a,i4,a,1x,a,i4,1x,a,i4,a,/)')
     &    'me:',me,'rank:',rank,'local rank:',local_rank,
     &    'rows(',my_low_r,',',my_high_r,')',
     &    'cols(',my_low_c,',',my_high_c,')'
      call flush(6)
      handle_A = gpshmalloci(4,size,index_A)
      call fillit(GPS_DPREAL(index_A),size,0.0d00)
      call gen_a(GPS_DPREAL(index_A),my_low_r,my_high_r,
     &    my_low_c,my_high_c,rank,rank)
      handle_B = gpshmalloci(4,size,index_B)
      call fillit(GPS_DPREAL(index_B),size,0.0d00)
      call gen_b(GPS_DPREAL(index_B),my_low_r,my_high_r,
     &    my_low_c,my_high_c,rank,rank)
!
      if ((corefirst+size).gt.sizeofcore) then
         stop ' mycore allocation error C'
      endif
      pcore_C = corefirst
      corefirst = corefirst + size
      if (master) write(6,*)' core used on node 0: ',(corefirst - 1)
      call fillit(mycore(pcore_C),size,0.0d00)
      if ((corefirst+size).gt.sizeofcore) then
         stop ' mycore allocation error Z'
      endif 
      pcore_Z = corefirst
      corefirst = corefirst + size
      if (master) write(6,*)' core used on node 0: ',(corefirst - 1)
      call fillit(mycore(pcore_Z),size,0.0d00)
      call gen_c(mycore(pcore_z),my_low_r,my_high_r,
     &    my_low_c,my_high_c,rank,rank,rank)
!
      if ((corefirst+size).gt.sizeofcore) then
         stop ' mycore allocation error Atmp'
      endif
      pcore_Atmp = corefirst
      corefirst = corefirst + size
      if (master) write(6,*)' core used on node 0: ',(corefirst - 1)
      call fillit(mycore(pcore_Atmp),size,0.0d00)
      if ((corefirst+size).gt.sizeofcore) then
         stop ' mycore allocation error Btmp'
      endif
      pcore_Btmp = corefirst
      corefirst = corefirst + size
      if (master) write(6,*)' core used on node 0: ',(corefirst - 1)
      call fillit(mycore(pcore_Btmp),size,0.0d00)
      call foxit(GPS_DPREAL(index_A),GPS_DPREAL(index_B),
     &    mycore(pcore_C),
     &    rank,local_rank,myrow,mycol,me,nproc,nn,
     &    mycore(pcore_Atmp),mycore(pcore_Btmp))
!
      call gpshmem_barrier_all()
      call computenorm(norm,mycore(pcore_C),mycore(pcore_Z),size)
      call gpshmem_barrier_all()
      write(6,*)' norm on node:',me,' is ',norm
      call flush(6)
      call gpshmem_barrier_all()
      i = 0
      j = 0
      call gpshmem_double_sum_to_all(normall,norm,1,0,0,nproc,i,j)
      call gpshmem_barrier_all()
      if (master)
     &    write(6,*)' global norm: ',normall
      call flush(6)
!
      call gpshfree_handle(handle_A)
      call gpshfree_handle(handle_B)
      call gpshmem_finalize()
!
          end
      subroutine foxit(a,b,c,rank,local_rank,myrow,mycol,me,nproc,nn,
     &    Atmp,Btmp)
      integer rank, local_rank, me, nproc, nn
      integer myrow, mycol
      double precision a(local_rank,local_rank)
      double precision b(local_rank,local_rank)
      double precision c(local_rank,local_rank)
      double precision Atmp(local_rank,local_rank)
      double precision Btmp(local_rank,local_rank)
!
      integer i,j,ii,jj,kk,n2ga,n2gb
      integer mysize
      integer stage
!
      integer procfromgrid, inc_wrap 
      external procfromgrid, inc_wrap
!
      mysize = local_rank*local_rank
!
      do i=1,local_rank
        do j=1,local_rank
          Atmp(i,j) = 0.0d00
          Btmp(i,j) = 0.0d00
        enddo
      enddo
      ii = myrow
      jj = mycol 
      do stage=0,(nn-1),1
        call gpshmem_barrier_all()
        kk = inc_wrap(ii,stage,nn)
        if (mycol == kk) then
          do i=1,local_rank
            do j=1,local_rank
              Atmp(i,j) = A(i,j)
            enddo
          enddo
        else
          n2ga = procfromgrid(ii,kk,nn)
          call gpshmem_get(Atmp,A,mysize,n2ga)
        endif
        if (myrow == kk) then
          do i=1,local_rank
            do j=1,local_rank
              Btmp(i,j) = B(i,j)
            enddo
          enddo
        else
          n2gb = procfromgrid(kk,jj,nn)
          call gpshmem_get(Btmp,B,mysize,n2gb)
        endif
!     call gpshmem_barrier_all()
        do i=1,local_rank
          do j=1,local_rank
            do k=1,local_rank
              C(i,j) = C(i,j) + Atmp(i,k)*Btmp(k,j)
            enddo
          enddo
        enddo
      enddo
!
      end
      integer function inc_wrap(i,j,n)
      integer i,j,n
      inc_wrap = mod((i+j+n),n)
      end
      integer function  procfromgrid(i,j,n)
      integer i,j,n
      procfromgrid =  i * n +j
      end
      subroutine fillit(Z,len,value)
      implicit none
      integer len
      double precision z(len)
      double precision value
      integer i
      do i = 1,len
        Z(i) = value
      enddo
      end
      subroutine gen_a(a,ilo,ihi,jlo,jhi,rowdim,coldim)
      implicit none
      double precision aval, bval, cval
      parameter (aval = 5.0d00/8.0d00)
      parameter (bval = 1.0d00/7.0d00)
      parameter (cval = 5.0d00)
      integer ilo, ihi, jlo, jhi
      integer rowdim, coldim
      double precision a(ilo:ihi,jlo:jhi)
!
      integer i,j, ii, jj
      logical oil, oih, ojl, ojh, ohilo, oall
      integer gpmype
      external gpmype
!
      ohilo = ((ilo > ihi).or.(jlo > jhi))
      oil   = ((ilo < 0) .or. (ilo > rowdim))
      oih   = ((ihi < 0) .or. (ihi > rowdim))
      ojl   = ((jlo < 0) .or. (jlo > coldim))
      ojh   = ((jhi < 0) .or. (jhi > coldim))
      oall = ohilo.or.oil.or.oih.or.ojl.or.ojh
      if (oall) then
        write(6,*)' something wrong with gen_a arguments ',gpmype()
        write(6,*)' ilo = ',ilo, '     ihi = ',ihi
        write(6,*)' jlo = ',jlo, '     jhi = ',jhi
        write(6,*)' rowdim = ',rowdim,'   coldim = ',coldim
        write(6,*)oall,ohilo,oil,oih,ojl,ojh
        stop ' fatal error gen_a'
      endif
      do i = ilo,ihi
        ii = i - 1
        do j = jlo,jhi
          jj = j - 1
          A(i,j) = AVAL*dble(ii) + BVAL*dble(jj) + CVAL
        enddo
      enddo
      end
      subroutine gen_b(b,ilo,ihi,jlo,jhi,rowdim,coldim)
      implicit none
      double precision dval, eval, fval
      parameter (dval = -2.0d00/13.0d00)
      parameter (eval = 6.0d00/5.0d00)
      parameter (fval = 2.0d00)
      integer ilo, ihi, jlo, jhi
      integer rowdim, coldim
      double precision b(ilo:ihi,jlo:jhi)
!
      integer i,j,ii,jj
      logical oil, oih, ojl, ojh, ohilo, oall
      integer gpmype
      external gpmype
!
      ohilo = ((ilo > ihi).or.(jlo > jhi))
      oil   = ((ilo < 0) .or. (ilo > rowdim))
      oih   = ((ihi < 0) .or. (ihi > rowdim))
      ojl   = ((jlo < 0) .or. (jlo > coldim))
      ojh   = ((jhi < 0) .or. (jhi > coldim))
      oall = ohilo.or.oil.or.oih.or.ojl.or.ojh
      if (oall) then
        write(6,*)' something wrong with gen_b arguments ',gpmype()
        write(6,*)' ilo = ',ilo, '     ihi = ',ihi
        write(6,*)' jlo = ',jlo, '     jhi = ',jhi
        write(6,*)' rowdim = ',rowdim,'   coldim = ',coldim
        write(6,*)oall,ohilo,oil,oih,ojl,ojh
        stop ' fatal error gen_b'
      endif
      do i = ilo,ihi
        ii = i - 1
        do j = jlo,jhi
          jj = j - 1
          B(i,j) = DVAL*dble(ii) + EVAL*dble(jj) + FVAL
        enddo
      enddo
      end
      subroutine gen_c(c,ilo,ihi,jlo,jhi,rowdim,coldim,crossdim)
      implicit none
      double precision aval, bval, cval, dval, eval, fval
      parameter (aval = 5.0d00/8.0d00)
      parameter (bval = 1.0d00/7.0d00)
      parameter (cval = 5.0d00)
      parameter (dval = -2.0d00/13.0d00)
      parameter (eval = 6.0d00/5.0d00)
      parameter (fval = 2.0d00)
      integer ilo, ihi, jlo, jhi
      integer rowdim, coldim, crossdim
      double precision c(ilo:ihi,jlo:jhi)
!
      double precision sum
      integer i,j,k, ii, jj, kk
      integer gpmype
      logical oil, oih, ojl, ojh, ohilo, oall
      external gpmype
!
      ohilo = ((ilo > ihi).or.(jlo > jhi))
      oil   = ((ilo < 0) .or. (ilo > rowdim))
      oih   = ((ihi < 0) .or. (ihi > rowdim))
      ojl   = ((jlo < 0) .or. (jlo > coldim))
      ojh   = ((jhi < 0) .or. (jhi > coldim))
      oall = ohilo.or.oil.or.oih.or.ojl.or.ojh
      if (oall) then
        write(6,*)' something wrong with gen_c arguments ',gpmype()
        write(6,*)' ilo = ',ilo, '     ihi = ',ihi
        write(6,*)' jlo = ',jlo, '     jhi = ',jhi
        write(6,*)' rowdim = ',rowdim,'   coldim = ',coldim
        write(6,*)oall,ohilo,oil,oih,ojl,ojh
        stop ' fatal error gen_c'
      endif
      do i = ilo,ihi
        ii = i - 1
        do j = jlo,jhi
          jj = j - 1
          sum = 0.0d00
          do k = 1,crossdim
            kk = k - 1
            sum = sum + (aval*dble(ii)+bval*dble(kk)+cval)*
     &          (dval*dble(kk)+eval*dble(jj)+fval)
          enddo
          C(i,j) = sum
        enddo
      enddo
      end
      subroutine computenorm(norm,A,B,len)
      implicit none
      integer len, i
      double precision norm, diff
      double precision A(len), B(len)
      norm = 0.0d00
      do i = 1,len
        diff = A(i) - B(i)
        norm = norm + diff*diff
      enddo
      end
      subroutine output (z,rowlow,rowhi,collow,colhi,rowdim,coldim,nctl)
!c.......................................................................
!c output prints a real*8 matrix in formatted form with numbered rows
!c and columns.  the input is as follows;
!c        matrix(*,*).........matrix to be output
!c        rowlow..............row number at which output is to begin
!c        rowhi...............row number at which output is to end
!c        collow..............column number at which output is to begin
!c        colhi...............column number at which output is to end
!c        rowdim..............row dimension of matrix(*,*)
!c        coldim..............column dimension of matrix(*,*)
!c        nctl................carriage control flag; 1 for single space
!c                                                   2 for double space
!c                                                   3 for triple space
!c the parameters that follow matrix are all of type integer*4.  the
!c program is set up to handle 5 columns/page with a 1p5d24.15 format for
!c the columns.  if a different number of columns is required, change
!c formats 1000 and 2000, and initialize kcol with the new number of
!c columns.
!c author;  nelson h.f. beebe, quantum theory project, university of
!c          florida, gainesville
!c.......................................................................
!C
      implicit none
      integer rowlow,rowhi,collow,colhi,rowdim,coldim,begin,kcol
      integer nctl, i, j, last, k
      double precision z(rowdim,coldim), zero
      character*8 asa(3), ctl, blank
      data asa/' ','00000000'  , '--------'  /,blank/' '/
      data kcol/8/
      data zero/0.d00/
      do 11 i=rowlow,rowhi
        do 10 j=collow,colhi
          if (z(i,j).ne.zero) go to 15
 10     continue
 11   continue
      write (6,3000)
 3000 format (/' zero matrix'/)
      go to 3
 15   continue
      ctl = blank
      if ((nctl.le.3).and.(nctl.gt.0)) ctl = asa(nctl)
      if (rowhi.lt.rowlow) go to 3
      if (colhi.lt.collow) go to 3
      last = min(colhi,collow+kcol-1)
      do 2 begin = collow,colhi,kcol
        write (6,1000) (i,i = begin,last)
        do 1 k = rowlow,rowhi
          do 4 i=begin,last
            if (z(k,i).ne.zero) go to 5
 4        continue
          go to 1
 5        write (6,2000) ctl,k,(z(k,i), i = begin,last)
 1      continue
        last = min(last+kcol,colhi)
 2    continue
 3    return
 1000 format (8x,7('   ',i3,3x),('   ',i3))
 2000 format (a1,i4,1x,8f9.4)
      end
