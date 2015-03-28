c-----------------------------------------------------------------------
c     file spline.f
c     fits functions to cubic splines.
c     Reference: H. Spaeth, "Spline Algorithms for Curves and Surfaces,"
c     Translated from the German by W. D. Hoskins and H. W. Sager.
c     Utilitas Mathematica Publishing Inc., Winnepeg, 1974.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     code organization.
c-----------------------------------------------------------------------
c     0. spline_type definition.
c     1. spline_alloc.
c     2. spline_dealloc.
c     3. spline_fit.
c     4. spline_fac.
c     5. spline_eval.
c     6. spline_all_eval.
c     7. spline_write.
c     8. spline_int.
c     9. spline_triluf.
c     10. spline_trilus.
c     11. spline_sherman.
c     12. spline_morrison.
c     13. spline_copy.
c-----------------------------------------------------------------------
c     subprogram 0. spline_type definition.
c     defines spline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      MODULE spline_mod
      USE iso_c_binding, only: c_double, c_int
      USE local_mod
      IMPLICIT NONE

      TYPE :: spline_type
      INTEGER :: mx,nqty,ix
      REAL(r8), DIMENSION(:), POINTER :: xs,f,f1,f2,f3
      REAL(r8), DIMENSION(:,:), POINTER :: fs,fs1,fsi
      CHARACTER(6), DIMENSION(:), POINTER :: title
      CHARACTER(6) :: name
      LOGICAL :: periodic
      END TYPE spline_type

      CONTAINS
c-----------------------------------------------------------------------
c     subprogram 1. spline_alloc.
c     allocates space for spline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_alloc(spl,mx,nqty)

      INTEGER, INTENT(IN) :: mx,nqty
      TYPE(spline_type), INTENT(INOUT) :: spl
c-----------------------------------------------------------------------
c     set scalars.
c-----------------------------------------------------------------------
      spl%mx=mx
      spl%nqty=nqty
      spl%ix=0
      spl%periodic=.FALSE.
c-----------------------------------------------------------------------
c     allocate space.
c-----------------------------------------------------------------------
      ALLOCATE(spl%xs(0:mx))
      ALLOCATE(spl%f(nqty))
      ALLOCATE(spl%f1(nqty))
      ALLOCATE(spl%f2(nqty))
      ALLOCATE(spl%f3(nqty))
      ALLOCATE(spl%title(0:nqty))
      ALLOCATE(spl%fs(0:mx,nqty))
      ALLOCATE(spl%fs1(0:mx,nqty))
      NULLIFY(spl%fsi)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_alloc
c-----------------------------------------------------------------------
c     subprogram 2. spline_dealloc.
c     deallocates space for spline_type.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_dealloc(spl)

      TYPE(spline_type), INTENT(INOUT) :: spl
c-----------------------------------------------------------------------
c     deallocate space.
c-----------------------------------------------------------------------
      DEALLOCATE(spl%xs)
      DEALLOCATE(spl%f)
      DEALLOCATE(spl%f1)
      DEALLOCATE(spl%f2)
      DEALLOCATE(spl%f3)
      DEALLOCATE(spl%title)
      DEALLOCATE(spl%fs)
      DEALLOCATE(spl%fs1)
      IF(ASSOCIATED(spl%fsi))DEALLOCATE(spl%fsi)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_dealloc
c-----------------------------------------------------------------------
c     subprogram 3. spline_fit.
c     fits real functions to cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_fit(spl,endmode)

      TYPE(spline_type), INTENT(INOUT) :: spl
      CHARACTER(*), INTENT(IN) :: endmode

      INTEGER :: iqty
      REAL(r8), DIMENSION(-1:1,0:spl%mx) :: a
      REAL(r8), DIMENSION(spl%mx) :: b
      REAL(r8), DIMENSION(4) :: cl,cr
c-----------------------------------------------------------------------
c     set periodicity
c-----------------------------------------------------------------------
      spl%periodic=endmode == "periodic"
      IF(spl%periodic)spl%fs(spl%mx,:)=spl%fs(0,:)
c-----------------------------------------------------------------------
c     set up grid matrix.
c-----------------------------------------------------------------------
      CALL spline_fac(spl,a,b,cl,cr,endmode)
c-----------------------------------------------------------------------
c     compute first derivatives, interior.
c-----------------------------------------------------------------------
      DO iqty=1,spl%nqty
         spl%fs1(1:spl%mx-1,iqty)=
     $        3*((spl%fs(2:spl%mx,iqty)-spl%fs(1:spl%mx-1,iqty))
     $        *b(2:spl%mx)
     $        +(spl%fs(1:spl%mx-1,iqty)-spl%fs(0:spl%mx-2,iqty))
     $        *b(1:spl%mx-1))
      ENDDO
c-----------------------------------------------------------------------
c     extrapolation boundary conditions.
c-----------------------------------------------------------------------
      IF(endmode == "extrap")THEN
         DO iqty=1,spl%nqty
            spl%fs1(0,iqty)=SUM(cl(1:4)*spl%fs(0:3,iqty))
            spl%fs1(spl%mx,iqty)=SUM(cr(1:4)
     $           *spl%fs(spl%mx:spl%mx-3:-1,iqty))
            spl%fs1(1,iqty)=spl%fs1(1,iqty)-spl%fs1(0,iqty)
     $           /(spl%xs(1)-spl%xs(0))
            spl%fs1(spl%mx-1,iqty)=
     $           spl%fs1(spl%mx-1,iqty)-spl%fs1(spl%mx,iqty)
     $           /(spl%xs(spl%mx)-spl%xs(spl%mx-1))
            CALL spline_trilus(a(:,1:),spl%fs1(1:,iqty),
     $           spl%mx-1,1)
         ENDDO
c-----------------------------------------------------------------------
c     not-a-knot boundary conditions.
c-----------------------------------------------------------------------
      ELSE IF(endmode == "not-a-knot")THEN
         spl%fs1(1,:)=spl%fs1(1,:)-(2*spl%fs(1,:)
     $        -spl%fs(0,:)-spl%fs(2,:))*2*b(1)
         spl%fs1(spl%mx-1,:)=spl%fs1(spl%mx-1,:)
     $        +(2*spl%fs(spl%mx-1,:)-spl%fs(spl%mx,:)
     $        -spl%fs(spl%mx-2,:))*2*b(spl%mx)
         DO iqty=1,spl%nqty
            CALL spline_trilus(a(:,1:),spl%fs1(1:,iqty),
     $           spl%mx-1,1)
         ENDDO
         spl%fs1(0,:)=(2*(2*spl%fs(1,:)-spl%fs(0,:)-spl%fs(2,:))
     $        +(spl%fs1(1,:)+spl%fs1(2,:))*(spl%xs(2)-spl%xs(1))
     $        -spl%fs1(1,:)*(spl%xs(1)-spl%xs(0)))/(spl%xs(1)-spl%xs(0))
         spl%fs1(spl%mx,:)=
     $        (2*(spl%fs(spl%mx-2,:)+spl%fs(spl%mx,:)
     $        -2*spl%fs(spl%mx-1,:))
     $        +(spl%fs1(spl%mx-1,:)+spl%fs1(spl%mx-2,:))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))
     $        -spl%fs1(spl%mx-1,:)
     $        *(spl%xs(spl%mx)-spl%xs(spl%mx-1)))
     $        /(spl%xs(spl%mx)-spl%xs(spl%mx-1))
c-----------------------------------------------------------------------
c     periodic boudary conditions.
c-----------------------------------------------------------------------
      ELSE IF(endmode == "periodic")THEN
         spl%fs1(0,:)=3*((spl%fs(1,:)-spl%fs(0,:))*b(1)
     $        +(spl%fs(0,:)-spl%fs(spl%mx-1,:))*b(spl%mx))
         DO iqty=1,spl%nqty
            CALL spline_morrison(a,spl%fs1(:,iqty),spl%mx,1)
         ENDDO
         spl%fs1(spl%mx,:)=spl%fs1(0,:)
      ENDIF
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_fit
c-----------------------------------------------------------------------
c     subprogram 4. spline_fac.
c     sets up matrix for cubic spline fitting.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_fac(spl,a,b,cl,cr,endmode)

      TYPE(spline_type), INTENT(IN) :: spl
      REAL(r8), DIMENSION(-1:1,0:spl%mx), INTENT(OUT) :: a
      REAL(r8), DIMENSION(spl%mx), INTENT(OUT) :: b
      REAL(r8), DIMENSION(4), INTENT(OUT) :: cl,cr
      CHARACTER(*), INTENT(IN) :: endmode

      INTEGER :: j
c-----------------------------------------------------------------------
c     compute interior matrix.
c-----------------------------------------------------------------------
      b=1/(spl%xs(1:spl%mx)-spl%xs(0:spl%mx-1))
      DO j=1,spl%mx-1
         a(-1,j)=b(j)
         a(0,j)=2*(b(j)+b(j+1))
         a(1,j)=b(j+1)
      ENDDO
c-----------------------------------------------------------------------
c     extrapolation boundary conditions.
c-----------------------------------------------------------------------
      IF(endmode == "extrap")THEN
         b=b*b
         cl(1)=(spl%xs(0)*(3*spl%xs(0)
     $        -2*(spl%xs(1)+spl%xs(2)+spl%xs(3)))
     $        +spl%xs(1)*spl%xs(2)+spl%xs(1)*spl%xs(3)
     $        +spl%xs(2)*spl%xs(3))
     $        /((spl%xs(0)-spl%xs(1))*(spl%xs(0)-spl%xs(2))
     $        *(spl%xs(0)-spl%xs(3)))
         cl(2)=((spl%xs(2)-spl%xs(0))*(spl%xs(3)-spl%xs(0)))
     $        /((spl%xs(1)-spl%xs(0))*(spl%xs(1)-spl%xs(2))
     $        *(spl%xs(1)-spl%xs(3)))
         cl(3)=((spl%xs(0)-spl%xs(1))*(spl%xs(3)-spl%xs(0)))
     $        /((spl%xs(0)-spl%xs(2))*(spl%xs(1)-spl%xs(2))
     $        *(spl%xs(3)-spl%xs(2)))
         cl(4)=((spl%xs(1)-spl%xs(0))*(spl%xs(2)-spl%xs(0)))
     $        /((spl%xs(3)-spl%xs(0))*(spl%xs(3)-spl%xs(1))
     $        *(spl%xs(3)-spl%xs(2)))
         cr(1)=(spl%xs(spl%mx)*(3*spl%xs(spl%mx)
     $        -2*(spl%xs(spl%mx-1)+spl%xs(spl%mx-2)
     $        +spl%xs(spl%mx-3)))+spl%xs(spl%mx-1)
     $        *spl%xs(spl%mx-2)+spl%xs(spl%mx-1)*spl%xs(spl%mx
     $        -3)+spl%xs(spl%mx-2)*spl%xs(spl%mx-3))
     $        /((spl%xs(spl%mx)-spl%xs(spl%mx-1))
     $        *(spl%xs(spl%mx)-spl%xs(spl%mx-2))*(spl%xs(spl%mx
     $        )-spl%xs(spl%mx-3)))
         cr(2)=((spl%xs(spl%mx-2)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx)))
     $        /((spl%xs(spl%mx-1)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-3)))
         cr(3)=((spl%xs(spl%mx)-spl%xs(spl%mx-1))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx)))
     $        /((spl%xs(spl%mx)-spl%xs(spl%mx-2))
     $        *(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx-2)))
         cr(4)=((spl%xs(spl%mx-1)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-2)-spl%xs(spl%mx)))
     $        /((spl%xs(spl%mx-3)-spl%xs(spl%mx))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx-1))
     $        *(spl%xs(spl%mx-3)-spl%xs(spl%mx-2)))
         CALL spline_triluf(a(:,1:),spl%mx-1)
c-----------------------------------------------------------------------
c     not-a-knot boundary conditions.
c-----------------------------------------------------------------------
      ELSE IF(endmode == "not-a-knot")THEN
         b=b*b
         a(0,1)=a(0,1)+(spl%xs(2)+spl%xs(0)-2*spl%xs(1))*b(1)
         a(1,1)=a(1,1)+(spl%xs(2)-spl%xs(1))*b(1)
         a(0,spl%mx-1)=a(0,spl%mx-1)
     $        +(2*spl%xs(spl%mx-1)-spl%xs(spl%mx-2)
     $        -spl%xs(spl%mx))*b(spl%mx)
         a(-1,spl%mx-1)=a(-1,spl%mx-1)
     $        +(spl%xs(spl%mx-1)-spl%xs(spl%mx-2))*b(spl%mx)
         CALL spline_triluf(a(:,1:),spl%mx-1)
c-----------------------------------------------------------------------
c     periodic boundary conditions.
c-----------------------------------------------------------------------
      ELSE IF(endmode == "periodic")THEN
         a(0,0:spl%mx:spl%mx)=2*(b(spl%mx)+b(1))
         a(1,0)=b(1)
         a(-1,0)=b(spl%mx)
         b=b*b
         CALL spline_sherman(a,spl%mx)
      ENDIF
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_fac
c-----------------------------------------------------------------------
c     subprogram 5. spline_eval.
c     evaluates real cubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_eval(spl,x,mode)

      TYPE(spline_type), INTENT(INOUT) :: spl
      REAL(r8), INTENT(IN) :: x
      INTEGER, INTENT(IN) :: mode

      INTEGER :: i
      REAL(r8) :: xx,d,z,z1
c-----------------------------------------------------------------------
c     preliminary computations.
c-----------------------------------------------------------------------
      xx=x
      spl%ix=MAX(spl%ix,0)
      spl%ix=MIN(spl%ix,spl%mx-1)
c-----------------------------------------------------------------------
c     normalize interval for periodic splines.
c-----------------------------------------------------------------------
      IF(spl%periodic)THEN
         DO
            IF(xx < spl%xs(spl%mx))EXIT
            xx=xx-spl%xs(spl%mx)
         ENDDO
         DO
            IF(xx >= spl%xs(0))EXIT
            xx=xx+spl%xs(spl%mx)
         ENDDO
      ENDIF
c-----------------------------------------------------------------------
c     find cubic spline interval.
c-----------------------------------------------------------------------
      DO
         IF(xx >= spl%xs(spl%ix).OR.spl%ix <= 0)EXIT
         spl%ix=spl%ix-1
      ENDDO
      DO
         IF(xx < spl%xs(spl%ix+1).OR.spl%ix >= spl%mx-1)EXIT
         spl%ix=spl%ix+1
      ENDDO
c-----------------------------------------------------------------------
c     evaluate ofspl%fset and related quantities.
c-----------------------------------------------------------------------
      d=spl%xs(spl%ix+1)-spl%xs(spl%ix)
      z=(xx-spl%xs(spl%ix))/d
      z1=1-z
c-----------------------------------------------------------------------
c     evaluate output values.
c-----------------------------------------------------------------------
      spl%f(1:spl%nqty)=spl%fs(spl%ix,1:spl%nqty)*z1*z1*(3-2*z1)
     $     +spl%fs(spl%ix+1,1:spl%nqty)*z*z*(3-2*z)
     $     +d*z*z1*(spl%fs1(spl%ix,1:spl%nqty)*z1
     $     -spl%fs1(spl%ix+1,1:spl%nqty)*z)
      IF(mode == 0)RETURN
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      spl%f1(1:spl%nqty)=6*(spl%fs(spl%ix+1,1:spl%nqty)
     $     -spl%fs(spl%ix,1:spl%nqty))*z*z1/d
     $     +spl%fs1(spl%ix,1:spl%nqty)*z1*(3*z1-2)
     $     +spl%fs1(spl%ix+1,1:spl%nqty)*z*(3*z-2)
      IF(mode == 1)RETURN
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      spl%f2(1:spl%nqty)=(6*(spl%fs(spl%ix+1,1:spl%nqty)
     $     -spl%fs(spl%ix,1:spl%nqty))*(z1-z)/d
     $     -spl%fs1(spl%ix,1:spl%nqty)*(6*z1-2)
     $     +spl%fs1(spl%ix+1,1:spl%nqty)*(6*z-2))/d
      IF(mode == 2)RETURN
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      spl%f3(1:spl%nqty)=(12*(spl%fs(spl%ix,1:spl%nqty)
     $     -spl%fs(spl%ix+1,1:spl%nqty))/d
     $     +6*(spl%fs1(spl%ix,1:spl%nqty)
     $     +spl%fs1(spl%ix+1,1:spl%nqty)))/(d*d)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_eval
c-----------------------------------------------------------------------
c     subprogram 6. spline_all_eval.
c     evaluates cubic spline function.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_all_eval(spl,z,f,f1,f2,f3,mode)

      TYPE(spline_type), INTENT(INOUT) :: spl
      REAL(r8), INTENT(IN) :: z
      REAL(r8), DIMENSION(spl%mx,spl%nqty), INTENT(OUT) :: f,f1,f2,f3
      INTEGER, INTENT(IN) :: mode

      INTEGER :: iqty,nqty,n
      REAL(r8) :: z1
      REAL(r8), DIMENSION(spl%mx) :: d
c-----------------------------------------------------------------------
c     evaluate offset and related quantities.
c-----------------------------------------------------------------------
      n=spl%mx
      nqty=spl%nqty
      z1=1-z
      d=spl%xs(1:n)-spl%xs(0:n-1)
c-----------------------------------------------------------------------
c     evaluate output values.
c-----------------------------------------------------------------------
      DO iqty=1,nqty
         f(:,iqty)=spl%fs(0:n-1,iqty)*z1*z1*(3-2*z1)
     $        +spl%fs(1:n,iqty)*z*z*(3-2*z)
     $        +d*z*z1*(spl%fs1(0:n-1,iqty)*z1-spl%fs1(1:n,iqty)*z)
      ENDDO
      IF(mode == 0)RETURN
c-----------------------------------------------------------------------
c     evaluate first derivatives.
c-----------------------------------------------------------------------
      DO iqty=1,nqty
         f1(:,iqty)=6*(spl%fs(1:n,iqty)-spl%fs(0:n-1,iqty))*z*z1/d
     $        +spl%fs1(0:n-1,iqty)*z1*(3*z1-2)
     $        +spl%fs1(1:n,iqty)*z*(3*z-2)
      ENDDO
      IF(mode == 1)RETURN
c-----------------------------------------------------------------------
c     evaluate second derivatives.
c-----------------------------------------------------------------------
      DO iqty=1,nqty
         f2(:,iqty)=(6*(spl%fs(1:n,iqty)-spl%fs(0:n-1,iqty))*(z1-z)/d
     $        -spl%fs1(0:n-1,iqty)*(6*z1-2)
     $        +spl%fs1(1:n,iqty)*(6*z-2))/d
      ENDDO
      IF(mode == 2)RETURN
c-----------------------------------------------------------------------
c     evaluate third derivatives.
c-----------------------------------------------------------------------
      DO iqty=1,nqty
         f3(:,iqty)=(12*(spl%fs(0:n-1,iqty)-spl%fs(1:n,iqty))/d
     $        +6*(spl%fs1(0:n-1,iqty)+spl%fs1(1:n,iqty)))/(d*d)
      ENDDO
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_all_eval
c-----------------------------------------------------------------------
c     subprogram 7. spline_write.
c     produces ascii and binary output for real cubic spline fits.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_write(spl,out,bin,iua,iub,interp)

      TYPE(spline_type), INTENT(INOUT) :: spl
      LOGICAL, INTENT(IN) :: out,bin
      INTEGER, INTENT(IN) :: iua,iub
      LOGICAL, INTENT(IN) :: interp

      CHARACTER(30) :: format1,format2
      INTEGER :: iqty,i,j,k
      REAL(r8) :: x,dx
c-----------------------------------------------------------------------
c     formats.
c-----------------------------------------------------------------------
 10   FORMAT('(/4x,''i'',',i2.2,'(4x,a6,1x)/)')
 20   FORMAT('(i5,1p,',i2.2,'e11.3)')
 30   FORMAT('(/4x,''i'',2x,''j'',',i2.2,'(4x,a6,1x)/)')
c-----------------------------------------------------------------------
c     print ascii tables of node values and derivatives.
c-----------------------------------------------------------------------
      IF(.NOT.out.AND..NOT.bin)RETURN
      IF(out)THEN
         WRITE(format1,10)spl%nqty+1
         WRITE(format2,20)spl%nqty+1
         WRITE(iua,'(/1x,a)')'input values:'
         WRITE(iua,format1)spl%title(0:spl%nqty)
         WRITE(iua,format2)(i,spl%xs(i),spl%fs(i,1:spl%nqty),
     $        i=0,spl%mx)
         WRITE(iua,format1)spl%title(0:spl%nqty)
         WRITE(iua,'(/1x,a)')'derivatives:'
         WRITE(iua,format1)spl%title(0:spl%nqty)
         WRITE(iua,format2)(i,spl%xs(i),spl%fs1(i,1:spl%nqty),
     $        i=0,spl%mx)
         WRITE(iua,format1)spl%title(0:spl%nqty)
      ENDIF
c-----------------------------------------------------------------------
c     print binary table of node values.
c-----------------------------------------------------------------------
      IF(bin)THEN
         DO i=0,spl%mx
            WRITE(iub)(/REAL(spl%xs(i),4),REAL(spl%fs(i,1:spl%nqty),4)/)
         ENDDO
         WRITE(iub)
      ENDIF
c-----------------------------------------------------------------------
c     print header for interpolated values.
c-----------------------------------------------------------------------
      IF(.NOT. interp)RETURN
      IF(out)THEN
         WRITE(iua,'(/1x,a)')'interpolated values:'
         WRITE(iua,format1)spl%title(0:spl%nqty)
      ENDIF
c-----------------------------------------------------------------------
c     print interpolated values.
c-----------------------------------------------------------------------
      DO i=0,spl%mx-1
         dx=(spl%xs(i+1)-spl%xs(i))/4
         DO j=0,4
            x=spl%xs(i)+j*dx
            CALL spline_eval(spl,x,0)
            IF(out)WRITE(iua,format2)i,x,spl%f(1:spl%nqty)
            IF(bin)WRITE(iub)(/REAL(x,4),REAL(spl%f(1:spl%nqty),4)/)
         ENDDO
         IF(out)WRITE(iua,'(1x)')
      ENDDO
c-----------------------------------------------------------------------
c     print final interpolated values.
c-----------------------------------------------------------------------
      x=spl%xs(spl%mx)
      CALL spline_eval(spl,x,0)
      IF(out)THEN
         WRITE(iua,format2)i,x,spl%f(1:spl%nqty)
         WRITE(iua,format1)spl%title(0:spl%nqty)
      ENDIF
      IF(bin)THEN
         WRITE(iub)(/REAL(x,4),REAL(spl%f(1:spl%nqty),4)/)
         WRITE(iub)
      ENDIF
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_write
c-----------------------------------------------------------------------
c     subprogram 8. spline_int.
c     integrates real cubic splines.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_int(spl)

      TYPE(spline_type), INTENT(INOUT) :: spl

      INTEGER :: i,iqty,stat
      REAL(r8) :: d
c-----------------------------------------------------------------------
c     integrate cubic splines.
c-----------------------------------------------------------------------
      IF(.NOT.ASSOCIATED(spl%fsi))
     $     ALLOCATE(spl%fsi(0:spl%mx,spl%nqty))
      DO iqty=1,spl%nqty
         spl%fsi(0,iqty)=0
         DO i=1,spl%mx
            d=spl%xs(i)-spl%xs(i-1)
            spl%fsi(i,iqty)=spl%fsi(i-1,iqty)
     $           +d/12*(6*(spl%fs(i-1,iqty)+spl%fs(i,iqty))
     $           +d*(spl%fs1(i-1,iqty)-spl%fs1(i,iqty)))
         ENDDO
      ENDDO
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_int
c-----------------------------------------------------------------------
c     subprogram 9. spline_triluf.
c     performs tridiagonal LU factorization.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_triluf(a,n)

      INTEGER, INTENT(IN) :: n
      REAL(r8), DIMENSION(-1:1,n), INTENT(INOUT) :: a

      INTEGER :: i,j,k,jmin,jmax
c-----------------------------------------------------------------------
c     begin loop over rows and define limits.
c-----------------------------------------------------------------------
      DO i=1,n
         jmin=MAX(1-i,-1)
         jmax=MIN(n-i,1)
c-----------------------------------------------------------------------
c     compute lower elements.
c-----------------------------------------------------------------------
         DO j=jmin,-1
            DO k=MAX(jmin,j-1),j-1
               a(j,i)=a(j,i)-a(k,i)*a(j-k,i+k)
            ENDDO
            a(j,i)=a(j,i)*a(0,i+j)
         ENDDO
c-----------------------------------------------------------------------
c     compute diagonal element
c-----------------------------------------------------------------------
         DO k=MAX(jmin,-1),-1
            a(0,i)=a(0,i)-a(k,i)*a(-k,i+k)
         ENDDO
         a(0,i)=1/a(0,i)
      ENDDO
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_triluf
c-----------------------------------------------------------------------
c     subprogram 10. spline_trilus.
c     performs tridiagonal LU solution.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_trilus(a,x,n,ns)

      INTEGER, INTENT(IN) :: n,ns
      REAL(r8), DIMENSION(-1:1,n), INTENT(IN) :: a
      REAL(r8), DIMENSION(n,ns), INTENT(INOUT) :: x

      INTEGER :: i,j,is
c-----------------------------------------------------------------------
c     down sweep.
c-----------------------------------------------------------------------
      DO is=1,ns
         DO i=1,n
            DO j=MAX(1-i,-1),-1
               x(i,is)=x(i,is)-a(j,i)*x(i+j,is)
            ENDDO
         ENDDO
c-----------------------------------------------------------------------
c     up sweep.
c-----------------------------------------------------------------------
         DO i=n,1,-1
            DO j=1,MIN(n-i,1)
               x(i,is)=x(i,is)-a(j,i)*x(i+j,is)
            ENDDO
            x(i,is)=x(i,is)*a(0,i)
         ENDDO
      ENDDO
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_trilus
c-----------------------------------------------------------------------
c     subprogram 11. spline_sherman.
c     uses Sherman-Morrison formula to factor periodic matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_sherman(a,n)

      INTEGER :: n
      REAL(r8), DIMENSION(-1:1,n), INTENT(INOUT) :: a

      INTEGER :: j
      REAL(r8), DIMENSION(n,1) :: u
c-----------------------------------------------------------------------
c     prepare matrices.
c-----------------------------------------------------------------------
      a(0,1)=a(0,1)-a(-1,1)
      a(0,n)=a(0,n)-a(-1,1)
      u=RESHAPE((/one,(zero,j=2,n-1),one/),SHAPE(u))
      CALL spline_triluf(a,n)
      CALL spline_trilus(a,u,n,1)
      a(-1,1)=a(-1,1)/(1+a(-1,1)*(u(1,1)+u(n,1)))
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_sherman
c-----------------------------------------------------------------------
c     subprogram 12. spline_morrison.
c     uses Sherman-Morrison formula to solve periodic matrix.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_morrison(a,x,n,ns)

      INTEGER :: n,ns
      REAL(r8), DIMENSION(-1:1,n), INTENT(IN) :: a
      REAL(r8), DIMENSION(n,ns), INTENT(INOUT) :: x

      REAL(r8), DIMENSION(n,ns) :: y
c-----------------------------------------------------------------------
c     solve for x.
c-----------------------------------------------------------------------
      y=x
      CALL spline_trilus(a,y,n,ns)
      x(1,:)=x(1,:)-a(-1,1)*(y(1,:)+y(n,:))
      x(n,:)=x(n,:)-a(-1,1)*(y(1,:)+y(n,:))
      CALL spline_trilus(a,x,n,ns)
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_morrison
c-----------------------------------------------------------------------
c     subprogram 13. spline_copy.
c     copies one spline_type to another.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     declarations.
c-----------------------------------------------------------------------
      SUBROUTINE spline_copy(spl1,spl2)

      TYPE(spline_type), INTENT(IN) :: spl1
      TYPE(spline_type), INTENT(INOUT) :: spl2
c-----------------------------------------------------------------------
c     computations.
c-----------------------------------------------------------------------
      IF(ASSOCIATED(spl2%xs))CALL spline_dealloc(spl2)
      CALL spline_alloc(spl2,spl1%mx,spl1%nqty)
      spl2%xs=spl1%xs
      spl2%fs=spl1%fs
      spl2%fs1=spl1%fs1
      spl2%name=spl1%name
      spl2%title=spl1%title
      spl2%periodic=spl1%periodic
c-----------------------------------------------------------------------
c     terminate.
c-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE spline_copy
      END MODULE spline_mod
