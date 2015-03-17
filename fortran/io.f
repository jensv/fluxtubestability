c-----------------------------------------------------------------------
c     file io.f.
c     input and output unit declarations.
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     module declarations.
c-----------------------------------------------------------------------
      MODULE io_mod
      USE iso_c_binding, only: c_double, c_int
      IMPLICIT NONE

      INTEGER(c_int) :: in_unit=1
      INTEGER(c_int) :: out_unit=2
      INTEGER(c_int) :: bin_unit=3
      INTEGER(c_int) :: term_unit=6
      INTEGER(c_int) :: sum_unit=10
      INTEGER(c_int) :: debug_unit=99

      END MODULE io_mod
