ctypedef float s

cdef extern void _slaed9 "slaed9_" (int *k, int *kstart, int *kstop, int *n, s *d, s *q, int *ldq, s *rho, s *dlamda, s *w, s *s, int *lds, int *info) nogil