ctypedef double DBlock[8][8]
cdef void init_dct_matrix()
cdef void apply_dct_8x8(const double block_img[8][8], double block_dct[8][8])
cdef void apply_idct_8x8(const double block_dct[8][8], double block_idct[8][8])

cdef void get_wavelet_filters(char* name, double* h, double* g, int* L)
cdef inline void dwt_1d_haar(const double* x, double* a, double* d, int n)
cdef inline void idwt_1d_haar(const double* a, const double* d, double* x, int n)
cdef inline void dwt_1d_l8(const double* x, double* a, double* d, int n, const double* h, const double* g)
cdef inline void idwt_1d_l8(const double* a, const double* d, double* x, int n, const double* h, const double* g)
cdef void dwt_2d_block(double[:, :] block,
                       double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH, 
                       double* temp, double* row_a, double* row_d,
                       double* col_in, double* col_a, double* col_d,
                       const double* h, const double* g, int L, int block_size)
cdef void idwt_2d_block(double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                        double[:, :] block, 
                        double* temp, double* col_a, double* col_d, double* col_out, 
                        double* row_a, double* row_d, double* row_out,
                        const double* h, const double* g, int L, int block_size)