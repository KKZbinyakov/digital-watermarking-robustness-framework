cimport numpy as cnp

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


cpdef int validate_cell_capacity(object cell_map, int n_cells, str stage)
cdef cnp.ndarray build_cell_map(int H, int W, int n_rho, int n_theta, double r1, double r2)
cdef void accumulate(double complex[:, ::1] F, int[:, ::1] cell, double[::1] sum_log, cnp.int64_t[::1] count) noexcept nogil
cdef void apply_gain(double complex[:, ::1] F, int[:, ::1] cell, double[::1] gain) noexcept nogil
cdef cnp.ndarray cell_mean_sync(double complex[:, ::1] F, int[:, ::1] cell, int n_rho, int n_theta)


cdef enum:
    N_SVD = 4
    NN_SVD = 16
    MAX_SWEEPS = 30
    MAX_POWER_ITER = 50

cdef void svd_jacobi(double *A, double *V, double *sv, bint want_v) noexcept nogil
cdef void top2(double *sv, int *idx1, double *s1, double *s2) noexcept nogil
cdef double qim_embed(double sigma, int bit, double delta) noexcept nogil
cdef int qim_extract(double sigma, double delta) noexcept nogil
cdef void power_iteration(double *A, double *u, double *v, double *sigma) noexcept nogil
cdef double power_iteration_sigma(double *A) noexcept nogil
cdef void embed_block(double[:, ::1] img, int by, int bx, int bit, double delta) noexcept nogil
cdef int extract_block(double[:, ::1] img, int by, int bx, double delta) noexcept nogil


cdef int capacity(int h, int w, int block, int n_sub) noexcept nogil
cdef void embed_subband(double[:, :] S, cnp.int32_t[:] wm, int wm_len,
                        int sub_i, int n_sub, double margin,
                        int block) noexcept nogil
cdef void extract_subband(double[:, :] S, cnp.int32_t[:] wm, int wm_len,
                          int sub_i, int n_sub, int block) noexcept nogil
cdef int count_subband_bit_errors(object subbands, cnp.int32_t[:] wm,
                                  int wm_len, int block)
cpdef object valid_contourlet_shape(int height, int width,
                                     int n_levels, int dfb_levels)
cdef void init_filters() noexcept nogil
cdef void init_offsets() noexcept nogil
cpdef contourlet_decompose(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image, int n_levels, int dfb_levels)
cpdef contourlet_reconstruct(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] lowpass, list bands, int dfb_levels)