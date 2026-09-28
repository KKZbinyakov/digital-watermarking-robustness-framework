from libc.stdint cimport uint64_t
cimport numpy as cnp


ctypedef double DBlock[8][8]
cdef void init_dct_matrix()
cdef void apply_dct_8x8(const double block_img[8][8], double block_dct[8][8]) noexcept nogil
cdef void apply_idct_8x8(const double block_dct[8][8], double block_img[8][8]) noexcept nogil
cdef cnp.ndarray validate_rgb_image(object image)
cdef cnp.ndarray rgb_to_ycbcr(const cnp.uint8_t[:, :, ::1] rgb)
cdef cnp.ndarray ycbcr_to_rgb(const double[:, :, ::1] ycbcr)


cdef object dwt_integer(object name, object value, object lower, object upper)
cdef object dwt_parameters(object parameters, bint extraction=*)
cdef object dwt_shape(object height, object width, int levels)
cdef object dwt_rgb(object image)
cdef object dwt_bits(object watermark)
cdef void dwt_check_capacity(object shape, object num_bits, object config) except *
cpdef object dwt_geometry(object image_shape, object levels=*, object geometry_mode=*)
cpdef object dwt_capacity(object image_shape, object levels=*, object repetitions=*, object geometry_mode=*)
cdef object dwt_image_region(object image, object config)
cdef object dwt_luminance(object rgb)
cdef object dwt_merge_region(object image, object luminance, object modified, object geometry)
cdef object dwt_require_decoded_bits(object details)
cdef object dwt_embed_coefficients_inplace(double[:, ::1] coeffs,
    const unsigned char[::1] bits, int levels, int q, Py_ssize_t repetitions,
    uint64_t key_seed, double span_epsilon)
cdef object dwt_decode_coefficients(const double[:, ::1] coeffs, Py_ssize_t num_bits,
    int levels, int q, Py_ssize_t repetitions, uint64_t key_seed,
    double span_epsilon, bint coarsest_only)
cdef void dwt_filters(double* h, double* g) noexcept nogil
cdef object dwt_forward(const double[:, ::1] source, int levels)
cdef object dwt_inverse(const double[:, ::1] source, int levels)
cdef object dwt_positions(Py_ssize_t count, Py_ssize_t needed,
                         uint64_t key_seed, int level,
                         Py_ssize_t height, Py_ssize_t width)
cdef object dwt_tie_bits(Py_ssize_t length, uint64_t key_seed)
cdef void dwt_sort3(double* values, int* indices) noexcept nogil
cdef double dwt_quantize(double a, double b, double c, int bit,
                        int q) noexcept nogil
cdef int dwt_decode(double a, double b, double c, int q) noexcept nogil
cdef object dwt_embed_band(double[:, ::1] coeffs, Py_ssize_t half_h,
                          Py_ssize_t half_w, const Py_ssize_t[::1] positions,
                          const unsigned char[::1] bits, int q,
                          double span_epsilon, int level)
cdef void dwt_extract_band(const double[:, ::1] coeffs, Py_ssize_t half_h,
                          Py_ssize_t half_w, const Py_ssize_t[::1] positions,
                          Py_ssize_t num_bits, int q, double span_epsilon,
                          long long[:, ::1] votes,
                          long long[::1] erased) noexcept nogil


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


cpdef object contourlet_qim_profile(object levels=*, object delta=*, object channel=*, object key=*, object repetitions=*)
cpdef object contourlet_qim_checked_profile(object profile)
cpdef object contourlet_qim_options(object kwargs, object defaults, object payload_name)

cpdef object contourlet_qim_integer(object value, object name, object minimum=*, object maximum=*)
cpdef object contourlet_qim_positive_step(object value)
cpdef object contourlet_qim_real_array(object value, object name)
cpdef object contourlet_qim_binary_array(object value)
cpdef object contourlet_qim_checked_dither(object dither, object delta, object shape)
cpdef object contourlet_qim_nearest_quantize(object value, object delta)
cpdef object contourlet_qim_embed_coefficients(object coefficients, object bits, object delta, object dither=*)
cpdef object contourlet_qim_extract_blocks(object coefficients, object delta, object repetitions=*, object dither=*)
cpdef object contourlet_qim_periodic_filter(object image, object taps)
cpdef object contourlet_qim_lowpass_analysis(object image)
cpdef object contourlet_qim_lowpass_synthesis(object coarse)
cpdef object contourlet_qim_analyse(object image, object levels)
cpdef object contourlet_qim_synthesise_delta(object coarse_delta, object levels)
cpdef object contourlet_qim_carrier_layout(object total, object count, object key, object delta)
cpdef object contourlet_qim_embed_plane_float(object image, object watermark_bits, object profile=*)
cpdef object contourlet_qim_extract_plane_float(object image, object num_bits, object profile=*)
cpdef object contourlet_qim_image_region(object image, object profile)
cpdef object contourlet_qim_capacity(object image, object profile=*)
cpdef object contourlet_qim_round_uint8(object value)
cpdef object contourlet_qim_embed_image(object image, object watermark_bits, object profile=*)
cpdef object contourlet_qim_extract_image(object image, object num_bits, object profile=*)


cpdef cnp.ndarray dft_validate_rgb_uint8(object image)
cpdef cnp.ndarray dft_validate_bits(object bits)
cpdef cnp.ndarray dft_rgb_to_y(object rgb)
cpdef cnp.ndarray dft_apply_luma_delta(object rgb, object old_y, object new_y)
cpdef int dft_binary_parity(int value)
cpdef cnp.ndarray dft_convolutional_encode(object bits)
cpdef cnp.ndarray dft_convolutional_decode_soft_batch(object values, int message_length)
cpdef cnp.ndarray dft_ecc_encode(object bits, str mode)
cpdef cnp.ndarray dft_ecc_decode_soft_batch(object values, int message_length, str mode)
cpdef cnp.ndarray dft_coded_antipodal_from_messages(object messages, str ecc_mode)
cpdef cnp.ndarray dft_interleaver_indices(int length, int seed)
cpdef cnp.ndarray dft_interleave_coded_bits(object bits, int seed)
cpdef cnp.ndarray dft_deinterleave_soft_matrix(object values, int seed)
cpdef cnp.ndarray dft_manchester_encode(object bits)
cpdef object dft_forward_lpm_geometry(int height, int width, int rho_samples, int theta_samples,
                                      double omega_min, double omega_max)
cpdef cnp.ndarray dft_forward_lpm(object magnitude, int rho_samples, int theta_samples,
                                  double omega_min, double omega_max)
cpdef cnp.ndarray dft_bilinear_periodic(object arr, object rho_f, object theta_f)
cpdef cnp.ndarray dft_inverse_lpm_backward(object lpm, object out_shape,
                                           double omega_min, double omega_max)
cpdef object dft_centered_fft_with_optional_padding(object y, int pad_factor)
cpdef cnp.ndarray dft_centered_magnitude_with_optional_padding(object y, int pad_factor)
cpdef cnp.ndarray dft_crop_center(object arr, object shape)
cpdef object dft_rt_invariant_from_image(object y, int rho_samples, int theta_samples,
                                         double omega_min, double omega_max, int pad_factor)
cpdef cnp.ndarray dft_rt_magnitude_from_image(object y, int rho_samples, int theta_samples,
                                                double omega_min, double omega_max, int pad_factor)
cpdef int dft_rho_index_for_omega(double omega, int rho_samples,
                                  double omega_min, double omega_max)
cpdef cnp.ndarray dft_scale_delta_to_psnr(object base_y, object delta, object target_psnr)
