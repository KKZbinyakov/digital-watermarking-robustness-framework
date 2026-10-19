cimport numpy as cnp


cdef cnp.ndarray validate_ycbcr_image(object image)


ctypedef double DBlock[8][8]
cdef void init_dct_matrix()
cdef void apply_dct_8x8(const double block_img[8][8], double block_dct[8][8]) noexcept nogil
cdef void apply_idct_8x8(const double block_dct[8][8], double block_img[8][8]) noexcept nogil


cdef void apply_haar_dwt(double[:, ::1] coeffs, double[:, ::1] scratch, int levels) noexcept nogil
cdef void apply_haar_idwt(double[:, ::1] coeffs, double[:, ::1] scratch, int levels) noexcept nogil


ctypedef double dwt_vector[32]

ctypedef bint (*dwt_embed_core_callback)(const double[:, ::1] unit, const double[:, ::1] tangent,
                               const double[:, ::1] stats, const cnp.intp_t[::1] permutation,
                               const double[::1] eta, const cnp.uint8_t[::1] bits,
                               int h, double[:, ::1] targets) noexcept nogil

cdef cnp.ndarray dwt_validate_image(object image, bint embedding)
cdef cnp.ndarray dwt_validate_bits(object watermark)
cdef object dwt_key_material(object key)
cdef void dwt_prepare_core(const double[:, ::1] coeffs, const cnp.intp_t[::1] permutation,
                            const double[:, ::1] reference, double[:, ::1] unit,
                            double[:, ::1] tangent, double[:, ::1] stats,
                            cnp.uint8_t[::1] flags) noexcept nogil
cdef object dwt_render(cnp.ndarray coefficients, const double[:, ::1] targets)
cdef object dwt_calibrate(cnp.ndarray image, cnp.ndarray coefficients,
                           const double[:, ::1] unit, const double[:, ::1] tangent,
                           const double[:, ::1] stats, const cnp.intp_t[::1] permutation,
                           const double[::1] eta, const cnp.uint8_t[::1] bits,
                           double[:, ::1] targets, dwt_embed_core_callback embed_core)

cdef object dwt_integer(object name, object value, object lower, object upper)
cdef double dwt_quantize(double value, double delta, double dither, int bit) noexcept nogil
cdef int dwt_prepare_vector(const double x[32], const double u[32],
                             double unit[32], double tangent[32], double stats[4]) noexcept nogil
cdef bint dwt_embed_vector(const double unit[32], const double tangent[32],
                            const double stats[4], int bit, double delta,
                            double dither, double output[32]) noexcept nogil
cdef int dwt_decode_vector(const double x[32], const double u[32], double delta,
                             double dither, int* erased, int* ambiguous,
                             double* margin) noexcept nogil
cdef void dwt_merge_bands(double[:, ::1] coeffs, const double[:, ::1] targets) noexcept nogil
cdef void dwt_round_pixels(const double[:, ::1] source, double[:, ::1] output) noexcept nogil
cdef void dwt_rounding_cost(const double[:, ::1] source,
                             cnp.uint32_t[:, :, ::1] costs) noexcept nogil
cdef long long dwt_rounding_sse(const double[:, ::1] coeffs,
                                 const double[:, ::1] targets,
                                 const cnp.uint32_t[:, :, ::1] costs,
                                 bint* near_half) noexcept nogil


cdef void contourlet_luma_core(const double[:, :, :] rgb, double[:, ::1] result) noexcept nogil
cdef object contourlet_validate_image(object value)
cdef object contourlet_validate_bits(object value)
cdef double contourlet_positive_float(object name, object value)
cdef object contourlet_integer(object name, object value, object minimum=*)
cdef bint contourlet_boolean(object name, object value)
cdef object contourlet_choice(object name, object value, object choices)
cdef object contourlet_to_luma(object image)
cdef object contourlet_lowpass_analysis(object image, object coefficient_scale=*)
cdef object contourlet_lowpass_synthesis_delta(object coarse_delta, object coefficient_scale=*)
cdef object contourlet_blocks(object image, object block_size)
cdef object contourlet_adaptive_steps(object blocks, object delta0)
cdef object contourlet_embed_core(object lowpass, object bits, object delta0, object block_size, object max_iterations=*)
cdef object contourlet_extract_core(object lowpass, object num_bits, object delta0, object block_size)
cdef object contourlet_arnold(object bits, object iterations, object inverse=*, object matrix=*)
cdef double contourlet_weighted_moment(object weights, object x, object y, object p, object q)
cdef object contourlet_moment_transform(object image, object target_std, object output_shape, object sign_mode=*)
cdef object contourlet_warp_forward(object image, object source_to_normalized, object output_shape)
cdef object contourlet_warp_inverse_delta(object normalized_delta, object source_to_normalized, object output_shape)
cdef object contourlet_full_bounds_transform(object image_shape, object source_to_normalized)
cdef object contourlet_invariant_centroid(object image, object radius, object iterations=*)
cdef object contourlet_roi_origin(object image, object roi_size, object centroid_radius)
cdef object contourlet_centered_roi_origin(object image, object roi_size)
cdef object contourlet_normalized_luma(object image, object normalize, object target_std, object canonical_shape, object bounds_mode=*, object sign_mode=*)
cdef object contourlet_profile_geometry(object roi_size, object block_size, object canonical_shape)

cdef cnp.ndarray dft_validate_image(object image)
cdef cnp.ndarray dft_validate_bits(object watermark)
cdef bytes dft_validate_key(object key)
cdef double dft_validate_strength(object value)
cdef object dft_layout()
cdef object dft_key_material(bytes key)
cdef bint dft_build_mask(const cnp.uint8_t[:, ::1] image, double[:, ::1] mask) noexcept nogil
cdef object dft_fit_weibull(cnp.ndarray magnitudes, cnp.ndarray zones)
cdef double dft_render_pixels(const cnp.uint8_t[:, ::1] image, const double[:, ::1] restored,
                              const double[:, ::1] mask, cnp.uint8_t[:, ::1] result,
                              int* clipped, double scale=*) noexcept nogil
