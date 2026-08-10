import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from dwarf.ready_solutions.embedding_solutions.frequency.dwt_svd_dct import DWTSVD_DCT


def extract_ber(img_bgr, original_wm, **kwargs):
    """Извлекает ЦВЗ из BGR-изображения и считает BER."""
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)[:, :, 0]
    y = np.ascontiguousarray(y, dtype=np.float64)
    extracted = DWTSVD_DCT.extraction(
        input_image=y,
        num_bits=original_wm.shape[0],
        **kwargs
    )
    errors = int(np.sum(original_wm != extracted))
    return errors / original_wm.shape[0], errors


def main():
    image_path = '1.jpg'
    img_bgr = cv2.imread(image_path)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    Y_channel = img_ycbcr[:, :, 0]

    block_size = 16
    H, W = Y_channel.shape
    H_crop = (H // block_size) * block_size
    W_crop = (W // block_size) * block_size
    Y_crop = Y_channel[:H_crop, :W_crop]

    wm_length = 100
    np.random.seed(42)
    original_wm = np.random.randint(0, 2, size=wm_length, dtype=np.int32)

    print(f"Длина водяного знака: {wm_length} бит")

    wavelet_name = "haar"
    delta = 40.0
    margin = 150.0
    threshold = 25.0

    Y_crop_contiguous = np.ascontiguousarray(Y_crop, dtype=np.float64)
    original_wm_contiguous = np.ascontiguousarray(original_wm, dtype=np.int32)

    print(f"\n[Встраивание] DWT-SVD-DCT, вейвлет: {wavelet_name}, block_size: {block_size}, "
          f"delta: {delta}, margin: {margin}, threshold: {threshold}")

    watermarked_Y = DWTSVD_DCT.embedding(
        input_image=Y_crop_contiguous,
        watermark_bits=original_wm_contiguous,
        block_size=block_size,
        delta=delta,
        margin=margin,
        threshold=threshold,
        wavelet_name=wavelet_name
    )

    watermarked_Y_clipped = np.clip(watermarked_Y, 0, 255)
    img_ycbcr[:H_crop, :W_crop, 0] = watermarked_Y_clipped
    watermarked_bgr = cv2.cvtColor(img_ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('watermarked_1.jpg', watermarked_bgr)
    print(f"Сохранено цветное изображение с водяным знаком: watermarked_1.jpg")

    watermarked_Y_contiguous = np.ascontiguousarray(watermarked_Y_clipped, dtype=np.float64)
    extracted_wm_clean = DWTSVD_DCT.extraction(
        input_image=watermarked_Y_contiguous,
        num_bits=wm_length,
        block_size=block_size,
        delta=delta,
        threshold=threshold,
        wavelet_name=wavelet_name
    )

    errors_clean = np.sum(original_wm != extracted_wm_clean)
    ber_clean = errors_clean / wm_length
    print(f"BER (Без атак): {ber_clean:.4f} ({errors_clean}/{wm_length} бит)")

    print("\n--- Устойчивость к атакам ---")
    attacks = {}

    for s in (0.5, 0.75, 1.5):
        attacks[f'Масштаб x{s}'] = cv2.resize(watermarked_bgr, None, fx=s, fy=s,
                                               interpolation=cv2.INTER_CUBIC)

    for angle in (2, 5, 15, 45, 90):
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
        attacks[f'Поворот {angle} град.'] = cv2.warpAffine(watermarked_bgr, M, (W, H))

    attacks['Кроп 448x448'] = watermarked_bgr[32:32 + 448, 32:32 + 448]

    for sigma in (0.01, 0.02, 0.05):
        noisy = watermarked_bgr.astype(np.float64) / 255.0
        noisy += np.random.normal(0, sigma, noisy.shape)
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        attacks[f'Гаусс. шум σ={sigma}'] = noisy

    for density in (0.01, 0.05, 0.10):
        noisy = watermarked_bgr.copy()
        mask = np.random.random(noisy.shape[:2]) < density
        noisy[mask, 0] = np.random.choice([0, 255])
        noisy[mask, 1] = np.random.choice([0, 255])
        noisy[mask, 2] = np.random.choice([0, 255])
        attacks[f'Salt&Pepper {density:.0%}'] = noisy

    for ksize in (3, 5, 7):
        blurred = cv2.GaussianBlur(watermarked_bgr, (ksize, ksize), 0)
        attacks[f'Размытие {ksize}x{ksize}'] = blurred

    for ksize in (3, 5, 7):
        median = cv2.medianBlur(watermarked_bgr, ksize)
        attacks[f'Медиана {ksize}x{ksize}'] = median

    for quality in (90, 75, 50, 30):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', watermarked_bgr, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        attacks[f'JPEG q={quality}'] = decoded

    for delta_val in (-30, -15, 15, 30):
        bright = cv2.convertScaleAbs(watermarked_bgr, alpha=1.0, beta=delta_val)
        attacks[f'Яркость {delta_val:+d}'] = bright

    for alpha in (0.7, 0.85, 1.15, 1.3):
        contrast = cv2.convertScaleAbs(watermarked_bgr, alpha=alpha, beta=0)
        attacks[f'Контраст x{alpha}'] = contrast

    for name, attacked in attacks.items():
        ber, errors = extract_ber(attacked, original_wm,
                                 block_size=block_size, delta=delta,
                                 threshold=threshold, wavelet_name=wavelet_name)
        print(f"{name:<25} BER = {ber:.4f} ({errors}/{wm_length} бит)")

    ber, errors = extract_ber(img_bgr, original_wm,
                             block_size=block_size, delta=delta,
                             threshold=threshold, wavelet_name=wavelet_name)
    print(f"\nКонтроль (оригинал без ЦВЗ): BER = {ber:.4f} ({errors}/{wm_length} бит)")


if __name__ == "__main__":
    main()