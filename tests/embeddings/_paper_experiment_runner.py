"""Measure article experiments without importing report generators or plotting."""

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data_embed"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(ROOT))

services = importlib.import_module("experiment_services")
attack = services.attack
bit_metric = services.bit_metric
image_metric = services.image_metric
load_services = services.load_services


HOSTS = {
    "DCT": {"Airplane": "4.2.05.tiff", "Lena": "4.2.04.tiff",
            "Peppers": "4.2.07.tiff", "Sailboat": "4.2.06.tiff"},
    "DWT": {"Peppers": "4.2.07.tiff", "Baboon": "4.2.03.tiff",
            "Barbara": "barbara.pgm", "Lena": "4.2.04.tiff"},
    "DFT": {"Lena": "4.2.04.tiff", "Tiffany": "4.2.02.tiff", "Lake": "4.2.06.tiff"},
}
DCT_ATTACKS = {
    "clean": ("clean", {}),
    "gaussian": ("AWGN", {"sigma": .01}),
    "salt_pepper": ("Salt_and_Pepper", {"density": .01}),
    "speckle": ("Speckle", {"variance": .01}),
    "median": ("Median_Filter", {"window": 3}),
    "wiener": ("Wiener_Filter", {"window": 3}),
    **{f"jpeg{q}": ("Jpeg", {"quality": q}) for q in (50, 60, 70, 80)},
}
DWT_ATTACKS = {
    "clean": ("clean", {}),
    "vsa1.5": ("Brightness_Contrast", {"brightness": 1.5, "contrast": 1.0}),
    "gaussian5": ("Gaussian_Blur", {"sigma": 1.1, "window": 5}),
    "median3": ("Median_Filter", {"window": 3}),
    "jpeg11": ("Jpeg", {"quality": 11, "grayscale": True}),
    "jpeg20": ("Jpeg", {"quality": 20, "grayscale": True}),
    "awgn10": ("AWGN", {"sigma": 10 / 255}),
    "awgn20": ("AWGN", {"sigma": 20 / 255}),
    "salt_pepper1": ("Salt_and_Pepper", {"density": .01}),
}
DFT_CONDITIONS = (
    [("clean", 0)]
    + [("median", size) for size in (3, 5, 7)]
    + [("jpeg", q) for q in (1, 3, 5, 8, 12, 18, 25, 35, 50, 75)]
    + [("jpeg2000", rate) for rate in (8, 12, 16, 24, 32, 48, 64, 96, 128)]
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"no observations for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_algorithm(name, backend):
    importlib.import_module("dwarf")

    for package, relative in (
        ("dwarf.ready_solutions", "dwarf/ready_solutions"),
        ("dwarf.ready_solutions.utils", "dwarf/ready_solutions/utils"),
        ("dwarf.ready_solutions.embedding_solutions", "dwarf/ready_solutions/embedding_solutions"),
        ("dwarf.ready_solutions.embedding_solutions.frequency",
         "dwarf/ready_solutions/embedding_solutions/frequency"),
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / relative)]
        sys.modules[package] = module
    for full_name, stem in (
        ("dwarf.ready_solutions.utils.embedding_utils_pyx", "embedding_utils_pyx"),
        (f"dwarf.ready_solutions.embedding_solutions.frequency.{name}", name),
    ):
        candidates = [path for path in backend.rglob(stem + "*") if path.suffix in (".pyd", ".so")]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one {stem} extension in {backend}, found {len(candidates)}")
        spec = importlib.util.spec_from_file_location(full_name, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[f"dwarf.ready_solutions.embedding_solutions.frequency.{name}"], name.upper())


def rgb_to_ycbcr(image):
    values = image.astype(np.float64)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    y = .299 * red + .587 * green + .114 * blue
    return np.ascontiguousarray(np.stack((y, (blue - y) / 1.772, (red - y) / 1.402), axis=-1))


def ycbcr_to_rgb(image):
    y, cb, cr = image[..., 0], image[..., 1], image[..., 2]
    red = y + 1.402 * cr
    blue = y + 1.772 * cb
    green = (y - .299 * red - .114 * blue) / .587
    return np.clip(np.floor(np.stack((red, green, blue), axis=-1) + .5), 0, 255).astype(np.uint8)


def mean(values):
    return float(np.mean(values))


def run_dct(algorithm, images_dir, output, seed, repeats):
    watermark = np.asarray(Image.open(DATA / "dct_watermark_approx.png").convert("L"))
    bits = np.ascontiguousarray((watermark < 128).astype(np.uint8).reshape(-1))
    assert bits.size == 4096
    rng = np.random.default_rng(seed)
    channel_rows, summary = [], []
    stochastic = {"gaussian", "salt_pepper", "speckle"}
    for name, filename in HOSTS["DCT"].items():
        original = np.asarray(Image.open(images_dir / filename).convert("RGB"), dtype=np.uint8)
        assert original.shape == (512, 512, 3)
        marked = ycbcr_to_rgb(algorithm.embedding(input_image=rgb_to_ycbcr(original),
                                                  watermark_bits=bits, strength=16.0))
        for channel in "RGB":
            channel_rows.append({"image": name, "channel": channel,
                                 "measured_psnr_db": image_metric("Channel_PSNR", original, marked,
                                                                  channel=channel)})
        for label, (attack_name, settings) in DCT_ATTACKS.items():
            observations = []
            for _ in range(repeats if label in stochastic else 1):
                parameters = dict(settings)
                if label in stochastic:
                    parameters["seed"] = int(rng.integers(0, 2**63))
                attacked = attack(marked, attack_name, **parameters)
                recovered = algorithm.extraction(input_image=rgb_to_ycbcr(attacked), num_bits=bits.size)
                observations.append({
                    "psnr": image_metric("PSNR", original, attacked),
                    "accuracy": bit_metric("Bit_Agreement", bits, recovered),
                    "ber": bit_metric("BER", bits, recovered),
                })
            summary.append({"image": name, "attack": label,
                            "psnr_db_mean": mean([row["psnr"] for row in observations]),
                            "accuracy_mean": mean([row["accuracy"] for row in observations]),
                            "ber_mean": mean([row["ber"] for row in observations])})
        print(f"DCT {name}: complete", flush=True)
    save_csv(output / "table1.csv", channel_rows)
    save_csv(output / "summary.csv", summary)


def run_dwt(algorithm, images_dir, output, seed, repeats):
    quality, summary = [], []
    stochastic = {"awgn10", "awgn20", "salt_pepper1"}
    for index, (name, filename) in enumerate(HOSTS["DWT"].items()):
        original = np.asarray(Image.open(images_dir / filename).convert("L"), dtype=np.uint8)
        assert original.shape == (512, 512)
        key = seed * 1000 + index
        bits = np.random.default_rng(np.random.SeedSequence([seed, index, 99])).integers(
            0, 2, 256, dtype=np.uint8)
        embedded = algorithm.embedding(input_image=original, watermark_bits=bits, key=key)
        marked = np.clip(np.floor(embedded["image"] + .5), 0, 255).astype(np.uint8)
        assert np.array_equal(marked.astype(np.float64), embedded["image"])
        quality.append({"image": name, "psnr_db": image_metric("PSNR", original, marked),
                        "mssim": image_metric("SSIM", original, marked)})
        for attack_index, (label, (attack_name, settings)) in enumerate(DWT_ATTACKS.items()):
            observations = []
            for trial in range(repeats if label in stochastic else 1):
                parameters = dict(settings)
                if label in stochastic:
                    parameters["seed"] = int(np.random.SeedSequence(
                        [seed, index, attack_index, trial]).generate_state(1)[0])
                attacked = attack(marked, attack_name, **parameters)
                decoded = algorithm.extraction(input_image=attacked, key=key,
                                               h=embedded["h"], return_details=True)
                observations.append(bit_metric("BER", bits, decoded["bits"]))
            summary.append({"image": name, "attack": label, "ber_mean": mean(observations)})
        print(f"DWT {name}: complete", flush=True)
    save_csv(output / "quality.csv", quality)
    save_csv(output / "summary.csv", summary)


def dft_attack(image, name, parameter):
    if name == "clean":
        return attack(image, "clean"), None
    if name == "median":
        return attack(image, "Median_Filter", window=parameter), None
    if name == "jpeg":
        pixels, bpp, _ = attack(image, "Jpeg", quality=parameter,
                               grayscale=True, return_stats=True)
        return pixels, bpp
    pixels, bpp, _ = attack(image, "Jpeg2000", compression_ratio=parameter,
                           grayscale=True, irreversible=False, return_stats=True)
    return pixels, bpp


def run_dft(algorithm, images_dir, output, seed, repeats):
    if not features.check("jpg_2000"):
        raise RuntimeError("Pillow JPEG 2000 support is required")
    quality, weibull, grouped = [], [], {condition: [] for condition in DFT_CONDITIONS}
    for index, (name, filename) in enumerate(HOSTS["DFT"].items()):
        original = np.asarray(Image.open(images_dir / filename).convert("L"), dtype=np.uint8)
        assert original.shape == (512, 512)
        for trial in range(repeats):
            bits = np.random.default_rng(np.random.SeedSequence([seed, index, trial])).integers(
                0, 2, 64, dtype=np.uint8)
            key = hashlib.sha256(f"dft:{seed}:{name}:{trial}".encode()).digest()
            embedded = algorithm.embedding(input_image=original, watermark_bits=bits, key=key)
            marked = embedded["image"]
            quality.append({"image": name, "trial": trial,
                            "psnr_db": image_metric("PSNR", original, marked)})
            if name == "Lena" and trial == 0:
                details = algorithm.extraction(input_image=marked, key=key,
                                               strength=embedded["strength"], return_details=True)
                parameters = details["weibull_parameters"]
                for region in range(16):
                    weibull.append({"region": region + 1,
                                    "measured_alpha": float(parameters[region, 0] / original.size),
                                    "measured_beta": float(parameters[region, 1])})
            for condition in DFT_CONDITIONS:
                attacked, bpp = dft_attack(marked, *condition)
                decoded = algorithm.extraction(input_image=attacked, key=key,
                                               strength=embedded["strength"])
                grouped[condition].append((bit_metric("BER", bits, decoded), bpp))
            if (trial + 1) % 16 == 0:
                print(f"DFT {name}: {trial + 1}/{repeats}", flush=True)
    summary = []
    for (name, parameter), observations in grouped.items():
        assert len(observations) == repeats * len(HOSTS["DFT"])
        bpp = [value for _, value in observations if value is not None]
        summary.append({"image": "ALL", "attack": name, "parameter": parameter,
                        "ber": mean([ber for ber, _ in observations]),
                        "bpp_mean": mean(bpp) if bpp else "",
                        "total_bits": 64 * len(observations)})
    save_csv(output / "quality.csv", quality)
    save_csv(output / "weibull.csv", weibull)
    save_csv(output / "summary.csv", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("DCT", "DWT", "DFT"), required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithm = load_algorithm(args.algorithm.lower(), args.backend_dir.resolve())
    load_services()
    images_dir = DATA / "images" / args.algorithm
    if args.algorithm == "DCT":
        run_dct(algorithm, images_dir, args.output_dir, args.seed, args.repeats)
    elif args.algorithm == "DWT":
        run_dwt(algorithm, images_dir, args.output_dir, args.seed, args.repeats)
    else:
        run_dft(algorithm, images_dir, args.output_dir, args.seed, args.repeats)
    manifest = {
        "run_kind": f"test_{args.algorithm}", "seed": args.seed, "repeats": args.repeats,
        "backend_dir": str(args.backend_dir.resolve()),
        "source_sha256": sha256(ROOT / "dwarf/ready_solutions/embedding_solutions/frequency" /
                                f"{args.algorithm.lower()}.pyx"),
        "paper_sha256": sha256(DATA / f"{args.algorithm.lower()}_paper_reference.json"),
        "inputs": {name: {"file": filename, "sha256": sha256(images_dir / filename)}
                   for name, filename in HOSTS[args.algorithm].items()},
    }
    if args.algorithm == "DCT":
        manifest["watermark_sha256"] = sha256(DATA / "dct_watermark_approx.png")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
