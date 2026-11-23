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
import yaml
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path):
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError(f"invalid paper protocol: {path}")
    return spec


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
        ("dwarf.ready_solutions.embedding_solutions.frequency", "dwarf/ready_solutions/embedding_solutions/frequency"),
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
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.ascontiguousarray(np.stack((y, (blue - y) / 1.772, (red - y) / 1.402), axis=-1))


def ycbcr_to_rgb(image):
    y, cb, cr = image[..., 0], image[..., 1], image[..., 2]
    red = y + 1.402 * cr
    blue = y + 1.772 * cb
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    return np.clip(np.floor(np.stack((red, green, blue), axis=-1) + 0.5), 0, 255).astype(np.uint8)


def mean(values):
    return float(np.mean(values))


def run_dct(algorithm, experiment, output):
    images_dir = ROOT / experiment["images_dir"]
    watermark_spec = experiment["watermark"]
    watermark = np.asarray(Image.open(ROOT / watermark_spec["file"]).convert("L"))
    bits = np.ascontiguousarray((watermark < watermark_spec["threshold"]).astype(np.uint8).reshape(-1))
    assert bits.size == watermark_spec["bit_count"]
    seed = experiment["seed"]
    repeats = experiment["repeats"]
    rng = np.random.default_rng(seed)
    channel_rows, summary = [], []
    for name, host in experiment["hosts"].items():
        original = np.asarray(Image.open(images_dir / host["file"]).convert(experiment["image_mode"]), dtype=np.uint8)
        assert original.shape == tuple(experiment["image_shape"])
        marked = ycbcr_to_rgb(
            algorithm.embedding(
                input_image=rgb_to_ycbcr(original), watermark_bits=bits, strength=experiment["embedding"]["strength"]
            )
        )
        for channel in experiment["channels"]:
            channel_rows.append(
                {
                    "image": name,
                    "channel": channel,
                    "measured_psnr_db": image_metric("Channel_PSNR", original, marked, channel=channel),
                }
            )
        for label, attack_spec in experiment["attacks"].items():
            observations = []
            for _ in range(repeats if attack_spec["stochastic"] else 1):
                parameters = dict(attack_spec["params"])
                if attack_spec["stochastic"]:
                    parameters["seed"] = int(rng.integers(0, 2**63))
                attacked = attack(marked, attack_spec["name"], **parameters)
                recovered = algorithm.extraction(input_image=rgb_to_ycbcr(attacked), num_bits=bits.size)
                observations.append(
                    {
                        "psnr": image_metric("PSNR", original, attacked),
                        "accuracy": bit_metric("Bit_Agreement", bits, recovered),
                        "ber": bit_metric("BER", bits, recovered),
                    }
                )
            summary.append(
                {
                    "image": name,
                    "attack": label,
                    "psnr_db_mean": mean([row["psnr"] for row in observations]),
                    "accuracy_mean": mean([row["accuracy"] for row in observations]),
                    "ber_mean": mean([row["ber"] for row in observations]),
                }
            )
        print(f"DCT {name}: complete", flush=True)
    save_csv(output / "table1.csv", channel_rows)
    save_csv(output / "summary.csv", summary)


def run_dwt(algorithm, experiment, output):
    images_dir = ROOT / experiment["images_dir"]
    seed = experiment["seed"]
    repeats = experiment["repeats"]
    watermark_spec = experiment["watermark"]
    quality, summary = [], []
    for index, (name, host) in enumerate(experiment["hosts"].items()):
        original = np.asarray(Image.open(images_dir / host["file"]).convert(experiment["image_mode"]), dtype=np.uint8)
        assert original.shape == tuple(experiment["image_shape"])
        key = seed * watermark_spec["key_multiplier"] + index
        bits = np.random.default_rng(
            np.random.SeedSequence([seed, index, watermark_spec["seed_sequence_suffix"]])
        ).integers(0, 2, watermark_spec["bit_count"], dtype=np.uint8)
        embedded = algorithm.embedding(input_image=original, watermark_bits=bits, key=key)
        marked = np.clip(np.floor(embedded["image"] + 0.5), 0, 255).astype(np.uint8)
        assert np.array_equal(marked.astype(np.float64), embedded["image"])
        quality.append(
            {
                "image": name,
                "psnr_db": image_metric("PSNR", original, marked),
                "mssim": image_metric("SSIM", original, marked),
            }
        )
        for attack_index, (label, attack_spec) in enumerate(experiment["attacks"].items()):
            observations = []
            for trial in range(repeats if attack_spec["stochastic"] else 1):
                parameters = dict(attack_spec["params"])
                if attack_spec["stochastic"]:
                    parameters["seed"] = int(
                        np.random.SeedSequence([seed, index, attack_index, trial]).generate_state(1)[0]
                    )
                attacked = attack(marked, attack_spec["name"], **parameters)
                decoded = algorithm.extraction(input_image=attacked, key=key, h=embedded["h"], return_details=True)
                observations.append(bit_metric("BER", bits, decoded["bits"]))
            summary.append({"image": name, "attack": label, "ber_mean": mean(observations)})
        print(f"DWT {name}: complete", flush=True)
    save_csv(output / "quality.csv", quality)
    save_csv(output / "summary.csv", summary)


def dft_attack(image, attack_spec, parameter):
    if attack_spec["name"] == "clean":
        return attack(image, "clean"), None
    parameters = dict(attack_spec.get("fixed_params", {}))
    parameters[attack_spec["parameter_name"]] = parameter
    result = attack(image, attack_spec["name"], **parameters)
    if attack_spec.get("records_bpp", False):
        pixels, bpp, _ = result
        return pixels, bpp
    return result, None


def run_dft(algorithm, experiment, output):
    if not features.check("jpg_2000"):
        raise RuntimeError("Pillow JPEG 2000 support is required")
    images_dir = ROOT / experiment["images_dir"]
    seed = experiment["seed"]
    repeats = experiment["repeats"]
    watermark_spec = experiment["watermark"]
    conditions = [
        (name, parameter)
        for name, attack_spec in experiment["attacks"].items()
        for parameter in attack_spec["parameters"]
    ]
    quality, weibull, grouped = [], [], {condition: [] for condition in conditions}
    for index, (name, host) in enumerate(experiment["hosts"].items()):
        original = np.asarray(Image.open(images_dir / host["file"]).convert(experiment["image_mode"]), dtype=np.uint8)
        assert original.shape == tuple(experiment["image_shape"])
        for trial in range(repeats):
            bits = np.random.default_rng(np.random.SeedSequence([seed, index, trial])).integers(
                0, 2, watermark_spec["bit_count"], dtype=np.uint8
            )
            key = hashlib.sha256(f"{watermark_spec['key_namespace']}:{seed}:{name}:{trial}".encode()).digest()
            embedded = algorithm.embedding(input_image=original, watermark_bits=bits, key=key)
            marked = embedded["image"]
            quality.append({"image": name, "trial": trial, "psnr_db": image_metric("PSNR", original, marked)})
            if name == experiment["weibull"]["host"] and trial == experiment["weibull"]["trial"]:
                details = algorithm.extraction(
                    input_image=marked, key=key, strength=embedded["strength"], return_details=True
                )
                parameters = details["weibull_parameters"]
                for region in range(experiment["weibull"]["regions"]):
                    weibull.append(
                        {
                            "region": region + 1,
                            "measured_alpha": float(parameters[region, 0] / original.size),
                            "measured_beta": float(parameters[region, 1]),
                        }
                    )
            for condition in conditions:
                attacked, bpp = dft_attack(marked, experiment["attacks"][condition[0]], condition[1])
                decoded = algorithm.extraction(input_image=attacked, key=key, strength=embedded["strength"])
                grouped[condition].append((bit_metric("BER", bits, decoded), bpp))
            if (trial + 1) % 16 == 0:
                print(f"DFT {name}: {trial + 1}/{repeats}", flush=True)
    summary = []
    for (name, parameter), observations in grouped.items():
        assert len(observations) == repeats * len(experiment["hosts"])
        bpp = [value for _, value in observations if value is not None]
        summary.append(
            {
                "image": "ALL",
                "attack": name,
                "parameter": parameter,
                "ber": mean([ber for ber, _ in observations]),
                "bpp_mean": mean(bpp) if bpp else "",
                "total_bits": watermark_spec["bit_count"] * len(observations),
            }
        )
    save_csv(output / "quality.csv", quality)
    save_csv(output / "weibull.csv", weibull)
    save_csv(output / "summary.csv", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    experiment = load_spec(config_path)["experiment"]
    algorithm_name = experiment["algorithm"]
    if algorithm_name not in ("DCT", "DWT", "DFT"):
        raise ValueError(f"unsupported paper algorithm: {algorithm_name}")
    images_dir = ROOT / experiment["images_dir"]
    for name, host in experiment["hosts"].items():
        if sha256(images_dir / host["file"]) != host["sha256"]:
            raise ValueError(f"host image changed: {name}")
    watermark = experiment["watermark"]
    if "file" in watermark and sha256(ROOT / watermark["file"]) != watermark["sha256"]:
        raise ValueError("watermark image changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithm = load_algorithm(algorithm_name.lower(), args.backend_dir.resolve())
    load_services()
    if algorithm_name == "DCT":
        run_dct(algorithm, experiment, args.output_dir)
    elif algorithm_name == "DWT":
        run_dwt(algorithm, experiment, args.output_dir)
    else:
        run_dft(algorithm, experiment, args.output_dir)
    manifest = {
        "run_kind": f"test_{algorithm_name}",
        "seed": experiment["seed"],
        "repeats": experiment["repeats"],
        "backend_dir": str(args.backend_dir.resolve()),
        "source_sha256": sha256(ROOT / experiment["source"]),
        "config_sha256": sha256(config_path),
        "inputs": {
            name: {"file": host["file"], "sha256": sha256(images_dir / host["file"])}
            for name, host in experiment["hosts"].items()
        },
    }
    if "file" in watermark:
        manifest["watermark_sha256"] = sha256(ROOT / watermark["file"])
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
