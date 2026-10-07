"""Tests specific to image compression attacks."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from .helpers import attack_class, mean_absolute_error, run_attack


def _mock_file_codec(monkeypatch, module, output):
    """Replace file-codec boundaries and return the calls made by the wrapper."""
    observed = {"lookups": [], "commands": [], "saved": [], "opened": []}
    decoded_image = object()

    class SourceImage:
        def save(self, path):
            observed["saved"].append(Path(path))

    def fake_which(executable):
        observed["lookups"].append(executable)
        return str(Path("mock-codecs") / executable)

    def fake_run(command, *, check):
        observed["commands"].append((list(command), check))

    def fake_to_pil(image):
        observed["input"] = image
        return SourceImage()

    def fake_open(path):
        observed["opened"].append(Path(path))
        return decoded_image

    def fake_to_array(image):
        assert image is decoded_image
        return output

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "to_pil", fake_to_pil)
    monkeypatch.setattr(module, "Image", SimpleNamespace(open=fake_open))
    monkeypatch.setattr(module, "to_array", fake_to_array)
    return observed


@pytest.mark.parametrize("compression", ["tiff_lzw", "tiff_adobe_deflate"])
def test_tiff_modes_are_bit_exact(compression, photo):
    assert np.array_equal(run_attack("Tiff", photo, compression=compression), photo)


def test_webp_lossless_mode_is_bit_exact(photo):
    assert np.array_equal(run_attack("Webp", photo, lossless=True), photo)


@pytest.mark.parametrize(("worse", "better"), [(5, 95), (30, 90), (50, 95)])
def test_jpeg_lower_quality_distorts_more(worse, better, photo):
    worse_error = mean_absolute_error(run_attack("Jpeg", photo, quality=worse), photo)
    better_error = mean_absolute_error(run_attack("Jpeg", photo, quality=better), photo)

    assert worse_error > better_error


def test_jpeg2000_higher_ratio_distorts_more(photo):
    mild_error = mean_absolute_error(run_attack("Jpeg2000", photo, compression_ratio=5), photo)
    harsh_error = mean_absolute_error(run_attack("Jpeg2000", photo, compression_ratio=200), photo)

    assert harsh_error > mild_error


def test_avif_success_forwards_format_and_quality(monkeypatch, small_photo):
    module = sys.modules["dwarf.ready_solutions.attack_solutions.compression.avif"]
    monkeypatch.setitem(sys.modules, "pillow_avif", ModuleType("pillow_avif"))
    output = np.full_like(small_photo, 17)
    calls = []

    def fake_roundtrip(image, fmt, **save_kwargs):
        calls.append((image, fmt, save_kwargs))
        return output

    monkeypatch.setattr(module, "roundtrip_buffer", fake_roundtrip)

    result = attack_class("Avif").attack(input_image=small_photo, quality=61)

    assert result is output
    assert len(calls) == 1
    forwarded_image, fmt, save_kwargs = calls[0]
    assert forwarded_image is small_photo
    assert fmt == "AVIF"
    assert save_kwargs == {"quality": 61}


def test_avif_success_via_pillow_heif_registers_avif_opener(monkeypatch, small_photo):
    module = sys.modules["dwarf.ready_solutions.attack_solutions.compression.avif"]
    pillow_heif = ModuleType("pillow_heif")
    registrations = []
    pillow_heif.register_avif_opener = lambda: registrations.append("AVIF")
    monkeypatch.setitem(sys.modules, "pillow_avif", None)
    monkeypatch.setitem(sys.modules, "pillow_heif", pillow_heif)
    output = np.full_like(small_photo, 23)
    calls = []

    def fake_roundtrip(image, fmt, **save_kwargs):
        calls.append((image, fmt, save_kwargs))
        return output

    monkeypatch.setattr(module, "roundtrip_buffer", fake_roundtrip)

    result = attack_class("Avif").attack(input_image=small_photo, quality=47)

    assert result is output
    assert registrations == ["AVIF"]
    assert len(calls) == 1
    assert calls[0][0] is small_photo
    assert calls[0][1:] == ("AVIF", {"quality": 47})


def test_heic_success_registers_opener_and_forwards_options(monkeypatch, small_photo):
    module = sys.modules["dwarf.ready_solutions.attack_solutions.compression.heic"]
    pillow_heif = ModuleType("pillow_heif")
    registrations = []
    pillow_heif.register_heif_opener = lambda: registrations.append("HEIF")
    monkeypatch.setitem(sys.modules, "pillow_heif", pillow_heif)
    output = np.full_like(small_photo, 29)
    calls = []

    def fake_roundtrip(image, fmt, **save_kwargs):
        calls.append((image, fmt, save_kwargs))
        return output

    monkeypatch.setattr(module, "roundtrip_buffer", fake_roundtrip)

    result = attack_class("Heic").attack(input_image=small_photo, quality=73)

    assert result is output
    assert registrations == ["HEIF"]
    assert len(calls) == 1
    assert calls[0][0] is small_photo
    assert calls[0][1:] == ("HEIF", {"quality": 73})


def test_bpg_success_forwards_quality_and_decodes_command_output(monkeypatch, small_photo):
    module = sys.modules["dwarf.ready_solutions.attack_solutions.compression.bpg"]
    output = np.full_like(small_photo, 31)
    observed = _mock_file_codec(monkeypatch, module, output)

    result = attack_class("Bpg").attack(input_image=small_photo, quality=17)

    source = observed["saved"][0]
    encoded = source.with_name("encoded.bpg")
    decoded = source.with_name("decoded.png")
    assert result is output
    assert observed["input"] is small_photo
    assert observed["lookups"] == ["bpgenc", "bpgdec"]
    assert observed["opened"] == [decoded]
    assert observed["commands"] == [
        (["bpgenc", "-q", "17", "-o", str(encoded), str(source)], True),
        (["bpgdec", "-o", str(decoded), str(encoded)], True),
    ]


def test_flif_success_runs_lossless_round_trip_commands(monkeypatch, small_photo):
    module = sys.modules["dwarf.ready_solutions.attack_solutions.compression.flif"]
    output = np.full_like(small_photo, 37)
    observed = _mock_file_codec(monkeypatch, module, output)

    result = attack_class("Flif").attack(input_image=small_photo)

    source = observed["saved"][0]
    encoded = source.with_name("encoded.flif")
    decoded = source.with_name("decoded.png")
    assert result is output
    assert observed["input"] is small_photo
    assert observed["lookups"] == ["flif"]
    assert observed["opened"] == [decoded]
    assert observed["commands"] == [
        (["flif", "-e", str(source), str(encoded)], True),
        (["flif", "-d", str(encoded), str(decoded)], True),
    ]


@pytest.mark.parametrize(
    ("name", "module_name", "expected_tool"),
    [
        ("Bpg", "dwarf.ready_solutions.attack_solutions.compression.bpg", "bpgenc"),
        ("Flif", "dwarf.ready_solutions.attack_solutions.compression.flif", "flif"),
    ],
)
def test_external_codec_reports_missing_executable(name, module_name, expected_tool, monkeypatch, small_photo):
    module = sys.modules[module_name]
    monkeypatch.setattr(module.shutil, "which", lambda _executable: None)

    with pytest.raises(RuntimeError, match=expected_tool):
        attack_class(name).attack(input_image=small_photo)


def test_heic_reports_missing_python_dependency(monkeypatch, small_photo):
    monkeypatch.setitem(sys.modules, "pillow_heif", None)

    with pytest.raises(RuntimeError, match="pillow-heif"):
        attack_class("Heic").attack(input_image=small_photo)


def test_avif_reports_missing_python_dependencies(monkeypatch, small_photo):
    monkeypatch.setitem(sys.modules, "pillow_avif", None)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)

    with pytest.raises(RuntimeError, match="pillow-avif-plugin or pillow-heif"):
        attack_class("Avif").attack(input_image=small_photo)
