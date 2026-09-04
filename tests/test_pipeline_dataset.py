"""Tests for deterministic directory manifests and canonical image loading."""

import json
import os
import pickle
import shutil

import numpy as np
import pytest
from PIL import Image

import dwarf.pipeline.datasets as dataset_module
from dwarf.pipeline import (
    DatasetDiscoveryError,
    DatasetIntegrityError,
    DatasetLoadError,
    DatasetManifest,
    DatasetManifestError,
    DirectoryDatasetSource,
    DirectoryDatasetSpec,
    ImageArtifact,
    SampleReference,
    make_sample_id,
)


def save_image(path, *, mode="RGB", size=(8, 6), color=None, exif=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        color = 90 if mode == "L" else (20, 80, 140)
    image = Image.new(mode, size, color=color)
    save_options = {} if exif is None else {"exif": exif}
    image.save(path, **save_options)
    return path


def make_spec(root, **updates):
    data = {
        "type": "directory",
        "path": root,
        "recursive": True,
        "extensions": [".png", ".jpg", ".jpeg", ".webp"],
    }
    data.update(updates)
    return DirectoryDatasetSpec.model_validate(data)


def relative_paths(manifest):
    return [sample.relative_path for sample in manifest.samples]


def test_recursive_discovery_is_filtered_and_deterministically_sorted(tmp_path):
    root = tmp_path / "images"
    save_image(root / "b.JPG")
    save_image(root / "a.png")
    save_image(root / "nested" / "c.jpeg")
    save_image(root / "nested" / "deeper" / "D.PNG")
    (root / "nested" / "notes.txt").write_text("not an image", encoding="utf-8")

    source = DirectoryDatasetSource(make_spec(root))
    manifest = source.build_manifest(seed=91)

    assert relative_paths(manifest) == [
        "a.png",
        "b.JPG",
        "nested/c.jpeg",
        "nested/deeper/D.PNG",
    ]
    assert manifest.root == root.resolve()
    assert manifest.seed == 91
    assert not manifest.shuffled
    assert len(manifest) == 4
    assert len(manifest.fingerprint) == 64
    assert all(sample.path.is_absolute() for sample in manifest)
    assert all(len(sample.checksum_sha256) == 64 for sample in manifest)




def test_relative_root_is_bound_when_source_is_created(tmp_path, monkeypatch):
    first_working_directory = tmp_path / "first"
    second_working_directory = tmp_path / "second"
    save_image(first_working_directory / "images" / "a.png")
    second_working_directory.mkdir()

    monkeypatch.chdir(first_working_directory)
    source = DirectoryDatasetSource(make_spec("images"))
    monkeypatch.chdir(second_working_directory)

    manifest = source.build_manifest()

    assert source.root == (first_working_directory / "images").resolve()
    assert manifest.root == source.root



def test_directory_dataset_source_is_pickle_safe(tmp_path):
    root = tmp_path / "images"
    save_image(root / "a.png")
    source = DirectoryDatasetSource(make_spec(root))

    restored = pickle.loads(pickle.dumps(source))
    manifest = restored.build_manifest(seed=17)

    assert restored.root == source.root
    assert relative_paths(manifest) == ["a.png"]

def test_non_recursive_discovery_ignores_nested_images(tmp_path):
    root = tmp_path / "images"
    save_image(root / "root.png")
    save_image(root / "nested" / "nested.png")

    source = DirectoryDatasetSource(make_spec(root, recursive=False))
    manifest = source.build_manifest()

    assert relative_paths(manifest) == ["root.png"]
    assert not manifest.recursive


def test_include_and_exclude_patterns_use_relative_posix_paths(tmp_path):
    root = tmp_path / "images"
    save_image(root / "keep" / "a.png")
    save_image(root / "keep" / "drop.png")
    save_image(root / "other" / "b.png")
    save_image(root / "root.png")

    source = DirectoryDatasetSource(
        make_spec(
            root,
            include=["keep/*.png", "root.png"],
            exclude=["**/drop.png"],
        )
    )
    manifest = source.build_manifest()

    assert relative_paths(manifest) == ["keep/a.png", "root.png"]


def test_root_level_file_matches_double_star_glob(tmp_path):
    root = tmp_path / "images"
    save_image(root / "root.png")

    manifest = DirectoryDatasetSource(
        make_spec(root, include=["**/*.png"])
    ).build_manifest()

    assert relative_paths(manifest) == ["root.png"]


def test_seeded_shuffle_is_reproducible_and_limit_is_applied_after_shuffle(tmp_path):
    root = tmp_path / "images"
    for index in range(12):
        save_image(root / f"image-{index:02d}.png", color=(index, index, index))

    source = DirectoryDatasetSource(make_spec(root, shuffle=True, limit=5))
    first = source.build_manifest(seed=1234)
    second = source.build_manifest(seed=1234)
    third = source.build_manifest(seed=9876)

    assert relative_paths(first) == relative_paths(second)
    assert first.fingerprint == second.fingerprint
    assert len(first) == 5
    assert first.shuffled
    assert first.limit == 5
    assert relative_paths(first) != relative_paths(third)
    assert first.fingerprint != third.fingerprint


def test_manifest_fingerprint_is_independent_of_absolute_root_and_mtime(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    save_image(first_root / "a.png", color=(1, 2, 3))
    save_image(first_root / "nested" / "b.png", color=(4, 5, 6))
    shutil.copytree(first_root, second_root)

    for path in second_root.rglob("*.png"):
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))

    first = DirectoryDatasetSource(make_spec(first_root)).build_manifest()
    second = DirectoryDatasetSource(make_spec(second_root)).build_manifest()

    assert first.root != second.root
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert first.fingerprint == second.fingerprint


def test_sample_ids_include_both_relative_path_and_content(tmp_path):
    root = tmp_path / "images"
    first_path = save_image(root / "a.png", color=(10, 20, 30))
    shutil.copy2(first_path, root / "copy.png")

    source = DirectoryDatasetSource(make_spec(root))
    before = source.build_manifest()
    before_by_path = {sample.relative_path: sample for sample in before}

    assert before_by_path["a.png"].checksum_sha256 == before_by_path["copy.png"].checksum_sha256
    assert before_by_path["a.png"].sample_id != before_by_path["copy.png"].sample_id

    save_image(root / "a.png", color=(30, 20, 10))
    after = source.build_manifest()
    after_by_path = {sample.relative_path: sample for sample in after}

    assert before_by_path["a.png"].sample_id != after_by_path["a.png"].sample_id
    assert before_by_path["copy.png"].sample_id == after_by_path["copy.png"].sample_id


def test_manifest_is_pickle_safe_and_has_json_representation(tmp_path):
    root = tmp_path / "images"
    save_image(root / "a.png")
    manifest = DirectoryDatasetSource(make_spec(root)).build_manifest(seed=7)

    restored = pickle.loads(pickle.dumps(manifest))
    payload = json.loads(manifest.to_json(include_absolute_paths=False))

    assert restored == manifest
    assert restored.fingerprint == manifest.fingerprint
    assert manifest.by_id(manifest.samples[0].sample_id) == manifest.samples[0]
    with pytest.raises(KeyError):
        manifest.by_id("0" * 64)
    assert payload["fingerprint"] == manifest.fingerprint
    assert "path" not in payload["samples"][0]




def test_manifest_canonicalises_unordered_selection_metadata(tmp_path):
    root = tmp_path / "images"
    save_image(root / "a.png")
    sample = DirectoryDatasetSource(make_spec(root)).build_manifest().samples[0]

    first = DatasetManifest(
        root=root.resolve(),
        samples=(sample,),
        recursive=True,
        extensions=(".PNG", ".jpg"),
        include=("b/**", "a/**"),
        exclude=("z/**", "y/**"),
        shuffled=False,
        seed=0,
        limit=None,
    )
    second = DatasetManifest(
        root=root.resolve(),
        samples=(sample,),
        recursive=True,
        extensions=(".jpg", ".png"),
        include=("a/**", "b/**"),
        exclude=("y/**", "z/**"),
        shuffled=False,
        seed=0,
        limit=None,
    )

    assert first.extensions == (".jpg", ".png")
    assert first.include == ("a/**", "b/**")
    assert first.exclude == ("y/**", "z/**")
    assert first == second
    assert first.fingerprint == second.fingerprint


def test_manifest_rejects_invalid_public_values(tmp_path):
    root = (tmp_path / "images").resolve()
    root.mkdir()
    checksum = "1" * 64

    with pytest.raises(DatasetManifestError, match="relative_path"):
        make_sample_id("../outside.png", checksum)
    with pytest.raises(DatasetManifestError, match="drive prefix"):
        make_sample_id("C:/outside.png", checksum)
    with pytest.raises(DatasetManifestError, match="64 hexadecimal"):
        make_sample_id("a.png", "not-a-checksum")

    sample = SampleReference(
        sample_id=make_sample_id("a.png", checksum),
        relative_path="a.png",
        path=root / "a.png",
        size_bytes=1,
        modified_time_ns=1,
        checksum_sha256=checksum,
    )

    with pytest.raises(DatasetManifestError, match="extensions"):
        DatasetManifest(
            root=root,
            samples=(sample,),
            recursive=True,
            extensions=(".png", ".PNG"),
            include=(),
            exclude=(),
            shuffled=False,
            seed=0,
            limit=None,
        )

def test_manifest_rejects_tampered_sample_identifier(tmp_path):
    root = (tmp_path / "images").resolve()
    root.mkdir()
    checksum = "1" * 64

    with pytest.raises(DatasetManifestError, match="sample_id"):
        SampleReference(
            sample_id="2" * 64,
            relative_path="a.png",
            path=root / "a.png",
            size_bytes=1,
            modified_time_ns=1,
            checksum_sha256=checksum,
        )


def test_manifest_rejects_sample_outside_root(tmp_path):
    root = (tmp_path / "images").resolve()
    other = (tmp_path / "other").resolve()
    root.mkdir()
    other.mkdir()
    checksum = "1" * 64
    sample = SampleReference(
        sample_id=make_sample_id("a.png", checksum),
        relative_path="a.png",
        path=other / "a.png",
        size_bytes=1,
        modified_time_ns=1,
        checksum_sha256=checksum,
    )

    with pytest.raises(DatasetManifestError, match="outside"):
        DatasetManifest(
            root=root,
            samples=(sample,),
            recursive=True,
            extensions=(".png",),
            include=(),
            exclude=(),
            shuffled=False,
            seed=0,
            limit=None,
        )


def test_load_converts_grayscale_resizes_and_returns_canonical_artifact(tmp_path):
    root = tmp_path / "images"
    save_image(root / "gray.png", mode="L", size=(7, 5), color=123)
    source = DirectoryDatasetSource(
        make_spec(
            root,
            preprocessing={
                "color_mode": "RGB",
                "resize": [4, 3],
                "resample": "nearest",
                "exif_transpose": True,
            },
        )
    )
    sample = source.build_manifest().samples[0]

    artifact = source.load(sample)

    assert isinstance(artifact, ImageArtifact)
    assert artifact.shape == (3, 4, 3)
    assert artifact.array.dtype == np.uint8
    assert artifact.array.flags.c_contiguous
    assert artifact.array.flags.writeable
    np.testing.assert_array_equal(artifact.array, np.full((3, 4, 3), 123, dtype=np.uint8))


def test_load_applies_exif_orientation_before_resize(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    save_image(root / "oriented.jpg", size=(2, 3), exif=exif)

    transposed_source = DirectoryDatasetSource(
        make_spec(root, preprocessing={"exif_transpose": True})
    )
    raw_source = DirectoryDatasetSource(
        make_spec(root, preprocessing={"exif_transpose": False})
    )
    sample = transposed_source.build_manifest().samples[0]

    assert transposed_source.load(sample).shape[:2] == (2, 3)
    assert raw_source.load(sample).shape[:2] == (3, 2)


def test_load_converts_rgba_to_rgb(tmp_path):
    root = tmp_path / "images"
    save_image(root / "alpha.png", mode="RGBA", color=(10, 20, 30, 40))
    source = DirectoryDatasetSource(make_spec(root))

    artifact = source.load(source.build_manifest().samples[0])

    assert artifact.shape == (6, 8, 3)
    assert artifact.array[0, 0].tolist() == [10, 20, 30]


def test_verify_and_load_detect_modified_file(tmp_path):
    root = tmp_path / "images"
    path = save_image(root / "a.png", color=(10, 20, 30))
    source = DirectoryDatasetSource(make_spec(root))
    sample = source.build_manifest().samples[0]

    save_image(path, color=(200, 100, 50))

    with pytest.raises(DatasetIntegrityError, match="manifest"):
        source.verify(sample)
    with pytest.raises(DatasetIntegrityError):
        source.load(sample)

    artifact = source.load(sample, verify_integrity=False)
    assert artifact.array[0, 0].tolist() == [200, 100, 50]


def test_load_rejects_sample_from_another_dataset(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    save_image(first_root / "a.png")
    save_image(second_root / "a.png")
    first_source = DirectoryDatasetSource(make_spec(first_root))
    second_source = DirectoryDatasetSource(make_spec(second_root))
    sample = first_source.build_manifest().samples[0]

    with pytest.raises(DatasetIntegrityError, match="does not belong"):
        second_source.load(sample, verify_integrity=False)


def test_corrupt_image_is_discovered_but_fails_with_dataset_load_error(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "broken.png").write_bytes(b"not a real image")
    source = DirectoryDatasetSource(make_spec(root))
    sample = source.build_manifest().samples[0]

    with pytest.raises(DatasetLoadError, match="failed to decode"):
        source.load(sample)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_build_manifest_rejects_invalid_seed(tmp_path, seed):
    root = tmp_path / "images"
    save_image(root / "a.png")
    source = DirectoryDatasetSource(make_spec(root))

    with pytest.raises((TypeError, ValueError)):
        source.build_manifest(seed=seed)


def test_missing_non_directory_and_empty_sources_fail_clearly(tmp_path):
    missing_source = DirectoryDatasetSource(make_spec(tmp_path / "missing"))
    with pytest.raises(DatasetDiscoveryError, match="does not exist"):
        missing_source.build_manifest()

    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    file_source = DirectoryDatasetSource(make_spec(file_path))
    with pytest.raises(DatasetDiscoveryError, match="not a directory"):
        file_source.build_manifest()

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty_source = DirectoryDatasetSource(make_spec(empty_root))
    with pytest.raises(DatasetDiscoveryError, match="no image files"):
        empty_source.build_manifest()


def test_symbolic_link_files_and_directories_are_not_followed(tmp_path):
    root = tmp_path / "images"
    outside = tmp_path / "outside"
    save_image(root / "real.png")
    save_image(outside / "external.png")

    try:
        (root / "linked-file.png").symlink_to(outside / "external.png")
        (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are not available in this environment")

    manifest = DirectoryDatasetSource(make_spec(root)).build_manifest()

    assert relative_paths(manifest) == ["real.png"]


def test_manifest_creation_detects_file_changed_while_hashing(tmp_path, monkeypatch):
    root = tmp_path / "images"
    path = save_image(root / "a.png", color=(1, 2, 3))
    source = DirectoryDatasetSource(make_spec(root))
    original_hash = dataset_module._sha256_file

    def hash_then_modify(target):
        checksum = original_hash(target)
        save_image(path, color=(3, 2, 1))
        return checksum

    monkeypatch.setattr(dataset_module, "_sha256_file", hash_then_modify)

    with pytest.raises(DatasetIntegrityError, match="changed while"):
        source.build_manifest()


def test_iter_loaded_is_lazy_and_rejects_manifest_from_other_root(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    save_image(first_root / "a.png")
    save_image(first_root / "b.png")
    save_image(second_root / "a.png")
    first_source = DirectoryDatasetSource(make_spec(first_root))
    second_source = DirectoryDatasetSource(make_spec(second_root))
    manifest = first_source.build_manifest()

    iterator = first_source.iter_loaded(manifest)
    first_sample, first_artifact = next(iterator)

    assert first_sample == manifest.samples[0]
    assert isinstance(first_artifact, ImageArtifact)

    with pytest.raises(DatasetManifestError, match="does not match"):
        next(second_source.iter_loaded(manifest))


def test_public_constructor_type_checks(tmp_path):
    with pytest.raises(TypeError, match="DirectoryDatasetSpec"):
        DirectoryDatasetSource({"type": "directory", "path": tmp_path})

    root = tmp_path / "images"
    save_image(root / "a.png")
    source = DirectoryDatasetSource(make_spec(root))
    manifest = source.build_manifest()

    with pytest.raises(TypeError, match="SampleReference"):
        source.load(manifest.samples[0].path)
    with pytest.raises(TypeError, match="SampleReference"):
        source.verify(manifest.samples[0].path)
    with pytest.raises(TypeError, match="boolean"):
        source.load(manifest.samples[0], verify_integrity=1)
    with pytest.raises(TypeError, match="DatasetManifest"):
        next(source.iter_loaded(manifest.samples))
