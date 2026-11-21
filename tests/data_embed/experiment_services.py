"""Adapters from standalone experiments to registered attacks and metrics."""

import numpy as np


def load_services():
    import dwarf.ready_solutions.attack_solutions
    import dwarf.ready_solutions.expertise_solutions


def rgb(image):
    array = np.asarray(image)
    return np.repeat(array[..., None], 3, axis=2) if array.ndim == 2 else array


def attack(image, name, **parameters):
    if name == "clean":
        return image.copy()
    from dwarf.core.attack_orchestrator.attack_core import Attack_Core

    result = Attack_Core.get_attack_class_by_name(name).attack(
        input_image=rgb(image), **parameters
    )
    if isinstance(result, tuple):
        pixels, *stats = result
        return (pixels[..., 0], *stats) if image.ndim == 2 else result
    return result[..., 0] if image.ndim == 2 else result


def metric(name, **parameters):
    from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core

    return Expertise_Core.get_expertise_class_by_name(name).expertise(**parameters)


def image_metric(name, original, distorted, **parameters):
    return metric(name, original_image=rgb(original), distorted_image=rgb(distorted),
                  **parameters)


def bit_metric(name, original, extracted):
    original = np.asarray(original)
    extracted = np.asarray(extracted)
    if np.any(extracted == -1):
        extracted = np.where(extracted == -1, 1 - original, extracted)
    return metric(name, original_bits=original, extracted_bits=extracted)
