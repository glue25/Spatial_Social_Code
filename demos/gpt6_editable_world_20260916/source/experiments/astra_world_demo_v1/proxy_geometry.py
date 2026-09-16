"""Geometry-only proxy fitting; no image-derived dimensions are claimed."""
import math


def containment_scale(subject_min, subject_max, target_min, target_max, clearance=0.005):
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError('invalid clearance')
    ratios = []
    for lo, hi, tlo, thi in zip(subject_min, subject_max, target_min, target_max):
        size, room = hi - lo, thi - tlo - 2 * clearance
        if not all(math.isfinite(x) for x in (lo, hi, tlo, thi)) or size <= 0 or room <= 0:
            raise ValueError('nonfinite or degenerate containment geometry')
        ratios.append(room / size)
    if len(ratios) != 3:
        raise ValueError('expected three-dimensional bounds')
    return min(1.0, *ratios)


def containment_center(subject_min, subject_max, target_min, target_max, offset_x=0, clearance=0.005):
    center = []
    for i in range(3):
        half = (subject_max[i] - subject_min[i]) / 2
        low, high = target_min[i] + half + clearance, target_max[i] - half - clearance
        if low > high + 1e-7:
            raise ValueError('subject still exceeds target after fitting')
        desired = (target_min[i] + target_max[i]) / 2 + (offset_x if i == 0 else 0)
        center.append(max(low, min(high, desired)))
    return center
