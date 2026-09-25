"""Minimum Euclidean correction preserving bearing's own equal-weight contribution."""
import torch


def bearing_floor(direction, bearing, number_domains):
    if number_domains < 1: raise ValueError(number_domains)
    square = bearing @ bearing
    before = bearing @ direction
    target = square / number_domains
    coefficient = (target-before).clamp_min(0) / square.clamp_min(1e-20)
    corrected = direction + coefficient * bearing
    info = dict(triggered=bool(coefficient > 0), coefficient=float(coefficient),
        dot_before=float(before), dot_after=float(bearing @ corrected), target=float(target),
        relative_correction=float((corrected-direction).norm()/direction.norm().clamp_min(1e-20)))
    return corrected, info


def route(rec, dyn, private, values, rho):
    return rec * rho, dyn, private, values
