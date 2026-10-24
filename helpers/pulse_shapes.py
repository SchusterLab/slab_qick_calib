"""
Pulse shapes configurable from the YAML: the readout, the FWM pump and buffer,
and the two f0-g1 reset drives.

Lives in `helpers/` so both the base classes and `helpers/f0g1.py` can import
it without a cycle; `experiments/general/qick_program.py` re-exports it.
"""

# "const" is a hard square pulse; "flat_top" adds gaussian ramps and plays its
# plateau at half a const pulse's amplitude for the same gain.
PULSE_TYPES = ("flat_top", "const")


def pulse_type_from_config(section, key, qi=None):
    """
    Pulse shape `section[key]` (indexed by `qi` if given), validated.

    A missing section or key means "flat_top", so older configs -- and Yuvi,
    which has no device.fwm or device.f0g1 -- keep today's behaviour.
    """
    if section is None or key not in section:
        return "flat_top"
    value = section[key] if qi is None else section[key][qi]
    if value not in PULSE_TYPES:
        raise ValueError(f"{key} must be one of {list(PULSE_TYPES)}, got {value!r}")
    return value


def ramp_length(pulse_type, ramp_sigma, ramp_sigma_inc):
    """Time a pulse runs past its flat `length`: the two ramps for flat_top, none for const."""
    return ramp_sigma * ramp_sigma_inc if pulse_type == "flat_top" else 0.0
