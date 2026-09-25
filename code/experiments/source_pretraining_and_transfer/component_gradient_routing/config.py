"""Prespecified COMPONENT_ROUTING component balancing and gradient-routing arms."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
DOMAINS = ("bearing", "battery", "milling")
JOINT_ARMS = ("joint_component", "joint_obsprivate", "joint_obsprivate_cagrad")
SINGLE_ARMS = tuple(f"single_{d}_{kind}" for d in DOMAINS for kind in ("component", "obsprivate"))
ARMS = JOINT_ARMS + SINGLE_ARMS
FRACTIONS = (.1, .2, 1.)
SEEDS = (42, 43, 44, 45, 46)


def settings(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(
        domains=DOMAINS if arm.startswith("joint_") else (arm.split("_")[1],),
        shared_reconstruction_fraction=.1 if "obsprivate" in arm else 1.,
        aggregation="cagrad" if arm.endswith("cagrad") else "mean",
    )


def control_for(arm, domain):
    kind = "obsprivate" if "obsprivate" in arm else "component"
    return f"single_{domain}_{kind}"
