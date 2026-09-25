from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
COMPONENT_ROUTING = SUITE / "component_gradient_routing"
CONDITIONAL_ROUTING = SUITE / "conditional_gradient_routing"
SELECTED_MODEL = SUITE / "selected_bearing_floor_cagrad"
ADAPTATION_CONTROLS = SUITE / "adaptation_protocol_controls"
DOMAINS = ("bearing", "battery", "milling")
FRACTIONS = (.1, .2, 1.)
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("multi_domain", "single_domain")
MODES = ("frozen", "partial", "full")


def checkpoint(model, domain):
    if model == "multi_domain":
        return SELECTED_MODEL / "models/joint_bearingfloor_cagrad/training/encoder.pt"
    if model != "single_domain":
        raise ValueError(model)
    return {
        "bearing": CONDITIONAL_ROUTING / "models/single_bearing_conditional/training/encoder.pt",
        "battery": SELECTED_MODEL / "models/single_battery_conditional/training/encoder.pt",
        "milling": COMPONENT_ROUTING / "models/single_milling_component/training/encoder.pt",
    }[domain]


source_for = checkpoint


def result_path(model, mode, domain, fraction):
    percent = round(fraction * 100)
    if mode == "full":
        if fraction == 1.:
            return ADAPTATION_CONTROLS / "downstream/baseline" / model / "full" / domain / f"p{percent}/results.json"
        if model == "multi_domain":
            root, arm = SELECTED_MODEL, "joint_bearingfloor_cagrad"
        else:
            root, arm = {
                "bearing": (CONDITIONAL_ROUTING, "single_bearing_conditional"),
                "battery": (SELECTED_MODEL, "single_battery_conditional"),
                "milling": (COMPONENT_ROUTING, "single_milling_component"),
            }[domain]
        return root / "downstream" / arm / domain / f"p{percent}/results.json"
    if mode == "partial" and domain == "bearing" and fraction == .1:
        return ADAPTATION_CONTROLS / "downstream/bearing10" / model / "partial/bearing/p10/results.json"
    return HERE / "downstream" / model / mode / domain / f"p{percent}/results.json"
