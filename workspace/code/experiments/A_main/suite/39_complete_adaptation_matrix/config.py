from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
E34 = SUITE / "34_shared_degradation_gradient_routing"
E35 = SUITE / "35_lowlabel_conditional_sharing"
E37 = SUITE / "37_bearing_sharing_followup"
E38 = SUITE / "38_automatic_routing_and_confirmation"
DOMAINS = ("bearing", "battery", "milling")
FRACTIONS = (.1, .2, 1.)
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("e37_joint", "e37_single")
MODES = ("frozen", "partial", "full")


def checkpoint(model, domain):
    if model == "e37_joint":
        return E37 / "models/joint_bearingfloor_cagrad/training/encoder.pt"
    if model != "e37_single":
        raise ValueError(model)
    return {
        "bearing": E35 / "models/single_bearing_conditional/training/encoder.pt",
        "battery": E37 / "models/single_battery_conditional/training/encoder.pt",
        "milling": E34 / "models/single_milling_component/training/encoder.pt",
    }[domain]


source_for = checkpoint


def result_path(model, mode, domain, fraction):
    percent = round(fraction * 100)
    if mode == "full":
        if fraction == 1.:
            return E38 / "downstream/baseline" / model / "full" / domain / f"p{percent}/results.json"
        if model == "e37_joint":
            root, arm = E37, "joint_bearingfloor_cagrad"
        else:
            root, arm = {
                "bearing": (E35, "single_bearing_conditional"),
                "battery": (E37, "single_battery_conditional"),
                "milling": (E34, "single_milling_component"),
            }[domain]
        return root / "downstream" / arm / domain / f"p{percent}/results.json"
    if mode == "partial" and domain == "bearing" and fraction == .1:
        return E38 / "downstream/bearing10" / model / "partial/bearing/p10/results.json"
    return HERE / "downstream" / model / mode / domain / f"p{percent}/results.json"
