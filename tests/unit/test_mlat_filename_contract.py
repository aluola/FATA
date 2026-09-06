from fata.detection.mlatd.analyze_mlat_attack_ablation import parse_key as parse_attack
from fata.detection.mlatd.analyze_mlat_full_grid import parse_key as parse_full


def test_mlat_analyzers_parse_namespaced_and_legacy_attack_files():
    namespaced = (
        "mlat_feat_VisionZIP_TextVQA_Open_"
        "cage_eps2_a0.5_s100_seed0_k64_start0_limit250_seed0.npz"
    )
    legacy = (
        "mlat_feat_FlowCut_ScienceQA_MC_"
        "caa_k32_start0_limit250_seed0.npz"
    )
    for parser in (parse_attack, parse_full):
        assert parser(namespaced)["attack"] == "cage"
        assert parser(namespaced)["attack_contract"] == "_eps2_a0.5_s100_seed0"
        assert parser(legacy)["attack"] == "caa"
        assert parser(legacy)["attack_contract"] == ""
