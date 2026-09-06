"""Paper-contract constants. Historical candidates are intentionally absent."""

DATASETS = ("TextVQA_Open", "VQAv2_Open", "ScienceQA_MC", "VQAv2_MC")
OPEN_DATASETS = frozenset(("TextVQA_Open", "VQAv2_Open"))
MC_DATASETS = frozenset(("ScienceQA_MC", "VQAv2_MC"))
COMPRESSORS = ("VisionZIP", "VisPruner", "PruMerge", "FlowCut")
LLAVA_BUDGETS = (576, 192, 128, 64, 32, 16)
LLAVA_COMPRESSED_BUDGETS = (192, 128, 64, 32, 16)
CROSS_MODEL_TRAJECTORY = ("Full", "1/3", "2/9", "1/9", "1/18", "1/36")


def practical_fraction(dataset: str) -> float:
    """Return the prespecified cross-model protocol; this is not K_prac."""
    if dataset in OPEN_DATASETS:
        return 1.0 / 9.0
    if dataset in MC_DATASETS:
        return 1.0 / 18.0
    raise ValueError(f"unknown paper dataset: {dataset!r}")
