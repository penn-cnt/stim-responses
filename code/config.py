from pathlib import Path
import numpy as np

_root = Path(__file__).resolve().parent.parent


class CONFIG:
    data_dir    = str(_root / "data")
    fig_dir     = str(_root / "figures")
    results_dir = str(_root / "results")
    tools_dir   = str(_root / "tools")
    dataset_dir = str(_root / "datasets")

    labels_to_remove = ["EKG", "ECG", "RATE", "RR"]

    papez_to_dkt = {
        'Hippocampus': [
            'Left-Hippocampus',
            'Right-Hippocampus',
        ],
        'Parahippocampal Gyrus': [
            'ctx-lh-parahippocampal',
            'ctx-rh-parahippocampal',
            'ctx-lh-entorhinal',
            'ctx-rh-entorhinal',
        ],
        'Cingulate Gyrus (anterior)': [
            'ctx-lh-rostralanteriorcingulate',
            'ctx-rh-rostralanteriorcingulate',
            'ctx-lh-caudalanteriorcingulate',
            'ctx-rh-caudalanteriorcingulate',
        ],
        'Cingulate Gyrus (posterior)': [
            'ctx-lh-posteriorcingulate',
            'ctx-rh-posteriorcingulate',
            'ctx-lh-isthmuscingulate',
            'ctx-rh-isthmuscingulate',
        ],
        'Thalamus (anterior)': [
            'Left-Thalamus-Proper',
            'Right-Thalamus-Proper',
        ],
        'Amygdala (not core Papez)': [
            'Right-Amygdala',
            'Left-Amygdala'
        ]
    }

    limbic_to_dkt = {
        "Hippocampus": [
            "Left-Hippocampus", "Right-Hippocampus"
        ],
        "Amygdala": [
            "Left-Amygdala", "Right-Amygdala"
        ],
        "Parahippocampal Gyrus": [
            "ctx-lh-parahippocampal", "ctx-rh-parahippocampal"
        ],
        "Entorhinal Cortex": [
            "ctx-lh-entorhinal", "ctx-rh-entorhinal"
        ],
        "Cingulate Gyrus (anterior)": [
            "ctx-lh-rostralanteriorcingulate", "ctx-rh-rostralanteriorcingulate",
            "ctx-lh-caudalanteriorcingulate", "ctx-rh-caudalanteriorcingulate"
        ],
        "Cingulate Gyrus (posterior)": [
            "ctx-lh-posteriorcingulate", "ctx-rh-posteriorcingulate",
            "ctx-lh-isthmuscingulate", "ctx-rh-isthmuscingulate"
        ],
        "Temporal Pole": [
            "ctx-lh-temporalpole", "ctx-rh-temporalpole"
        ],
    }

    cortical_to_dkt = {
        "Insular Cortex": [
            "ctx-lh-insula", "ctx-rh-insula"
        ],
        "Anterior Cingulate": [
            "ctx-lh-rostralanteriorcingulate", "ctx-rh-rostralanteriorcingulate",
            "ctx-lh-caudalanteriorcingulate", "ctx-rh-caudalanteriorcingulate"
        ],
        "Posterior Cingulate": [
            "ctx-lh-posteriorcingulate", "ctx-rh-posteriorcingulate",
            "ctx-lh-isthmuscingulate", "ctx-rh-isthmuscingulate"
        ],
        "Prefrontal Cortex": [
            "ctx-lh-superiorfrontal", "ctx-rh-superiorfrontal",
            "ctx-lh-rostralmiddlefrontal", "ctx-rh-rostralmiddlefrontal",
            "ctx-lh-caudalmiddlefrontal", "ctx-rh-caudalmiddlefrontal",
            "ctx-lh-lateralorbitofrontal", "ctx-rh-lateralorbitofrontal",
            "ctx-lh-medialorbitofrontal", "ctx-rh-medialorbitofrontal"
        ],
        "Somatosensory Cortex": [
            "ctx-lh-postcentral", "ctx-rh-postcentral",
            "ctx-lh-paracentral", "ctx-rh-paracentral"
        ],
        "Motor Cortex": [
            "ctx-lh-precentral", "ctx-rh-precentral"
        ]
    }

    thalamic_subcortical_to_dkt = {
        "Thalamus": [
            "Left-Thalamus-Proper", "Right-Thalamus-Proper"
        ],
        "Striatum": [
            "Left-Caudate", "Right-Caudate",
            "Left-Putamen", "Right-Putamen",
            "Left-Accumbens-area", "Right-Accumbens-area"
        ],
        "Amygdala": [
            "Left-Amygdala", "Right-Amygdala"
        ],
        "Hippocampus": [
            "Left-Hippocampus", "Right-Hippocampus"
        ]
    }

    reward_to_dkt = {
        "Ventral Striatum": [
            "Left-Accumbens-area", "Right-Accumbens-area"
        ],
        "Caudate": [
            "Left-Caudate", "Right-Caudate"
        ],
        "Putamen": [
            "Left-Putamen", "Right-Putamen"
        ],
        "Orbitofrontal Cortex": [
            "ctx-lh-lateralorbitofrontal", "ctx-rh-lateralorbitofrontal",
            "ctx-lh-medialorbitofrontal", "ctx-rh-medialorbitofrontal"
        ],
        "Amygdala": [
            "Left-Amygdala", "Right-Amygdala"
        ]
    }

    networks_to_dkt = {
        "papez": papez_to_dkt,
        "limbic": limbic_to_dkt,
        "cortical": cortical_to_dkt,
        "thalamic_subcortical": thalamic_subcortical_to_dkt,
        "reward": reward_to_dkt
    }
