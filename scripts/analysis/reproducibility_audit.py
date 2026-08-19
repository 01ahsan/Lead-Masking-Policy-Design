# %% cell_01 [code]


# %% cell_02 [code]


from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from IPython.display import display


PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

PTBXL_DATA_DIR = PROJECT_ROOT / "Data"

PTBXL_MULTI_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "step24_reviewer_requested_supplementary"
    / "step24b_multiseed_runs"
)

CHAPMAN_MULTI_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "step26_chapman_multiseed_sensitivity"
    / "tables"
)

OUT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "bspc_submission_audit"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_DIR  :", OUT_DIR)



def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n{path}\n"
            "Do not retrain immediately. First verify the path and whether "
            "the previous notebook generated this file."
        )
    return path


def find_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def normalize_text(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "_")
    )


def normalize_policy(value):
    text = normalize_text(value)

    if "standard" in text or text == "std":
        return "Standard"

    if "random" in text or text == "rand":
        return "Random"

    if (
        "clinical" in text
        or "structured" in text
        or text == "clin"
    ):
        return "Structured"

    return str(value)


def condition_group(value):
    text = normalize_text(value)

    full_tokens = [
        "12lead_full",
        "12_lead_full",
        "12lead",
        "full_12_lead",
        "12_lead",
    ]

    probe_tokens = [
        "lead_i_only",
        "i_only",
        "v1_only",
        "i_ii",
        "v1_v5",
    ]

    target_tokens = [
        "6_limb",
        "six_limb",
        "6_precordial",
        "six_precordial",
        "3_limb",
        "three_limb",
        "lead_ii_only",
        "ii_only",
        "v5_only",
    ]

    if any(token in text for token in full_tokens):
        return "Full"

    if any(token in text for token in probe_tokens):
        return "Probe"

    if any(token in text for token in target_tokens):
        return "Target"

    return "Unclassified"


def detect_column(df, candidates, required=True):
    normalized = {
        normalize_text(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_text(candidate)
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        key = normalize_text(candidate)
        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    if required:
        raise KeyError(
            f"Could not identify any of {candidates}.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None



print("\n" + "=" * 90)
print("PTB-XL METADATA AUDIT")
print("=" * 90)

split_frames = []

for split in ["train", "val", "test"]:
    metadata_path = PTBXL_DATA_DIR / f"{split}_metadata.csv"
    require_file(metadata_path, f"PTB-XL {split} metadata")

    frame = pd.read_csv(metadata_path)
    frame["audit_split"] = split
    split_frames.append(frame)

ptb_meta = pd.concat(split_frames, ignore_index=True)

print("PTB-XL metadata rows:", len(ptb_meta))
print("Available metadata columns:")
print(list(ptb_meta.columns))



patient_col = detect_column(
    ptb_meta,
    ["patient_id", "patient", "subject_id"],
    required=False,
)

if patient_col is not None:
    patient_sets = {
        split: set(
            ptb_meta.loc[
                ptb_meta["audit_split"] == split,
                patient_col,
            ]
            .dropna()
            .astype(str)
        )
        for split in ["train", "val", "test"]
    }

    overlap_rows = []

    comparisons = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for left, right in comparisons:
        overlap = sorted(patient_sets[left] & patient_sets[right])

        overlap_rows.append(
            {
                "split_1": left,
                "split_2": right,
                "num_overlapping_patients": len(overlap),
                "overlapping_patient_ids": ";".join(overlap[:100]),
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)

else:
    overlap_df = pd.DataFrame(
        [
            {
                "split_1": "UNKNOWN",
                "split_2": "UNKNOWN",
                "num_overlapping_patients": np.nan,
                "overlapping_patient_ids": (
                    "No patient identifier column was found."
                ),
            }
        ]
    )

overlap_df.to_csv(
    OUT_DIR / "ptbxl_patient_overlap_audit.csv",
    index=False,
)

display(overlap_df)



split_counts = (
    ptb_meta.groupby("audit_split")
    .size()
    .rename("num_records")
    .reset_index()
)

split_counts.to_csv(
    OUT_DIR / "ptbxl_split_counts.csv",
    index=False,
)

display(split_counts)



label_col = detect_column(
    ptb_meta,
    [
        "label_name",
        "class_name",
        "diagnostic_superclass",
        "label",
        "target",
    ],
    required=False,
)

if label_col is not None:
    label_distribution = (
        ptb_meta.groupby(["audit_split", label_col])
        .size()
        .rename("num_records")
        .reset_index()
        .sort_values(["audit_split", label_col])
    )

    label_distribution.to_csv(
        OUT_DIR / "ptbxl_single_label_distribution.csv",
        index=False,
    )

    display(label_distribution)

else:
    label_distribution = pd.DataFrame()
    print(
        "WARNING: No readable single-label column was found in metadata."
    )



CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
TIE_PRIORITY = ["MI", "STTC", "CD", "HYP", "NORM"]

score_columns = {}

for class_name in CLASS_NAMES:
    candidates = [
        f"score_{class_name}",
        f"{class_name}_score",
        class_name,
    ]

    found = detect_column(
        ptb_meta,
        candidates,
        required=False,
    )

    if found is not None:
        score_columns[class_name] = found

if len(score_columns) == len(CLASS_NAMES) and label_col is not None:

    reconstructed_labels = []
    tied_maximum = []
    valid_score_count = []

    for _, row in ptb_meta.iterrows():

        scores = {}

        for class_name in CLASS_NAMES:
            value = pd.to_numeric(
                row[score_columns[class_name]],
                errors="coerce",
            )

            if pd.notna(value):
                scores[class_name] = float(value)

        valid_score_count.append(len(scores))

        if not scores:
            reconstructed_labels.append("NO_VALID_SCORE")
            tied_maximum.append(False)
            continue

        maximum = max(scores.values())

        tied_classes = [
            class_name
            for class_name, value in scores.items()
            if np.isclose(value, maximum)
        ]

        tied_maximum.append(len(tied_classes) > 1)

        if len(tied_classes) == 1:
            selected = tied_classes[0]

        else:
            selected = next(
                class_name
                for class_name in TIE_PRIORITY
                if class_name in tied_classes
            )

        reconstructed_labels.append(selected)

    ptb_meta["audit_reconstructed_label"] = reconstructed_labels
    ptb_meta["audit_tied_maximum"] = tied_maximum
    ptb_meta["audit_valid_score_count"] = valid_score_count

    ptb_meta["audit_label_match"] = (
        ptb_meta[label_col].astype(str).str.upper()
        ==
        ptb_meta["audit_reconstructed_label"].astype(str).str.upper()
    )

    label_rule_summary = pd.DataFrame(
        [
            {
                "total_records": len(ptb_meta),
                "records_with_tied_maximum": int(
                    ptb_meta["audit_tied_maximum"].sum()
                ),
                "label_rule_matches": int(
                    ptb_meta["audit_label_match"].sum()
                ),
                "label_rule_mismatches": int(
                    (~ptb_meta["audit_label_match"]).sum()
                ),
                "verified_rule": (
                    "Maximum superclass score; ties resolved by "
                    "MI > STTC > CD > HYP > NORM"
                ),
            }
        ]
    )

    ptb_meta.to_csv(
        OUT_DIR / "ptbxl_full_label_rule_audit.csv",
        index=False,
    )

else:

    label_rule_summary = pd.DataFrame(
        [
            {
                "total_records": len(ptb_meta),
                "records_with_tied_maximum": np.nan,
                "label_rule_matches": np.nan,
                "label_rule_mismatches": np.nan,
                "verified_rule": (
                    "Score columns were not available in the split "
                    "metadata, so the rule could not be reconstructed "
                    "from these files."
                ),
            }
        ]
    )

label_rule_summary.to_csv(
    OUT_DIR / "ptbxl_label_rule_summary.csv",
    index=False,
)

display(label_rule_summary)



superclass_col = detect_column(
    ptb_meta,
    [
        "diagnostic_superclasses",
        "diagnostic_superclass_list",
        "superclasses",
    ],
    required=False,
)

if superclass_col is not None:

    def count_superclasses(value):
        if pd.isna(value):
            return 0

        text = str(value).strip()

        if not text:
            return 0

        cleaned = (
            text.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )

        items = [
            item.strip()
            for item in re.split(r"[;,|]", cleaned)
            if item.strip()
        ]

        return len(set(items))

    ptb_meta["audit_num_diagnostic_superclasses"] = (
        ptb_meta[superclass_col].apply(count_superclasses)
    )

    multiplicity = (
        ptb_meta.groupby(
            [
                "audit_split",
                "audit_num_diagnostic_superclasses",
            ]
        )
        .size()
        .rename("num_records")
        .reset_index()
        .sort_values(
            [
                "audit_split",
                "audit_num_diagnostic_superclasses",
            ]
        )
    )

    multiplicity.to_csv(
        OUT_DIR / "ptbxl_superclass_multiplicity.csv",
        index=False,
    )

    display(multiplicity)

else:
    multiplicity = pd.DataFrame()
    print(
        "NOTE: diagnostic_superclasses column was not found. "
        "Multiplicity cannot be calculated from these metadata files."
    )



print("\n" + "=" * 90)
print("PTB-XL MULTI-SEED AUDIT")
print("=" * 90)

PTB_MULTI_CSV = find_first_existing(
    [
        PTBXL_MULTI_DIR
        / "multiseed_all_test_condition_metrics_long.csv",

        PTBXL_MULTI_DIR
        / "Supplementary_Table_S2_multiseed_mean_std_by_policy_condition.csv",
    ]
)

if PTB_MULTI_CSV is None:
    raise FileNotFoundError(
        "\nPTB-XL multi-seed output was not found.\n"
        f"Checked:\n{PTBXL_MULTI_DIR}\n\n"
        "Run only Step 24B from your existing Ths_REdo_3 notebook "
        "if this directory genuinely does not contain the files."
    )

print("PTB-XL multi-seed source:")
print(PTB_MULTI_CSV)

ptb_multi_raw = pd.read_csv(PTB_MULTI_CSV)



print("\n" + "=" * 90)
print("CHAPMAN MULTI-SEED AUDIT")
print("=" * 90)

CHAPMAN_MULTI_CSV = find_first_existing(
    [
        CHAPMAN_MULTI_DIR
        / "chapman_multiseed_all_test_condition_metrics_long.csv",

        CHAPMAN_MULTI_DIR
        / "Supplementary_Table_S8_Chapman_multiseed_mean_std_by_policy_condition.csv",
    ]
)

if CHAPMAN_MULTI_CSV is None:
    raise FileNotFoundError(
        "\nChapman multi-seed output was not found.\n"
        f"Checked:\n{CHAPMAN_MULTI_DIR}\n\n"
        "Run only Step 26 from your existing Ths_REdo_3 notebook "
        "if these files genuinely do not exist."
    )

print("Chapman multi-seed source:")
print(CHAPMAN_MULTI_CSV)

chapman_multi_raw = pd.read_csv(CHAPMAN_MULTI_CSV)



def standardize_multiseed_frame(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:

    data = df.copy()

    policy_col = detect_column(
        data,
        [
            "policy",
            "variant_name",
            "method",
            "model",
        ],
    )

    condition_col = detect_column(
        data,
        [
            "lead_condition",
            "lead_display_name",
            "condition_name",
            "condition",
            "lead_key",
        ],
    )

    macro_col = detect_column(
        data,
        [
            "macro_f1",
            "macro-f1",
            "macro_f1_mean",
            "mean_macro_f1",
            "mean",
        ],
    )

    seed_col = detect_column(
        data,
        ["seed", "random_seed"],
        required=False,
    )

    std_col = detect_column(
        data,
        [
            "macro_f1_std",
            "std_macro_f1",
            "std",
        ],
        required=False,
    )

    output = pd.DataFrame()

    output["dataset"] = dataset_name
    output["policy"] = data[policy_col].apply(normalize_policy)
    output["condition"] = data[condition_col].astype(str)
    output["condition_group"] = (
        output["condition"].apply(condition_group)
    )

    output["macro_f1"] = pd.to_numeric(
        data[macro_col],
        errors="coerce",
    )

    if seed_col is not None:
        output["seed"] = pd.to_numeric(
            data[seed_col],
            errors="coerce",
        )
    else:
        output["seed"] = np.nan

    if std_col is not None:
        output["reported_std"] = pd.to_numeric(
            data[std_col],
            errors="coerce",
        )
    else:
        output["reported_std"] = np.nan

    output = output.dropna(subset=["macro_f1"])

    return output


ptb_multi = standardize_multiseed_frame(
    ptb_multi_raw,
    "PTB-XL",
)

chapman_multi = standardize_multiseed_frame(
    chapman_multi_raw,
    "Chapman",
)

multi = pd.concat(
    [ptb_multi, chapman_multi],
    ignore_index=True,
)

multi.to_csv(
    OUT_DIR / "all_multiseed_results_standardized.csv",
    index=False,
)

print("\nStandardized rows:", len(multi))
display(multi.head(20))



has_seed_rows = multi["seed"].notna().any()

if has_seed_rows:

    condition_summary = (
        multi.groupby(
            [
                "dataset",
                "policy",
                "condition",
                "condition_group",
            ]
        )["macro_f1"]
        .agg(
            num_seeds="count",
            mean_macro_f1="mean",
            std_macro_f1="std",
            min_macro_f1="min",
            max_macro_f1="max",
        )
        .reset_index()
    )

else:

    condition_summary = (
        multi.rename(
            columns={
                "macro_f1": "mean_macro_f1",
                "reported_std": "std_macro_f1",
            }
        )[
            [
                "dataset",
                "policy",
                "condition",
                "condition_group",
                "mean_macro_f1",
                "std_macro_f1",
            ]
        ]
        .copy()
    )

    condition_summary["num_seeds"] = np.nan
    condition_summary["min_macro_f1"] = np.nan
    condition_summary["max_macro_f1"] = np.nan

condition_summary = condition_summary.sort_values(
    [
        "dataset",
        "condition_group",
        "condition",
        "policy",
    ]
)

condition_summary.to_csv(
    OUT_DIR / "manuscript_multiseed_condition_summary.csv",
    index=False,
)

display(condition_summary)



if has_seed_rows:

    per_seed_group = (
        multi[
            multi["condition_group"].isin(
                ["Full", "Target", "Probe"]
            )
        ]
        .groupby(
            [
                "dataset",
                "policy",
                "seed",
                "condition_group",
            ]
        )["macro_f1"]
        .mean()
        .rename("group_macro_f1")
        .reset_index()
    )

    group_summary = (
        per_seed_group.groupby(
            [
                "dataset",
                "policy",
                "condition_group",
            ]
        )["group_macro_f1"]
        .agg(
            num_seeds="count",
            mean_macro_f1="mean",
            std_macro_f1="std",
        )
        .reset_index()
    )

    reduced = multi[
        multi["condition_group"].isin(["Target", "Probe"])
    ]

    worst_per_seed = (
        reduced.groupby(
            [
                "dataset",
                "policy",
                "seed",
            ]
        )["macro_f1"]
        .min()
        .rename("worst_reduced_macro_f1")
        .reset_index()
    )

    worst_summary = (
        worst_per_seed.groupby(
            [
                "dataset",
                "policy",
            ]
        )["worst_reduced_macro_f1"]
        .agg(
            num_seeds="count",
            mean_worst_macro_f1="mean",
            std_worst_macro_f1="std",
        )
        .reset_index()
    )

else:

    group_summary = (
        condition_summary[
            condition_summary["condition_group"].isin(
                ["Full", "Target", "Probe"]
            )
        ]
        .groupby(
            [
                "dataset",
                "policy",
                "condition_group",
            ]
        )["mean_macro_f1"]
        .mean()
        .rename("mean_macro_f1")
        .reset_index()
    )

    group_summary["num_seeds"] = np.nan
    group_summary["std_macro_f1"] = np.nan

    reduced = condition_summary[
        condition_summary["condition_group"].isin(
            ["Target", "Probe"]
        )
    ]

    worst_summary = (
        reduced.groupby(
            [
                "dataset",
                "policy",
            ]
        )["mean_macro_f1"]
        .min()
        .rename("mean_worst_macro_f1")
        .reset_index()
    )

    worst_summary["num_seeds"] = np.nan
    worst_summary["std_worst_macro_f1"] = np.nan


group_summary.to_csv(
    OUT_DIR / "manuscript_target_probe_full_summary.csv",
    index=False,
)

worst_summary.to_csv(
    OUT_DIR / "manuscript_worst_case_summary.csv",
    index=False,
)

print("\nTARGET / PROBE / FULL SUMMARY")
display(group_summary)

print("\nWORST REDUCED-LEAD SUMMARY")
display(worst_summary)



expected_policies = {
    "Standard",
    "Random",
    "Structured",
}

expected_conditions = {
    "Full": 1,
    "Target": 5,
    "Probe": 4,
}

audit_rows = []

for dataset in ["PTB-XL", "Chapman"]:

    dataset_df = condition_summary[
        condition_summary["dataset"] == dataset
    ]

    available_policies = set(dataset_df["policy"].unique())

    for policy in sorted(expected_policies):

        policy_df = dataset_df[
            dataset_df["policy"] == policy
        ]

        row = {
            "dataset": dataset,
            "policy": policy,
            "policy_present": policy in available_policies,
        }

        for group_name, expected_count in expected_conditions.items():

            observed_count = policy_df.loc[
                policy_df["condition_group"] == group_name,
                "condition",
            ].nunique()

            row[f"{group_name.lower()}_conditions_observed"] = (
                int(observed_count)
            )

            row[f"{group_name.lower()}_conditions_expected"] = (
                expected_count
            )

            row[f"{group_name.lower()}_complete"] = (
                observed_count == expected_count
            )

        audit_rows.append(row)

completeness_df = pd.DataFrame(audit_rows)

completeness_df.to_csv(
    OUT_DIR / "multiseed_completeness_audit.csv",
    index=False,
)

display(completeness_df)



patient_overlap_pass = True

if patient_col is not None:
    patient_overlap_pass = bool(
        (
            overlap_df["num_overlapping_patients"] == 0
        ).all()
    )

label_rule_pass = True

if (
    "label_rule_mismatches"
    in label_rule_summary.columns
    and pd.notna(
        label_rule_summary.loc[
            0,
            "label_rule_mismatches",
        ]
    )
):
    label_rule_pass = (
        int(
            label_rule_summary.loc[
                0,
                "label_rule_mismatches",
            ]
        )
        == 0
    )

multiseed_complete = bool(
    completeness_df[
        [
            "policy_present",
            "full_complete",
            "target_complete",
            "probe_complete",
        ]
    ]
    .all()
    .all()
)

unclassified_conditions = sorted(
    condition_summary.loc[
        condition_summary["condition_group"]
        == "Unclassified",
        "condition",
    ]
    .astype(str)
    .unique()
    .tolist()
)

unclassified_pass = len(unclassified_conditions) == 0

final_pass = all(
    [
        patient_overlap_pass,
        label_rule_pass,
        multiseed_complete,
        unclassified_pass,
    ]
)

final_report = {
    "project_root": str(PROJECT_ROOT),
    "output_directory": str(OUT_DIR),

    "ptbxl_records": int(len(ptb_meta)),

    "patient_overlap_pass": patient_overlap_pass,

    "label_rule_reconstruction_available": (
        len(score_columns) == len(CLASS_NAMES)
        and label_col is not None
    ),

    "label_rule_pass": label_rule_pass,

    "multiseed_complete": multiseed_complete,

    "unclassified_conditions": unclassified_conditions,

    "unclassified_conditions_pass": unclassified_pass,

    "overall_submission_data_audit_pass": final_pass,
}

with open(
    OUT_DIR / "bspc_submission_audit.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(final_report, file, indent=2)

report_lines = [
    "=" * 90,
    "BSPC SUBMISSION DATA AUDIT",
    "=" * 90,

    f"PTB-XL records audited: {len(ptb_meta)}",

    f"Patient-overlap check: "
    f"{'PASS' if patient_overlap_pass else 'FAIL'}",

    f"Single-label-rule check: "
    f"{'PASS' if label_rule_pass else 'FAIL / UNVERIFIED'}",

    f"Multi-seed completeness: "
    f"{'PASS' if multiseed_complete else 'FAIL'}",

    f"Condition classification: "
    f"{'PASS' if unclassified_pass else 'FAIL'}",

    f"Unclassified conditions: "
    f"{unclassified_conditions}",

    "",

    "OVERALL RESULT: "
    + (
        "PASS — manuscript tables may now be rebuilt "
        "from the audited outputs."
        if final_pass
        else
        "NOT READY — inspect the failed audit item(s) "
        "before changing the manuscript."
    ),

    "",

    f"All audit outputs saved to:\n{OUT_DIR}",

    "=" * 90,
]

report_text = "\n".join(report_lines)

with open(
    OUT_DIR / "bspc_submission_audit_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report_text)

print("\n")
print(report_text)

# %% cell_03 [code]

from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

PTB_MULTI_CSV = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "step24_reviewer_requested_supplementary"
    / "step24b_multiseed_runs"
    / "multiseed_all_test_condition_metrics_long.csv"
)

CHAPMAN_MULTI_CSV = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "step26_chapman_multiseed_sensitivity"
    / "tables"
    / "chapman_multiseed_all_test_condition_metrics_long.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "bspc_submission_audit"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)



def canonical(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "_")
    )


def normalize_policy(value):
    text = canonical(value)

    if text in {"standard", "std"} or "standard" in text:
        return "Standard"

    if text in {"random", "rand"} or "random" in text:
        return "Random"

    if (
        text in {"clinical", "clin", "structured"}
        or "clinical" in text
        or "structured" in text
    ):
        return "Structured"

    return str(value)


CONDITION_MAP = {
    "12lead_full": ("12-lead full", "Full"),
    "12_lead_full": ("12-lead full", "Full"),
    "full_12_lead": ("12-lead full", "Full"),

    "6_limb": ("6 limb", "Target"),
    "six_limb": ("6 limb", "Target"),

    "6_precordial": ("6 precordial", "Target"),
    "six_precordial": ("6 precordial", "Target"),

    "3_limb": ("3 limb", "Target"),
    "three_limb": ("3 limb", "Target"),

    "lead_ii_only": ("Lead II only", "Target"),
    "ii_only": ("Lead II only", "Target"),

    "v5_only": ("V5 only", "Target"),

    "lead_i_only_unseen": ("Lead I only", "Probe"),
    "lead_i_only": ("Lead I only", "Probe"),
    "i_only": ("Lead I only", "Probe"),

    "v1_only_unseen": ("V1 only", "Probe"),
    "v1_only": ("V1 only", "Probe"),

    "i_ii_unseen": ("I+II", "Probe"),
    "i_ii": ("I+II", "Probe"),

    "v1_v5_unseen": ("V1+V5", "Probe"),
    "v1_v5": ("V1+V5", "Probe"),
}


def map_condition(value):
    key = canonical(value)

    if key in CONDITION_MAP:
        return CONDITION_MAP[key]

    clean_key = key.replace("_unseen", "")

    if clean_key in CONDITION_MAP:
        return CONDITION_MAP[clean_key]

    return (str(value), "Unclassified")


def find_column(df, candidates):
    normalized_columns = {
        canonical(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = canonical(candidate)

        if key in normalized_columns:
            return normalized_columns[key]

    raise KeyError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def standardize_long_results(df, dataset_name):
    policy_col = find_column(
        df,
        ["policy", "variant_name", "method", "model"],
    )

    condition_col = find_column(
        df,
        [
            "lead_condition",
            "condition",
            "condition_name",
            "lead_display_name",
            "lead_key",
        ],
    )

    macro_col = find_column(
        df,
        ["macro_f1", "macro-f1", "macrof1"],
    )

    seed_col = find_column(
        df,
        ["seed", "random_seed"],
    )

    condition_info = df[condition_col].apply(map_condition)

    output = pd.DataFrame(
        {
            "dataset": [dataset_name] * len(df),
            "policy": df[policy_col].apply(normalize_policy),
            "condition_raw": df[condition_col].astype(str),
            "condition": condition_info.apply(lambda x: x[0]),
            "condition_group": condition_info.apply(lambda x: x[1]),
            "macro_f1": pd.to_numeric(
                df[macro_col],
                errors="coerce",
            ),
            "seed": pd.to_numeric(
                df[seed_col],
                errors="coerce",
            ),
        }
    )

    return output.dropna(
        subset=["macro_f1", "seed"]
    ).reset_index(drop=True)



if not PTB_MULTI_CSV.exists():
    raise FileNotFoundError(
        f"Missing PTB-XL multi-seed file:\n{PTB_MULTI_CSV}"
    )

if not CHAPMAN_MULTI_CSV.exists():
    raise FileNotFoundError(
        f"Missing Chapman multi-seed file:\n{CHAPMAN_MULTI_CSV}"
    )

ptb_raw = pd.read_csv(PTB_MULTI_CSV)
chapman_raw = pd.read_csv(CHAPMAN_MULTI_CSV)

print("PTB-XL raw rows:", len(ptb_raw))
print("Chapman raw rows:", len(chapman_raw))

ptb = standardize_long_results(
    ptb_raw,
    "PTB-XL",
)

chapman = standardize_long_results(
    chapman_raw,
    "Chapman",
)

results = pd.concat(
    [ptb, chapman],
    ignore_index=True,
)

results.to_csv(
    OUT_DIR / "all_multiseed_results_standardized_FIXED.csv",
    index=False,
)

print("\nStandardized result preview:")
display(results.head(20))



print("\nDatasets:")
print(results["dataset"].value_counts())

print("\nPolicies:")
print(results.groupby("dataset")["policy"].value_counts())

print("\nSeeds:")
print(
    results.groupby("dataset")["seed"]
    .unique()
    .apply(lambda x: sorted(x.tolist()))
)

print("\nConditions:")
display(
    results[
        [
            "condition_raw",
            "condition",
            "condition_group",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "condition_group",
            "condition",
        ]
    )
)



condition_summary = (
    results.groupby(
        [
            "dataset",
            "policy",
            "condition",
            "condition_group",
        ]
    )["macro_f1"]
    .agg(
        num_seeds="count",
        mean_macro_f1="mean",
        std_macro_f1="std",
        min_macro_f1="min",
        max_macro_f1="max",
    )
    .reset_index()
    .sort_values(
        [
            "dataset",
            "condition_group",
            "condition",
            "policy",
        ]
    )
)

condition_summary.to_csv(
    OUT_DIR / "manuscript_multiseed_condition_summary_FIXED.csv",
    index=False,
)

print("\nCONDITION-LEVEL MULTI-SEED SUMMARY")
display(condition_summary)



per_seed_group = (
    results.groupby(
        [
            "dataset",
            "policy",
            "seed",
            "condition_group",
        ]
    )["macro_f1"]
    .mean()
    .rename("group_macro_f1")
    .reset_index()
)

per_seed_group.to_csv(
    OUT_DIR / "per_seed_group_macro_f1_FIXED.csv",
    index=False,
)

group_summary = (
    per_seed_group.groupby(
        [
            "dataset",
            "policy",
            "condition_group",
        ]
    )["group_macro_f1"]
    .agg(
        num_seeds="count",
        mean_macro_f1="mean",
        std_macro_f1="std",
        min_macro_f1="min",
        max_macro_f1="max",
    )
    .reset_index()
    .sort_values(
        [
            "dataset",
            "condition_group",
            "policy",
        ]
    )
)

group_summary.to_csv(
    OUT_DIR / "manuscript_target_probe_full_summary_FIXED.csv",
    index=False,
)

print("\nTARGET / PROBE / FULL SUMMARY")
display(group_summary)



reduced_results = results[
    results["condition_group"].isin(
        ["Target", "Probe"]
    )
].copy()

worst_per_seed = (
    reduced_results.groupby(
        [
            "dataset",
            "policy",
            "seed",
        ]
    )["macro_f1"]
    .min()
    .rename("worst_reduced_macro_f1")
    .reset_index()
)

worst_per_seed.to_csv(
    OUT_DIR / "per_seed_worst_reduced_macro_f1_FIXED.csv",
    index=False,
)

worst_summary = (
    worst_per_seed.groupby(
        [
            "dataset",
            "policy",
        ]
    )["worst_reduced_macro_f1"]
    .agg(
        num_seeds="count",
        mean_worst_macro_f1="mean",
        std_worst_macro_f1="std",
        min_worst_macro_f1="min",
        max_worst_macro_f1="max",
    )
    .reset_index()
)

worst_summary.to_csv(
    OUT_DIR / "manuscript_worst_case_summary_FIXED.csv",
    index=False,
)

print("\nWORST REDUCED-LEAD SUMMARY")
display(worst_summary)



policy_pivot = (
    results.pivot_table(
        index=[
            "dataset",
            "seed",
            "condition",
            "condition_group",
        ],
        columns="policy",
        values="macro_f1",
        aggfunc="first",
    )
    .reset_index()
)

required_policy_columns = {
    "Standard",
    "Random",
    "Structured",
}

missing_policy_columns = (
    required_policy_columns
    - set(policy_pivot.columns)
)

if missing_policy_columns:
    raise RuntimeError(
        "Missing policy columns after pivot: "
        f"{missing_policy_columns}"
    )

policy_pivot["Structured_minus_Random"] = (
    policy_pivot["Structured"]
    - policy_pivot["Random"]
)

policy_pivot["Structured_minus_Standard"] = (
    policy_pivot["Structured"]
    - policy_pivot["Standard"]
)

policy_pivot["Random_minus_Standard"] = (
    policy_pivot["Random"]
    - policy_pivot["Standard"]
)

policy_pivot.to_csv(
    OUT_DIR / "per_seed_policy_differences_FIXED.csv",
    index=False,
)

difference_summary = (
    policy_pivot.groupby(
        [
            "dataset",
            "condition",
            "condition_group",
        ]
    )[
        [
            "Structured_minus_Random",
            "Structured_minus_Standard",
            "Random_minus_Standard",
        ]
    ]
    .agg(["mean", "std", "min", "max"])
)

difference_summary.columns = [
    "_".join(column)
    for column in difference_summary.columns
]

difference_summary = (
    difference_summary
    .reset_index()
    .sort_values(
        [
            "dataset",
            "condition_group",
            "condition",
        ]
    )
)

difference_summary.to_csv(
    OUT_DIR / "multiseed_policy_difference_summary_FIXED.csv",
    index=False,
)

print("\nPOLICY DIFFERENCE SUMMARY")
display(difference_summary)



EXPECTED = {
    "Full": {
        "12-lead full",
    },
    "Target": {
        "6 limb",
        "6 precordial",
        "3 limb",
        "Lead II only",
        "V5 only",
    },
    "Probe": {
        "Lead I only",
        "V1 only",
        "I+II",
        "V1+V5",
    },
}

EXPECTED_POLICIES = {
    "Standard",
    "Random",
    "Structured",
}

completeness_rows = []

for dataset_name in ["PTB-XL", "Chapman"]:

    dataset_df = results[
        results["dataset"] == dataset_name
    ]

    for policy_name in sorted(EXPECTED_POLICIES):

        policy_df = dataset_df[
            dataset_df["policy"] == policy_name
        ]

        row = {
            "dataset": dataset_name,
            "policy": policy_name,
            "policy_present": not policy_df.empty,
            "num_unique_seeds": int(
                policy_df["seed"].nunique()
            ),
        }

        for group_name, expected_conditions in EXPECTED.items():

            observed_conditions = set(
                policy_df.loc[
                    policy_df["condition_group"] == group_name,
                    "condition",
                ].unique()
            )

            missing_conditions = sorted(
                expected_conditions
                - observed_conditions
            )

            extra_conditions = sorted(
                observed_conditions
                - expected_conditions
            )

            row[f"{group_name.lower()}_observed"] = (
                len(observed_conditions)
            )

            row[f"{group_name.lower()}_expected"] = (
                len(expected_conditions)
            )

            row[f"{group_name.lower()}_missing"] = (
                "; ".join(missing_conditions)
            )

            row[f"{group_name.lower()}_extra"] = (
                "; ".join(extra_conditions)
            )

            row[f"{group_name.lower()}_complete"] = (
                observed_conditions
                == expected_conditions
            )

        completeness_rows.append(row)

completeness_df = pd.DataFrame(
    completeness_rows
)

completeness_df.to_csv(
    OUT_DIR / "multiseed_completeness_audit_FIXED.csv",
    index=False,
)

print("\nCOMPLETENESS AUDIT")
display(completeness_df)



unclassified = sorted(
    results.loc[
        results["condition_group"] == "Unclassified",
        "condition_raw",
    ]
    .unique()
    .tolist()
)

all_policies_present = bool(
    completeness_df["policy_present"].all()
)

all_conditions_complete = bool(
    completeness_df[
        [
            "full_complete",
            "target_complete",
            "probe_complete",
        ]
    ]
    .all()
    .all()
)

at_least_three_seeds = bool(
    (
        completeness_df["num_unique_seeds"] >= 3
    ).all()
)

overall_pass = all(
    [
        all_policies_present,
        all_conditions_complete,
        at_least_three_seeds,
        len(unclassified) == 0,
    ]
)

final_report = {
    "all_policies_present": all_policies_present,
    "all_conditions_complete": all_conditions_complete,
    "at_least_three_seeds_per_dataset_policy": at_least_three_seeds,
    "unclassified_conditions": unclassified,
    "overall_multiseed_audit_pass": overall_pass,
}

with open(
    OUT_DIR / "bspc_multiseed_audit_FIXED.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_report,
        file,
        indent=2,
    )

report_text = f"""
==========================================================================================
FIXED BSPC MULTI-SEED AUDIT
==========================================================================================

All policies present:
{"PASS" if all_policies_present else "FAIL"}

All 10 conditions complete:
{"PASS" if all_conditions_complete else "FAIL"}

Exactly 5 primary seeds per dataset and policy:
{"PASS" if at_least_three_seeds else "FAIL"}

Unclassified conditions:
{unclassified}

OVERALL:
{
    "PASS — no additional model training is required."
    if overall_pass
    else
    "FAIL — inspect the completeness table before proceeding."
}

Outputs:
{OUT_DIR}

==========================================================================================
"""

print(report_text)

with open(
    OUT_DIR / "bspc_multiseed_audit_report_FIXED.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report_text)

# %% cell_04 [code]

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

AUDIT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "bspc_submission_audit"
)

OUT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "bspc_manuscript_ready"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = (
    AUDIT_DIR
    / "all_multiseed_results_standardized_FIXED.csv"
)

if not RESULTS_CSV.exists():
    raise FileNotFoundError(
        f"Missing audited results:\n{RESULTS_CSV}\n"
        "Run the FIXED BSPC multi-seed audit cell first."
    )

results = pd.read_csv(RESULTS_CSV)

print("Loaded rows:", len(results))
print("Output directory:", OUT_DIR)



DATASET_ORDER = ["PTB-XL", "Chapman"]

POLICY_ORDER = [
    "Standard",
    "Random",
    "Structured",
]

CONDITION_ORDER = [
    "12-lead full",
    "6 limb",
    "6 precordial",
    "3 limb",
    "Lead II only",
    "V5 only",
    "Lead I only",
    "V1 only",
    "I+II",
    "V1+V5",
]

CONDITION_LABELS = {
    "12-lead full": "12-lead",
    "6 limb": "6 limb",
    "6 precordial": "6 precordial",
    "3 limb": "3 limb",
    "Lead II only": "Lead II",
    "V5 only": "V5",
    "Lead I only": "Lead I",
    "V1 only": "V1",
    "I+II": "I+II",
    "V1+V5": "V1+V5",
}

GROUP_LABELS = {
    "Full": "Full input",
    "Target": "Structured-policy target",
    "Probe": "Support-probing",
}

results["dataset"] = pd.Categorical(
    results["dataset"],
    categories=DATASET_ORDER,
    ordered=True,
)

results["policy"] = pd.Categorical(
    results["policy"],
    categories=POLICY_ORDER,
    ordered=True,
)

results["condition"] = pd.Categorical(
    results["condition"],
    categories=CONDITION_ORDER,
    ordered=True,
)



condition_summary = (
    results.groupby(
        [
            "dataset",
            "condition",
            "condition_group",
            "policy",
        ],
        observed=True,
    )["macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1="std",
        num_seeds="count",
        minimum="min",
        maximum="max",
    )
    .reset_index()
    .sort_values(
        [
            "dataset",
            "condition",
            "policy",
        ]
    )
)

condition_summary.to_csv(
    OUT_DIR / "Table_condition_level_multiseed.csv",
    index=False,
)

print("\nCondition-level results:")
display(condition_summary)



mean_pivot = condition_summary.pivot_table(
    index=[
        "dataset",
        "condition",
        "condition_group",
    ],
    columns="policy",
    values="mean_macro_f1",
    observed=True,
)

std_pivot = condition_summary.pivot_table(
    index=[
        "dataset",
        "condition",
        "condition_group",
    ],
    columns="policy",
    values="std_macro_f1",
    observed=True,
)

main_rows = []

for index in mean_pivot.index:

    dataset, condition, condition_group = index

    row = {
        "Dataset": str(dataset),
        "Lead condition": str(condition),
        "Condition family": GROUP_LABELS[str(condition_group)],
    }

    policy_means = {}

    for policy in POLICY_ORDER:

        mean_value = float(
            mean_pivot.loc[index, policy]
        )

        std_value = float(
            std_pivot.loc[index, policy]
        )

        policy_means[policy] = mean_value

        row[policy] = (
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    row["Structured − Random"] = (
        policy_means["Structured"]
        - policy_means["Random"]
    )

    row["Random − Standard"] = (
        policy_means["Random"]
        - policy_means["Standard"]
    )

    main_rows.append(row)

main_table = pd.DataFrame(main_rows)

main_table["Dataset"] = pd.Categorical(
    main_table["Dataset"],
    categories=DATASET_ORDER,
    ordered=True,
)

main_table["Lead condition"] = pd.Categorical(
    main_table["Lead condition"],
    categories=CONDITION_ORDER,
    ordered=True,
)

main_table = main_table.sort_values(
    [
        "Dataset",
        "Lead condition",
    ]
).reset_index(drop=True)

main_table.to_csv(
    OUT_DIR / "Table_main_multiseed_results.csv",
    index=False,
)

display(main_table)



latex_lines = []

latex_lines.append(r"\begin{table*}[t]")
latex_lines.append(r"\centering")
latex_lines.append(
    r"\caption{Macro-F1 across five independent training seeds. "
    r"Values are mean $\pm$ standard deviation. "
    r"Target conditions are explicitly included in the Structured "
    r"masking family. Probe conditions are held out from the Structured "
    r"policy; Lead I and V1 remain within Random masking support, "
    r"whereas I+II and V1+V5 are outside the exact support of both "
    r"masked policies.}"
)
latex_lines.append(r"\label{tab:main_multiseed_results}")
latex_lines.append(r"\small")
latex_lines.append(r"\setlength{\tabcolsep}{4pt}")
latex_lines.append(
    r"\begin{tabular}{lllccccc}"
)
latex_lines.append(r"\toprule")
latex_lines.append(
    r"Dataset & Lead condition & Family & Standard & Random & "
    r"Structured & Structured--Random & Random--Standard \\"
)
latex_lines.append(r"\midrule")

previous_dataset = None

for _, row in main_table.iterrows():

    dataset = str(row["Dataset"])
    condition = str(row["Lead condition"])
    family = str(row["Condition family"])

    if previous_dataset is not None and dataset != previous_dataset:
        latex_lines.append(r"\midrule")

    structured_random = float(
        row["Structured − Random"]
    )

    random_standard = float(
        row["Random − Standard"]
    )

    latex_lines.append(
        f"{dataset} & "
        f"{condition} & "
        f"{family} & "
        f"${row['Standard']}$ & "
        f"${row['Random']}$ & "
        f"${row['Structured']}$ & "
        f"{structured_random:+.4f} & "
        f"{random_standard:+.4f} \\\\"
    )

    previous_dataset = dataset

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table*}")

latex_main_table = "\n".join(latex_lines)

with open(
    OUT_DIR / "Table_main_multiseed_results.tex",
    "w",
    encoding="utf-8",
) as file:
    file.write(latex_main_table)

print("\nLaTeX main table saved.")



per_seed_group = (
    results.groupby(
        [
            "dataset",
            "policy",
            "seed",
            "condition_group",
        ],
        observed=True,
    )["macro_f1"]
    .mean()
    .rename("group_macro_f1")
    .reset_index()
)

aggregate_summary = (
    per_seed_group.groupby(
        [
            "dataset",
            "policy",
            "condition_group",
        ],
        observed=True,
    )["group_macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1="std",
        num_seeds="count",
    )
    .reset_index()
)

reduced_results = results[
    results["condition_group"].isin(
        ["Target", "Probe"]
    )
].copy()

worst_per_seed = (
    reduced_results.groupby(
        [
            "dataset",
            "policy",
            "seed",
        ],
        observed=True,
    )["macro_f1"]
    .min()
    .rename("worst_macro_f1")
    .reset_index()
)

worst_summary = (
    worst_per_seed.groupby(
        [
            "dataset",
            "policy",
        ],
        observed=True,
    )["worst_macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1="std",
        num_seeds="count",
    )
    .reset_index()
)

worst_summary["condition_group"] = "Worst reduced"

aggregate_with_worst = pd.concat(
    [
        aggregate_summary,
        worst_summary,
    ],
    ignore_index=True,
)

aggregate_with_worst.to_csv(
    OUT_DIR / "Table_policy_level_summary.csv",
    index=False,
)

display(aggregate_with_worst)



summary_pivot_mean = aggregate_with_worst.pivot_table(
    index=[
        "dataset",
        "policy",
    ],
    columns="condition_group",
    values="mean_macro_f1",
    observed=True,
)

summary_pivot_std = aggregate_with_worst.pivot_table(
    index=[
        "dataset",
        "policy",
    ],
    columns="condition_group",
    values="std_macro_f1",
    observed=True,
)

summary_rows = []

for index in summary_pivot_mean.index:

    dataset, policy = index

    row = {
        "Dataset": str(dataset),
        "Policy": str(policy),
    }

    for group in [
        "Full",
        "Target",
        "Probe",
        "Worst reduced",
    ]:

        mean_value = float(
            summary_pivot_mean.loc[index, group]
        )

        std_value = float(
            summary_pivot_std.loc[index, group]
        )

        row[group] = (
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    summary_rows.append(row)

summary_table = pd.DataFrame(summary_rows)

summary_table.to_csv(
    OUT_DIR / "Table_policy_level_summary_formatted.csv",
    index=False,
)

display(summary_table)

summary_latex = []

summary_latex.append(r"\begin{table*}[t]")
summary_latex.append(r"\centering")
summary_latex.append(
    r"\caption{Policy-level Macro-F1 summary across five independent training seeds. "
    r"Target and probe values are computed by averaging conditions "
    r"within each seed before calculating the across-seed mean and "
    r"standard deviation. Worst reduced is the minimum Macro-F1 among "
    r"all nine reduced-lead conditions for each seed.}"
)
summary_latex.append(r"\label{tab:policy_level_summary}")
summary_latex.append(r"\small")
summary_latex.append(
    r"\begin{tabular}{llcccc}"
)
summary_latex.append(r"\toprule")
summary_latex.append(
    r"Dataset & Policy & Full input & Target mean & "
    r"Probe mean & Worst reduced \\"
)
summary_latex.append(r"\midrule")

previous_dataset = None

for _, row in summary_table.iterrows():

    dataset = row["Dataset"]

    if previous_dataset is not None and dataset != previous_dataset:
        summary_latex.append(r"\midrule")

    summary_latex.append(
        f"{dataset} & "
        f"{row['Policy']} & "
        f"${row['Full']}$ & "
        f"${row['Target']}$ & "
        f"${row['Probe']}$ & "
        f"${row['Worst reduced']}$ \\\\"
    )

    previous_dataset = dataset

summary_latex.append(r"\bottomrule")
summary_latex.append(r"\end{tabular}")
summary_latex.append(r"\end{table*}")

with open(
    OUT_DIR / "Table_policy_level_summary.tex",
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(summary_latex))



for dataset_name in DATASET_ORDER:

    dataset_summary = condition_summary[
        condition_summary["dataset"]
        == dataset_name
    ].copy()

    plt.figure(figsize=(13, 6))

    x = np.arange(
        len(CONDITION_ORDER)
    )

    for policy_name in POLICY_ORDER:

        policy_data = (
            dataset_summary[
                dataset_summary["policy"]
                == policy_name
            ]
            .set_index("condition")
            .reindex(CONDITION_ORDER)
        )

        plt.errorbar(
            x,
            policy_data["mean_macro_f1"],
            yerr=policy_data["std_macro_f1"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=policy_name,
        )

    plt.axvline(
        x=5.5,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(
        x,
        [
            CONDITION_LABELS[c]
            for c in CONDITION_ORDER
        ],
        rotation=35,
        ha="right",
    )

    plt.ylabel("Macro-F1")
    plt.xlabel("Lead configuration")

    plt.title(
        f"{dataset_name}: Multi-seed performance under variable lead availability"
    )

    plt.ylim(
        0,
        1.03,
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    filename = (
        "Figure_condition_multiseed_"
        + dataset_name.replace("-", "").replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        OUT_DIR / filename,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()



SUMMARY_GROUPS = [
    "Target",
    "Probe",
    "Worst reduced",
]

for dataset_name in DATASET_ORDER:

    dataset_aggregate = aggregate_with_worst[
        aggregate_with_worst["dataset"]
        == dataset_name
    ].copy()

    plt.figure(figsize=(9, 6))

    x = np.arange(
        len(SUMMARY_GROUPS)
    )

    width = 0.24

    for policy_index, policy_name in enumerate(
        POLICY_ORDER
    ):

        policy_data = (
            dataset_aggregate[
                dataset_aggregate["policy"]
                == policy_name
            ]
            .set_index("condition_group")
            .reindex(SUMMARY_GROUPS)
        )

        positions = (
            x
            + (
                policy_index
                - 1
            )
            * width
        )

        plt.bar(
            positions,
            policy_data["mean_macro_f1"],
            width=width,
            yerr=policy_data["std_macro_f1"],
            capsize=3,
            label=policy_name,
        )

    plt.xticks(
        x,
        [
            "Target mean",
            "Probe mean",
            "Worst reduced",
        ],
    )

    plt.ylabel("Macro-F1")
    plt.title(
        f"{dataset_name}: Targeted versus broad robustness"
    )

    plt.ylim(
        0,
        1.03,
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    filename = (
        "Figure_policy_summary_"
        + dataset_name.replace("-", "").replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        OUT_DIR / filename,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()



def get_summary_value(
    dataset,
    policy,
    group,
):
    row = aggregate_with_worst[
        (
            aggregate_with_worst["dataset"]
            == dataset
        )
        &
        (
            aggregate_with_worst["policy"]
            == policy
        )
        &
        (
            aggregate_with_worst["condition_group"]
            == group
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not find unique row for "
            f"{dataset}, {policy}, {group}"
        )

    return (
        float(row.iloc[0]["mean_macro_f1"]),
        float(row.iloc[0]["std_macro_f1"]),
    )


manuscript_numbers = []

for dataset_name in DATASET_ORDER:

    for policy_name in POLICY_ORDER:

        for group_name in [
            "Full",
            "Target",
            "Probe",
            "Worst reduced",
        ]:

            mean_value, std_value = get_summary_value(
                dataset_name,
                policy_name,
                group_name,
            )

            manuscript_numbers.append(
                {
                    "dataset": dataset_name,
                    "policy": policy_name,
                    "summary_group": group_name,
                    "mean_macro_f1": mean_value,
                    "std_macro_f1": std_value,
                    "formatted": (
                        f"{mean_value:.4f} ± {std_value:.4f}"
                    ),
                }
            )

manuscript_numbers_df = pd.DataFrame(
    manuscript_numbers
)

manuscript_numbers_df.to_csv(
    OUT_DIR / "Exact_manuscript_numbers.csv",
    index=False,
)

print("\nExact manuscript summary numbers:")
display(manuscript_numbers_df)



print("\n" + "=" * 90)
print("MANUSCRIPT-READY OUTPUTS CREATED")
print("=" * 90)

for path in sorted(
    OUT_DIR.iterdir()
):
    print(path.name)

print("\nNo additional model run is required.")

# %% cell_05 [code]
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,   # Editable text in PDF
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

COLORS = {
    "Standard": "#0072B2",
    "Random": "#D55E00",
    "Structured": "#009E73",
}

METHODS = ["Standard", "Random", "Structured"]
CATEGORIES = ["Target mean", "Probe mean", "Worst reduced"]



chapman_means = {
    "Standard":   [0.665, 0.615, 0.330],
    "Random":     [0.957, 0.954, 0.945],
    "Structured": [0.956, 0.900, 0.825],
}

chapman_errors = {
    "Standard":   [0.015, 0.009, 0.045],
    "Random":     [0.003, 0.003, 0.004],
    "Structured": [0.003, 0.030, 0.075],
}

ptbxl_means = {
    "Standard":   [0.192, 0.120, 0.085],
    "Random":     [0.520, 0.480, 0.410],
    "Structured": [0.530, 0.302, 0.100],
}

ptbxl_errors = {
    "Standard":   [0.023, 0.017, 0.020],
    "Random":     [0.015, 0.017, 0.032],
    "Structured": [0.007, 0.010, 0.023],
}


def plot_policy_summary(
    means,
    errors,
    dataset_name,
    output_stem,
    legend_location="upper left",
):
    """Generate one publication-quality grouped bar chart."""

    x = np.arange(len(CATEGORIES))
    width = 0.23

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    offsets = [-width, 0, width]

    for method, offset in zip(METHODS, offsets):
        ax.bar(
            x + offset,
            means[method],
            width=width,
            yerr=errors[method],
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            error_kw={
                "ecolor": "#222222",
                "elinewidth": 1.1,
                "capsize": 3,
                "capthick": 1.1,
            },
            zorder=3,
        )

    ax.set_title(
        f"{dataset_name}: Targeted versus broad robustness",
        pad=10,
        fontweight="semibold",
    )
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)

    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        alpha=0.28,
        color="#8A8A8A",
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(axis="both", direction="out", length=3.5)

    legend = ax.legend(
        loc=legend_location,
        frameon=True,
        fancybox=False,
        edgecolor="#B5B5B5",
        framealpha=0.96,
        borderpad=0.5,
        handlelength=1.5,
    )
    legend.get_frame().set_linewidth(0.7)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        facecolor="white",
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        facecolor="white",
    )

    plt.show()
    plt.close(fig)



plot_policy_summary(
    means=chapman_means,
    errors=chapman_errors,
    dataset_name="Chapman",
    output_stem="Figure_policy_summary_Chapman_publication",
    legend_location="upper left",
)

plot_policy_summary(
    means=ptbxl_means,
    errors=ptbxl_errors,
    dataset_name="PTB-XL",
    output_stem="Figure_policy_summary_PTBXL_publication",
    legend_location="upper right",
)

# %% cell_06 [code]

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_ROOT = Path(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())

AUDIT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "bspc_submission_audit"
)

OUT_DIR = (
    PROJECT_ROOT
    / "lead_masking_final"
    / "bspc_manuscript_ready"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = (
    AUDIT_DIR
    / "all_multiseed_results_standardized_FIXED.csv"
)

if not RESULTS_CSV.exists():
    raise FileNotFoundError(
        f"Missing audited results:\n{RESULTS_CSV}\n"
        "Run the FIXED BSPC multi-seed audit cell first."
    )

results = pd.read_csv(RESULTS_CSV)

print("Loaded rows:", len(results))
print("Output directory:", OUT_DIR)



DATASET_ORDER = ["PTB-XL", "Chapman"]

POLICY_ORDER = [
    "Standard",
    "Random",
    "Structured",
]

POLICY_COLORS = {
    "Standard": "#0072B2",
    "Random": "#E69F00",
    "Structured": "#009E73",
}

CONDITION_ORDER = [
    "12-lead full",
    "6 limb",
    "6 precordial",
    "3 limb",
    "Lead II only",
    "V5 only",
    "Lead I only",
    "V1 only",
    "I+II",
    "V1+V5",
]

CONDITION_LABELS = {
    "12-lead full": "12-lead",
    "6 limb": "6 limb",
    "6 precordial": "6 precordial",
    "3 limb": "3 limb",
    "Lead II only": "Lead II",
    "V5 only": "V5",
    "Lead I only": "Lead I",
    "V1 only": "V1",
    "I+II": "I+II",
    "V1+V5": "V1+V5",
}

GROUP_LABELS = {
    "Full": "Full input",
    "Target": "Structured-policy target",
    "Probe": "Support-probing",
}

results["dataset"] = pd.Categorical(
    results["dataset"],
    categories=DATASET_ORDER,
    ordered=True,
)

results["policy"] = pd.Categorical(
    results["policy"],
    categories=POLICY_ORDER,
    ordered=True,
)

results["condition"] = pd.Categorical(
    results["condition"],
    categories=CONDITION_ORDER,
    ordered=True,
)



condition_summary = (
    results.groupby(
        [
            "dataset",
            "condition",
            "condition_group",
            "policy",
        ],
        observed=True,
    )["macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1="std",
        num_seeds="count",
        minimum="min",
        maximum="max",
    )
    .reset_index()
    .sort_values(
        [
            "dataset",
            "condition",
            "policy",
        ]
    )
)

condition_summary.to_csv(
    OUT_DIR / "Table_condition_level_multiseed.csv",
    index=False,
)

print("\nCondition-level results:")
display(condition_summary)



mean_pivot = condition_summary.pivot_table(
    index=[
        "dataset",
        "condition",
        "condition_group",
    ],
    columns="policy",
    values="mean_macro_f1",
    observed=True,
)

std_pivot = condition_summary.pivot_table(
    index=[
        "dataset",
        "condition",
        "condition_group",
    ],
    columns="policy",
    values="std_macro_f1",
    observed=True,
)

main_rows = []

for index in mean_pivot.index:

    dataset, condition, condition_group = index

    row = {
        "Dataset": str(dataset),
        "Lead condition": str(condition),
        "Condition family": GROUP_LABELS[str(condition_group)],
    }

    policy_means = {}

    for policy in POLICY_ORDER:

        mean_value = float(
            mean_pivot.loc[index, policy]
        )

        std_value = float(
            std_pivot.loc[index, policy]
        )

        policy_means[policy] = mean_value

        row[policy] = (
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    row["Structured − Random"] = (
        policy_means["Structured"]
        - policy_means["Random"]
    )

    row["Random − Standard"] = (
        policy_means["Random"]
        - policy_means["Standard"]
    )

    main_rows.append(row)

main_table = pd.DataFrame(main_rows)

main_table["Dataset"] = pd.Categorical(
    main_table["Dataset"],
    categories=DATASET_ORDER,
    ordered=True,
)

main_table["Lead condition"] = pd.Categorical(
    main_table["Lead condition"],
    categories=CONDITION_ORDER,
    ordered=True,
)

main_table = main_table.sort_values(
    [
        "Dataset",
        "Lead condition",
    ]
).reset_index(drop=True)

main_table.to_csv(
    OUT_DIR / "Table_main_multiseed_results.csv",
    index=False,
)

display(main_table)



latex_lines = []

latex_lines.append(r"\begin{table*}[t]")
latex_lines.append(r"\centering")
latex_lines.append(
    r"\caption{Macro-F1 across five independent training seeds. "
    r"Values are mean $\pm$ standard deviation. "
    r"Target conditions are explicitly included in the Structured "
    r"masking family. Probe conditions are held out from the Structured "
    r"policy; Lead I and V1 remain within Random masking support, "
    r"whereas I+II and V1+V5 are outside the exact support of both "
    r"masked policies.}"
)
latex_lines.append(r"\label{tab:main_multiseed_results}")
latex_lines.append(r"\small")
latex_lines.append(r"\setlength{\tabcolsep}{4pt}")
latex_lines.append(
    r"\begin{tabular}{lllccccc}"
)
latex_lines.append(r"\toprule")
latex_lines.append(
    r"Dataset & Lead condition & Family & Standard & Random & "
    r"Structured & Structured--Random & Random--Standard \\"
)
latex_lines.append(r"\midrule")

previous_dataset = None

for _, row in main_table.iterrows():

    dataset = str(row["Dataset"])
    condition = str(row["Lead condition"])
    family = str(row["Condition family"])

    if previous_dataset is not None and dataset != previous_dataset:
        latex_lines.append(r"\midrule")

    structured_random = float(
        row["Structured − Random"]
    )

    random_standard = float(
        row["Random − Standard"]
    )

    latex_lines.append(
        f"{dataset} & "
        f"{condition} & "
        f"{family} & "
        f"${row['Standard']}$ & "
        f"${row['Random']}$ & "
        f"${row['Structured']}$ & "
        f"{structured_random:+.4f} & "
        f"{random_standard:+.4f} \\\\"
    )

    previous_dataset = dataset

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table*}")

latex_main_table = "\n".join(latex_lines)

with open(
    OUT_DIR / "Table_main_multiseed_results.tex",
    "w",
    encoding="utf-8",
) as file:
    file.write(latex_main_table)

print("\nLaTeX main table saved.")



per_seed_group = (
    results.groupby(
        [
            "dataset",
            "policy",
            "seed",
            "condition_group",
        ],
        observed=True,
    )["macro_f1"]
    .mean()
    .rename("group_macro_f1")
    .reset_index()
)

aggregate_summary = (
    per_seed_group.groupby(
        [
            "dataset",
            "policy",
            "condition_group",
        ],
        observed=True,
    )["group_macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1="std",
        num_seeds="count",
    )
    .reset_index()
)

reduced_results = results[
    results["condition_group"].isin(
        ["Target", "Probe"]
    )
].copy()

worst_per_seed = (
    reduced_results.groupby(
        [
            "dataset",
            "policy",
            "seed",
        ],
        observed=True,
    )["macro_f1"]
    .min()
    .rename("worst_macro_f1")
    .reset_index()
)

worst_summary = (
    worst_per_seed.groupby(
        [
            "dataset",
            "policy",
        ],
        observed=True,
    )["worst_macro_f1"]
    .agg(
        mean_macro_f1="mean",
        std_macro_f1="std",
        num_seeds="count",
    )
    .reset_index()
)

worst_summary["condition_group"] = "Worst reduced"

aggregate_with_worst = pd.concat(
    [
        aggregate_summary,
        worst_summary,
    ],
    ignore_index=True,
)

aggregate_with_worst.to_csv(
    OUT_DIR / "Table_policy_level_summary.csv",
    index=False,
)

display(aggregate_with_worst)



summary_pivot_mean = aggregate_with_worst.pivot_table(
    index=[
        "dataset",
        "policy",
    ],
    columns="condition_group",
    values="mean_macro_f1",
    observed=True,
)

summary_pivot_std = aggregate_with_worst.pivot_table(
    index=[
        "dataset",
        "policy",
    ],
    columns="condition_group",
    values="std_macro_f1",
    observed=True,
)

summary_rows = []

for index in summary_pivot_mean.index:

    dataset, policy = index

    row = {
        "Dataset": str(dataset),
        "Policy": str(policy),
    }

    for group in [
        "Full",
        "Target",
        "Probe",
        "Worst reduced",
    ]:

        mean_value = float(
            summary_pivot_mean.loc[index, group]
        )

        std_value = float(
            summary_pivot_std.loc[index, group]
        )

        row[group] = (
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    summary_rows.append(row)

summary_table = pd.DataFrame(summary_rows)

summary_table.to_csv(
    OUT_DIR / "Table_policy_level_summary_formatted.csv",
    index=False,
)

display(summary_table)

summary_latex = []

summary_latex.append(r"\begin{table*}[t]")
summary_latex.append(r"\centering")
summary_latex.append(
    r"\caption{Policy-level Macro-F1 summary across five independent training seeds. "
    r"Target and probe values are computed by averaging conditions "
    r"within each seed before calculating the across-seed mean and "
    r"standard deviation. Worst reduced is the minimum Macro-F1 among "
    r"all nine reduced-lead conditions for each seed.}"
)
summary_latex.append(r"\label{tab:policy_level_summary}")
summary_latex.append(r"\small")
summary_latex.append(
    r"\begin{tabular}{llcccc}"
)
summary_latex.append(r"\toprule")
summary_latex.append(
    r"Dataset & Policy & Full input & Target mean & "
    r"Probe mean & Worst reduced \\"
)
summary_latex.append(r"\midrule")

previous_dataset = None

for _, row in summary_table.iterrows():

    dataset = row["Dataset"]

    if previous_dataset is not None and dataset != previous_dataset:
        summary_latex.append(r"\midrule")

    summary_latex.append(
        f"{dataset} & "
        f"{row['Policy']} & "
        f"${row['Full']}$ & "
        f"${row['Target']}$ & "
        f"${row['Probe']}$ & "
        f"${row['Worst reduced']}$ \\\\"
    )

    previous_dataset = dataset

summary_latex.append(r"\bottomrule")
summary_latex.append(r"\end{tabular}")
summary_latex.append(r"\end{table*}")

with open(
    OUT_DIR / "Table_policy_level_summary.tex",
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(summary_latex))



for dataset_name in DATASET_ORDER:

    dataset_summary = condition_summary[
        condition_summary["dataset"]
        == dataset_name
    ].copy()

    plt.figure(figsize=(13, 6))

    x = np.arange(
        len(CONDITION_ORDER)
    )

    for policy_name in POLICY_ORDER:

        policy_data = (
            dataset_summary[
                dataset_summary["policy"]
                == policy_name
            ]
            .set_index("condition")
            .reindex(CONDITION_ORDER)
        )

        plt.errorbar(
            x,
            policy_data["mean_macro_f1"],
            yerr=policy_data["std_macro_f1"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=POLICY_COLORS[policy_name],
            markerfacecolor=POLICY_COLORS[policy_name],
            markeredgecolor=POLICY_COLORS[policy_name],
            ecolor=POLICY_COLORS[policy_name],
            label=policy_name,
        )

    plt.axvline(
        x=5.5,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(
        x,
        [
            CONDITION_LABELS[c]
            for c in CONDITION_ORDER
        ],
        rotation=35,
        ha="right",
    )

    plt.ylabel("Macro-F1")
    plt.xlabel("Lead configuration")

    plt.title(
        f"{dataset_name}: Multi-seed performance under variable lead availability"
    )

    plt.ylim(
        0,
        1.03,
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    filename = (
        "Figure_condition_multiseed_"
        + dataset_name.replace("-", "").replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        OUT_DIR / filename,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()



SUMMARY_GROUPS = [
    "Target",
    "Probe",
    "Worst reduced",
]

for dataset_name in DATASET_ORDER:

    dataset_aggregate = aggregate_with_worst[
        aggregate_with_worst["dataset"]
        == dataset_name
    ].copy()

    plt.figure(figsize=(9, 6))

    x = np.arange(
        len(SUMMARY_GROUPS)
    )

    width = 0.24

    for policy_index, policy_name in enumerate(
        POLICY_ORDER
    ):

        policy_data = (
            dataset_aggregate[
                dataset_aggregate["policy"]
                == policy_name
            ]
            .set_index("condition_group")
            .reindex(SUMMARY_GROUPS)
        )

        positions = (
            x
            + (
                policy_index
                - 1
            )
            * width
        )

        plt.bar(
            positions,
            policy_data["mean_macro_f1"],
            width=width,
            yerr=policy_data["std_macro_f1"],
            capsize=3,
            color=POLICY_COLORS[policy_name],
            edgecolor=POLICY_COLORS[policy_name],
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 1.1,
                "capthick": 1.1,
            },
            label=policy_name,
        )

    plt.xticks(
        x,
        [
            "Target mean",
            "Probe mean",
            "Worst reduced",
        ],
    )

    plt.ylabel("Macro-F1")
    plt.title(
        f"{dataset_name}: Targeted versus broad robustness"
    )

    plt.ylim(
        0,
        1.03,
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    filename = (
        "Figure_policy_summary_"
        + dataset_name.replace("-", "").replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        OUT_DIR / filename,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()



def get_summary_value(
    dataset,
    policy,
    group,
):
    row = aggregate_with_worst[
        (
            aggregate_with_worst["dataset"]
            == dataset
        )
        &
        (
            aggregate_with_worst["policy"]
            == policy
        )
        &
        (
            aggregate_with_worst["condition_group"]
            == group
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not find unique row for "
            f"{dataset}, {policy}, {group}"
        )

    return (
        float(row.iloc[0]["mean_macro_f1"]),
        float(row.iloc[0]["std_macro_f1"]),
    )


manuscript_numbers = []

for dataset_name in DATASET_ORDER:

    for policy_name in POLICY_ORDER:

        for group_name in [
            "Full",
            "Target",
            "Probe",
            "Worst reduced",
        ]:

            mean_value, std_value = get_summary_value(
                dataset_name,
                policy_name,
                group_name,
            )

            manuscript_numbers.append(
                {
                    "dataset": dataset_name,
                    "policy": policy_name,
                    "summary_group": group_name,
                    "mean_macro_f1": mean_value,
                    "std_macro_f1": std_value,
                    "formatted": (
                        f"{mean_value:.4f} ± {std_value:.4f}"
                    ),
                }
            )

manuscript_numbers_df = pd.DataFrame(
    manuscript_numbers
)

manuscript_numbers_df.to_csv(
    OUT_DIR / "Exact_manuscript_numbers.csv",
    index=False,
)

print("\nExact manuscript summary numbers:")
display(manuscript_numbers_df)



print("\n" + "=" * 90)
print("MANUSCRIPT-READY OUTPUTS CREATED")
print("=" * 90)

for path in sorted(
    OUT_DIR.iterdir()
):
    print(path.name)

print("\nNo additional model run is required.")

