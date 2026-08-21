from __future__ import annotations

import os
import json
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from .vision import call_vision_model, call_text_model, NINE

# 8 risk indicators = the 7 damage/condition cues + (site_accessible == False)
_RISK = ["house_destruction", "structural_damage", "exterior_debris", "open_doors_windows",
         "exterior_mud", "emergency_markings", "major_repairs"]


def decide_occupancy(api_output_str: str, tau: int = 2) -> str:
    """
    One-stage deterministic rule (paper Table 1 / Eq. 3), canonical snake_case schema.

    Predict 'Not Occupied' iff (r - v) >= tau, where r is the number of true risk
    indicators (the 7 damage/condition cues plus site_accessible == False) and
    v = 1 if a vehicle is present. Effective threshold is 2 with no vehicle, 3 with one.

    Args:
        api_output_str (str): JSON string with the 9 boolean attributes.
        tau (int): base threshold (default 2, the paper's operating point).
    Returns:
        str: 'Occupied' or 'Not Occupied'.
    """
    data = json.loads(api_output_str)
    a = {k: bool(data.get(k, False)) for k in NINE}
    r = sum(a[k] for k in _RISK) + (0 if a["site_accessible"] else 1)
    v = 1 if a["vehicle_presence"] else 0
    return "Not Occupied" if (r - v) >= tau else "Occupied"


def plot_confusion_matrix(y_true, y_pred, title, labels):
    """
    Ploting confusion matrix for classification results.
    Args:
        y_true (series): True labels.
        y_pred (series): Predicted labels.
        title (str): Title for the plot.
        labels (list): List of class labels for display order.
    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title(title)
    plt.show()


def run_vlm_pipeline(
    csv_path: str,
    base_image_dir: str,
    output_csv_path: str,
    use_llm: bool = True,
    ground_truth_csv: str | None = None,
    image_id_column: str | None = None,
) -> None:
    """
    Run the VLM pipeline over images listed in CSV, producing predictions.

    Args:
        csv_path: Input CSV; must contain either 'objectid' or 'ObjectId'.
        base_image_dir: Directory containing images named '<objectid>.jpg'.
        output_csv_path: Where to write predictions.
        use_llm: If True, use text LLM for final decision; else rule-based.
        ground_truth_csv: Optional path to a CSV with labels for evaluation.
        image_id_column: Optional explicit column name for image id.
    """
    df = pd.read_csv(csv_path)

    # Determine id column and derive image file names
    id_col = image_id_column
    if id_col is None:
        if "objectid" in df.columns:
            id_col = "objectid"
        elif "ObjectId" in df.columns:
            id_col = "ObjectId"
        else:
            raise ValueError("CSV must contain 'objectid' or 'ObjectId' column or specify --image_id_column")

    ids = df[id_col].astype(str)
    image_filenames = ids + ".jpg"

    results = []
    vision_outputs = []
    vision_token = 0
    llm_token = 0

    for filename in image_filenames:
        image_path = os.path.join(base_image_dir, filename)
        try:
            v_out, vt = call_vision_model(image_path)
            vision_outputs.append(v_out)
            vision_token += vt or 0
            time.sleep(1)  # rate limiting
            if use_llm:
                occ, lt = call_text_model(v_out)
                results.append(occ)
                llm_token += lt or 0
            else:
                # Expect strict JSON; if model wraps it, attempt to extract
                txt = v_out.strip()
                # naive strip of surrounding code fences or text
                if txt.startswith("```"):
                    txt = txt.strip("`\n").split("\n", 1)[-1]
                if not txt.startswith("{"):
                    # fallback: find first '{' to end '}'
                    s = txt.find("{")
                    e = txt.rfind("}")
                    if s != -1 and e != -1:
                        txt = txt[s : e + 1]
                occ = decide_occupancy(txt)
                results.append(occ)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            vision_outputs.append("")
            results.append("")

    print("Total Vision Tokens:", vision_token)
    print("Total LLM Tokens:", llm_token)

    df_out = df.copy()
    df_out["Vision Model Output"] = vision_outputs
    df_out["Occupancy Prediction"] = results
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    # Optional evaluation
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        df_eval = pd.read_csv(ground_truth_csv)
        if "label" not in df_eval.columns or "Occupancy Prediction" not in df_out.columns:
            print("Ground truth CSV must contain 'label' column for evaluation.")
            return
        # Align by id_col if present in both
        if id_col in df_eval.columns:
            merged = df_eval[[id_col, "label"]].merge(
                df_out[[id_col, "Occupancy Prediction"]], on=id_col, how="inner"
            )
            y_true = merged["label"]
            y_pred = merged["Occupancy Prediction"]
        else:
            # fallback: assume same order
            y_true = df_eval["label"]
            y_pred = df_out["Occupancy Prediction"][: len(y_true)]

        y_true = y_true.dropna()
        y_pred = y_pred.iloc[y_true.index]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label="Not Occupied")
        f1w = f1_score(y_true, y_pred, average="weighted")
        print("Evaluation:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (Not Occupied): {f1:.4f}")
        print(f"  F1 (weighted): {f1w:.4f}")
        plot_confusion_matrix(
            y_true,
            y_pred,
            "Confusion Matrix",
            ["Not Occupied", "Occupied"],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run VLM occupancy pipeline over images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", help="CSV with 'ObjectId' or 'objectid' column")
    parser.add_argument("image_dir", help="Directory containing images named '<id>.jpg'")
    parser.add_argument("out_csv", help="Output CSV path for predictions")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based classifier instead of LLM")
    parser.add_argument("--gt_csv", default=None, help="Optional ground truth CSV for evaluation")
    parser.add_argument("--image_id_column", default=None, help="Explicit image id column name if not standard")

    args = parser.parse_args()
    run_vlm_pipeline(
        csv_path=args.csv,
        base_image_dir=args.image_dir,
        output_csv_path=args.out_csv,
        use_llm=not args.no_llm,
        ground_truth_csv=args.gt_csv,
        image_id_column=args.image_id_column,
    )
