import re
from pathlib import Path


DETAIL_METRICS_RE = re.compile(
    r"^\s*(-?\d+\.\d{4})\s+(-?\d+\.\d{4})\s+(-?\d+\.\d{4})\s+(-?\d+\.\d{4})\s*$"
)
GLOBAL_SUMMARY_RE = re.compile(
    r"^(.*?)\s+(-?\d+\.\d{4})\s+(-?\d+\.\d{4})\s+(-?\d+\.\d{4})\s+(-?\d+\.\d{4})\s+(\d+)\s*$"
)


def latex_escape(text: str) -> str:
    escaped = text.replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def parse_paper_style(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()

    detail_rows = []
    global_rows = []
    in_global_summary = False
    current_feature = ""

    for line in lines:
        if "GLOBAL MODEL SUMMARY" in line:
            in_global_summary = True
            continue

        if in_global_summary:
            if not line.strip() or line.startswith("=") or line.startswith("-") or line.strip().startswith("Feature"):
                continue
            m = GLOBAL_SUMMARY_RE.match(line)
            if not m:
                continue
            feature = m.group(1).strip()
            global_rows.append(
                {
                    "feature": feature,
                    "accuracy": float(m.group(2)),
                    "precision": float(m.group(3)),
                    "recall": float(m.group(4)),
                    "f1": float(m.group(5)),
                    "samples": int(m.group(6)),
                }
            )
            continue

        if not line.strip() or line.startswith("=") or line.startswith("-") or line.strip().startswith("Feature"):
            continue

        if len(line) < 70:
            continue

        feature = line[:30].strip()
        cluster = line[31:56].strip()
        model = line[57:72].strip()
        metrics = line[72:]

        m = DETAIL_METRICS_RE.match(metrics)
        if not m:
            continue

        if feature:
            current_feature = feature
        else:
            feature = current_feature

        detail_rows.append(
            {
                "feature": feature,
                "cluster": cluster,
                "model": model,
                "accuracy": float(m.group(1)),
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
                "is_global": cluster == "** GLOBAL **",
            }
        )

    return detail_rows, global_rows


def build_latex(title: str, detail_rows, global_rows):
    out = []
    out.append("% Auto-generated from paper_style_results*.txt")
    out.append(r"\begin{table}[htbp]")
    out.append(r"\centering")
    out.append(rf"\caption{{{latex_escape(title)} - Global Results}}")
    out.append(r"\begin{tabular}{lrrrr}")
    out.append(r"\hline")
    out.append(r"Feature & Accuracy & Precision & Recall & F1-Score \\")
    out.append(r"\hline")
    for row in global_rows:
        out.append(
            f"{latex_escape(row['feature'])} & "
            f"{row['accuracy']:.4f} & "
            f"{row['precision']:.4f} & "
            f"{row['recall']:.4f} & "
            f"{row['f1']:.4f} \\\\"
        )
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")

    out.append(r"\begin{table}[htbp]")
    out.append(r"\centering")
    out.append(rf"\caption{{{latex_escape(title)} - Cluster Results}}")
    out.append(r"\begin{tabular}{llrrrr}")
    out.append(r"\hline")
    out.append(r"Feature & Cluster & Accuracy & Precision & Recall & F1-Score \\")
    out.append(r"\hline")
    prev_feature = None
    for row in detail_rows:
        if row["is_global"]:
            continue
        if prev_feature is not None and row["feature"] != prev_feature:
            out.append(r"\hline")
        out.append(
            f"{latex_escape(row['feature'])} & "
            f"{latex_escape(row['cluster'])} & "
            f"{row['accuracy']:.4f} & "
            f"{row['precision']:.4f} & "
            f"{row['recall']:.4f} & "
            f"{row['f1']:.4f} \\\\"
        )
        prev_feature = row["feature"]
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")
    return "\n".join(out)


def convert_one(input_path: Path, output_path: Path, title: str):
    detail_rows, global_rows = parse_paper_style(input_path)
    latex = build_latex(title, detail_rows, global_rows)
    output_path.write_text(latex, encoding="utf-8")
    return len(detail_rows), len(global_rows)


def main():
    root = Path(__file__).resolve().parent
    jobs = [
        (
            root / "dataset_1" / "paper_style_results_regression.txt",
            root / "dataset_1" / "paper_style_results_regression_tables.tex",
            "Dataset 1 (Regression)",
        ),
        (
            root / "dataset_1" / "paper_style_results_classification.txt",
            root / "dataset_1" / "paper_style_results_classification_tables.tex",
            "Dataset 1 (Classification)",
        ),
        (
            root / "dataset_2" / "paper_style_results_regression.txt",
            root / "dataset_2" / "paper_style_results_regression_tables.tex",
            "Dataset 2 (Regression)",
        ),
        (
            root / "dataset_2" / "paper_style_results_classification.txt",
            root / "dataset_2" / "paper_style_results_classification_tables.tex",
            "Dataset 2 (Classification)",
        ),
        (
            root / "dataset_3" / "paper_style_results_classification.txt",
            root / "dataset_3" / "paper_style_results_classification_tables.tex",
            "Dataset 3 (Classification)",
        ),
        (
            root / "dataset_3" / "paper_style_results_regression.txt",
            root / "dataset_3" / "paper_style_results_regression_tables.tex",
            "Dataset 3 (Regression)",
        ),
    ]

    for input_path, output_path, title in jobs:
        detail_count, global_count = convert_one(input_path, output_path, title)
        print(f"Generated {output_path} (detail rows: {detail_count}, global rows: {global_count})")


if __name__ == "__main__":
    main()
