#!/usr/bin/env python3
"""Validates a results_finetune.csv against the acceptance criteria: <=10% error on
max_allocated_gb for every non-OOM row, and a correct OOM call on every row.

Reads predicted_allocated_gb / predicted_oom straight out of results_finetune.csv (columns
benchmark_finetune.py already writes from predictions_finetune.py) -- this script doesn't
recompute anything, it just checks what was written against what was measured. No GPU
needed. Same shape as validate_results.py; different column set.
"""
import argparse
import csv
import sys

ERROR_THRESHOLD = 0.10


def _to_bool(s):
    return str(s).strip().lower() in ("true", "1")


def validate(rows):
    report_rows = []
    all_ok = True

    for r in rows:
        label = f"{r['lever']}={r['lever_value']}"
        oom = _to_bool(r["oom"])
        predicted_oom = _to_bool(r.get("predicted_oom", "False"))
        oom_correct = oom == predicted_oom

        if oom:
            report_rows.append({
                "label": label, "predicted_gb": None, "measured_gb": None,
                "abs_error_gb": None, "pct_error": None,
                "oom": True, "predicted_oom": predicted_oom, "oom_correct": oom_correct,
            })
            all_ok = all_ok and oom_correct
            continue

        try:
            predicted_gb = float(r["predicted_allocated_gb"])
            measured_gb = float(r["max_allocated_gb"])
        except (TypeError, ValueError):
            report_rows.append({
                "label": label, "predicted_gb": None, "measured_gb": None,
                "abs_error_gb": None, "pct_error": None,
                "oom": False, "predicted_oom": predicted_oom, "oom_correct": False,
            })
            all_ok = False
            continue

        abs_error_gb = predicted_gb - measured_gb
        pct_error = abs(abs_error_gb) / measured_gb
        within_threshold = pct_error <= ERROR_THRESHOLD

        report_rows.append({
            "label": label, "predicted_gb": predicted_gb, "measured_gb": measured_gb,
            "abs_error_gb": abs_error_gb, "pct_error": pct_error,
            "oom": False, "predicted_oom": predicted_oom, "oom_correct": oom_correct,
        })
        all_ok = all_ok and oom_correct and within_threshold

    return report_rows, all_ok


def print_report(report_rows):
    header = f"{'config':<32} {'predicted_gb':>12} {'measured_gb':>12} {'abs_err_gb':>11} {'pct_err':>9} {'oom_call':>10}"
    print(header)
    print("-" * len(header))
    for r in report_rows:
        if r["oom"]:
            oom_call = "correct" if r["oom_correct"] else "WRONG"
            print(f"{r['label']:<32} {'OOM':>12} {'OOM':>12} {'--':>11} {'--':>9} {oom_call:>10}")
            continue
        oom_call = "correct" if r["oom_correct"] else "WRONG"
        flag = "" if r["pct_error"] <= ERROR_THRESHOLD else "  <-- FAIL"
        print(f"{r['label']:<32} {r['predicted_gb']:>12.3f} {r['measured_gb']:>12.3f} "
              f"{r['abs_error_gb']:>+11.3f} {r['pct_error']:>8.1%} {oom_call:>10}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results_finetune.csv")
    args = p.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    report_rows, all_ok = validate(rows)
    print_report(report_rows)

    n_oom_wrong = sum(1 for r in report_rows if r["oom"] and not r["oom_correct"])
    n_over_threshold = sum(1 for r in report_rows
                          if not r["oom"] and r["pct_error"] is not None
                          and r["pct_error"] > ERROR_THRESHOLD)
    n_oom_call_wrong_on_non_oom = sum(1 for r in report_rows
                                     if not r["oom"] and not r["oom_correct"])

    print()
    print("Acceptance: <=10% error on max_allocated_gb (non-OOM rows), correct OOM call on all rows.")
    print(f"  Non-OOM rows over 10% error: {n_over_threshold}")
    print(f"  OOM rows where OOM wasn't predicted: {n_oom_wrong}")
    print(f"  Non-OOM rows where OOM was wrongly predicted: {n_oom_call_wrong_on_non_oom}")
    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
