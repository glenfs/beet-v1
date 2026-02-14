import os
from pathlib import Path
import csv
from music21 import converter, stream, note, chord

MIDI_DATASET_PATH = r"D:\My Projects\pythonprojects\music21-corpus-test\dataset\bach\4_parts_only"

# --------- TUNABLE THRESHOLDS ----------
MIN_SOUNDING_RATIO = 0.60     # if a part sounds less than this fraction of total duration -> suspicious
MIN_EMPTY_MEAS_RUN = 4        # if a part has >= this many empty measures consecutively -> suspicious
MIN_SCORE_DURATION_QL = 8.0    # ignore super-short files (in quarterLength) if needed
# --------------------------------------

SUPPORTED_EXTS = {".xml", ".musicxml", ".mxl", ".mid", ".midi", ".krn"}


def part_sounding_duration_ql(part: stream.Part) -> float:
    """Sum quarterLength of Notes + Chords (ignores rests)."""
    total = 0.0
    for el in part.recurse():
        if isinstance(el, (note.Note, chord.Chord)):
            try:
                total += float(el.duration.quarterLength)
            except Exception:
                pass
    return total


def longest_empty_measure_run(part: stream.Part) -> int:
    """
    Compute longest consecutive run of measures that contain no Notes/Chords.
    Requires measures to exist or will attempt to make them.
    """
    # Ensure measures exist
    measures = part.getElementsByClass(stream.Measure)
    if len(measures) == 0:
        try:
            part = part.makeMeasures()
            measures = part.getElementsByClass(stream.Measure)
        except Exception:
            return 0

    longest = 0
    current = 0

    for m in measures:
        has_sound = False
        for el in m.recurse():
            if isinstance(el, (note.Note, chord.Chord)):
                has_sound = True
                break
        if has_sound:
            longest = max(longest, current)
            current = 0
        else:
            current += 1

    longest = max(longest, current)
    return int(longest)


def analyze_score(filepath: Path) -> dict:
    score = converter.parse(str(filepath))

    # Total duration (quarterLength)
    total_dur = float(getattr(score, "highestTime", 0.0) or 0.0)

    # If score is too short, still analyze but mark it
    short_score = total_dur < MIN_SCORE_DURATION_QL

    # Collect per-part metrics
    parts = list(score.parts) if hasattr(score, "parts") else []
    part_rows = []

    for idx, p in enumerate(parts):
        pid = p.id if p.id else f"Part{idx+1}"
        sounding = part_sounding_duration_ql(p)
        ratio = (sounding / total_dur) if total_dur > 0 else 0.0
        empty_run = longest_empty_measure_run(p)

        part_rows.append({
            "part_id": pid,
            "sounding_ql": sounding,
            "sounding_ratio": ratio,
            "longest_empty_measures_run": empty_run
        })

    # Decide if suspicious
    suspicious_reasons = []
    for pr in part_rows:
        if pr["sounding_ratio"] < MIN_SOUNDING_RATIO:
            suspicious_reasons.append(
                f"{pr['part_id']} low sounding_ratio={pr['sounding_ratio']:.2f}"
            )
        if pr["longest_empty_measures_run"] >= MIN_EMPTY_MEAS_RUN:
            suspicious_reasons.append(
                f"{pr['part_id']} empty_measures_run={pr['longest_empty_measures_run']}"
            )

    return {
        "file": str(filepath),
        "total_duration_ql": total_dur,
        "short_score": short_score,
        "num_parts": len(parts),
        "parts": part_rows,
        "suspicious": len(suspicious_reasons) > 0,
        "reasons": "; ".join(suspicious_reasons)
    }


def main():
    root = Path(MIDI_DATASET_PATH)
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

    print(f"Found {len(files)} files under: {root}")

    results = []
    bad = []
    errors = []

    for i, fp in enumerate(files, 1):
        try:
            r = analyze_score(fp)
            results.append(r)
            if r["suspicious"]:
                bad.append(r)
                print(f"[SUSPICIOUS] {fp.name} -> {r['reasons']}")
        except Exception as e:
            errors.append((str(fp), repr(e)))
            print(f"[ERROR] {fp.name} -> {e}")

    # Write a CSV report
    out_csv = root / "bach_missing_parts_report.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "total_duration_ql",
            "num_parts",
            "short_score",
            "suspicious",
            "reasons",
            "part_id",
            "sounding_ql",
            "sounding_ratio",
            "longest_empty_measures_run"
        ])

        for r in results:
            if not r["parts"]:
                w.writerow([r["file"], r["total_duration_ql"], r["num_parts"], r["short_score"],
                            r["suspicious"], r["reasons"], "", "", "", ""])
            else:
                for pr in r["parts"]:
                    w.writerow([
                        r["file"],
                        r["total_duration_ql"],
                        r["num_parts"],
                        r["short_score"],
                        r["suspicious"],
                        r["reasons"],
                        pr["part_id"],
                        pr["sounding_ql"],
                        f"{pr['sounding_ratio']:.4f}",
                        pr["longest_empty_measures_run"]
                    ])

    print("\n---------------------")
    print(f"Suspicious files: {len(bad)}")
    print(f"Errors: {len(errors)}")
    print(f"CSV report saved to: {out_csv}")

    if errors:
        err_txt = root / "bach_parse_errors.txt"
        with open(err_txt, "w", encoding="utf-8") as f:
            for fp, msg in errors:
                f.write(f"{fp}\n{msg}\n\n")
        print(f"Error log saved to: {err_txt}")


if __name__ == "__main__":
    main()
