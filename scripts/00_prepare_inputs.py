import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import export, subject_id


MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def column(frame, names, required=True):
    keys = {re.sub(r"[^a-z0-9]", "", str(x).lower()): x for x in frame.columns}
    for name in names:
        hit = keys.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if hit is not None:
            return hit
    for name in names:
        token = re.sub(r"[^a-z0-9]", "", name.lower())
        if len(token) >= 3:
            for key, original in keys.items():
                if token in key:
                    return original
    if required:
        raise KeyError(f"Missing columns {names}; found {list(frame.columns)}")
    return None


def clean_text(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def matching_text(value):
    text = clean_text(value).lower()
    for symbol in ("’", "‘", "`"):
        text = text.replace(symbol, "'")
    return text


def occurrence_keys(values):
    counts = {}
    keys = []
    for value in values:
        count = counts.get(value, 0)
        keys.append(f"{value}@@{count}")
        counts[value] = count + 1
    return keys


def find_behavior_files(root, folders):
    target = [x.lower() for x in folders]
    paths = {}
    for path in root.rglob("*.xlsx"):
        if path.name.startswith("~$"):
            continue
        parts = [part.lower() for part in path.parent.parts]
        if all(item in parts for item in target) and re.match(r"sub[-_]?\d+", path.stem, re.I):
            key = subject_id(path.stem)
            if key in paths:
                raise ValueError(f"Duplicate {key}: {paths[key]} and {path}")
            paths[key] = path
    return paths


def subject_info(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if lines and all(re.fullmatch(r"(?:sub[-_]?)?\d+", line, re.I) for line in lines):
        return pd.DataFrame({"subject": [subject_id(line) for line in lines]})
    table = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    key = column(table, ["subject", "participant_id", "BIDS_ID", "subject_id", "id"], required=False)
    if key is None and len(table.columns) == 1:
        table = pd.read_csv(path, header=None, names=["subject"], encoding="utf-8-sig")
        key = "subject"
    if key is None:
        raise KeyError(f"Cannot identify participant IDs in {path}")
    table = table.rename(columns={key: "subject"})
    table["subject"] = table["subject"].map(subject_id)
    return table.drop_duplicates("subject")


def sex_binary(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    return {"male": 1.0, "m": 1.0, "man": 1.0, "1": 1.0, "1.0": 1.0,
            "female": 0.0, "f": 0.0, "woman": 0.0, "0": 0.0, "0.0": 0.0}.get(value, np.nan)


def prepare_participants(behavior_root, info, selected):
    files = sorted(behavior_root.rglob("ThinkAloudQuestionnaires_summary.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one questionnaires/ThinkAloudQuestionnaires_summary.csv under {behavior_root}; found {len(files)}")
    surveys = pd.read_csv(files[0])
    key = column(surveys, ["BIDS_ID", "participant_id", "subject", "subject_id"])
    surveys = surveys.rename(columns={key: "subject"})
    surveys["subject"] = surveys["subject"].map(subject_id)
    surveys = surveys.drop_duplicates("subject")
    ids = info[["subject"]].drop_duplicates().merge(surveys, on="subject", how="left")
    for out, names in {
        "AutomaticThought": ["AutomaticThought", "ATQ", "ATQ30", "ATQ_total", "ATQ_score"],
        "age": ["Age", "age"],
        "sex_M": ["sex_M", "sex_m"],
        "sex": ["Gender", "sex", "gender"],
    }.items():
        field = column(ids, names, required=False)
        if field is not None:
            ids[out] = ids[field]
    for field in ["age", "sex_M", "sex"]:
        source = column(info, [field, "Age" if field == "age" else "Gender"], required=False)
        if source is not None:
            fallback = info.set_index("subject")[source]
            if field not in ids.columns:
                ids[field] = ids["subject"].map(fallback)
            else:
                ids[field] = ids["subject"].map(fallback).combine_first(ids[field])
    if "sex_M" not in ids.columns:
        ids["sex_M"] = np.nan
    ids["sex_M"] = pd.to_numeric(ids["sex_M"], errors="coerce")
    if "sex" in ids.columns:
        ids["sex_M"] = ids["sex_M"].combine_first(ids["sex"].map(sex_binary))
    if "age" not in ids.columns:
        ids["age"] = np.nan
    ids["age"] = pd.to_numeric(ids["age"], errors="coerce")
    ids["AutomaticThought"] = pd.to_numeric(ids["AutomaticThought"], errors="coerce")
    return ids.loc[ids["subject"].isin(selected), ["subject", "age", "sex_M", "AutomaticThought"]].sort_values("subject")


def read_reference(path):
    frame = pd.read_excel(path)
    text_key = column(frame, ["Transcribed Sentence", "sentence", "text", "transcript"])
    start_key = column(frame, ["Start Time", "start_time", "onset", "start"], required=False)
    ref = pd.DataFrame({"text": frame[text_key].map(clean_text)})
    ref["norm"] = ref["text"].map(matching_text)
    ref = ref.loc[~ref["norm"].isin(MISSING_TEXT)].copy()
    if start_key is not None:
        ref["onset"] = pd.to_numeric(frame.loc[ref.index, start_key], errors="coerce")
        if ref["onset"].notna().sum() >= max(2, int(.8 * len(ref))):
            ref = ref.sort_values("onset", kind="stable")
    if "onset" not in ref:
        ref["onset"] = np.nan
    ref = ref.reset_index(drop=True)
    ref["reference_order"] = np.arange(len(ref))
    ref["key"] = occurrence_keys(ref["norm"])
    return ref


def prepare_sentences(subject, rated_path, reference_path, strict_order=False):
    frame = pd.read_excel(rated_path)
    text_key = column(frame, ["Transcribed Sentence", "sentence", "text", "transcript"])
    intensity_key = column(frame, ["Emotional intensity", "emotional_intensity", "valence", "emotion"])
    rated = pd.DataFrame({"text": frame[text_key].map(clean_text),
                          "valence": pd.to_numeric(frame[intensity_key], errors="coerce")})
    rated["source_row"] = np.arange(len(rated))
    rated["norm"] = rated["text"].map(matching_text)
    missing_text = int(rated["norm"].isin(MISSING_TEXT).sum())
    rated = rated.loc[~rated["norm"].isin(MISSING_TEXT)].copy()
    rated["key"] = occurrence_keys(rated["norm"])
    match_rate = np.nan
    order_verified = False
    reference_used = False
    if reference_path is not None:
        reference = read_reference(reference_path)
        aligned = rated.merge(reference[["key", "reference_order", "onset"]], on="key", how="left", validate="one_to_one")
        match_rate = float(aligned["reference_order"].notna().mean())
        if match_rate >= .95:
            maximum = reference["reference_order"].max()
            absent = aligned["reference_order"].isna()
            aligned.loc[absent, "reference_order"] = maximum + 1 + np.arange(absent.sum())
            rated = aligned.sort_values(["reference_order", "source_row"], kind="stable")
            order_verified = bool(match_rate == 1.0)
            reference_used = True
    if strict_order and not order_verified:
        raise ValueError(f"Sentence order not fully verified for {subject} (match={match_rate})")
    rated = rated.reset_index(drop=True)
    rated["sentence_index"] = np.arange(len(rated))
    rated["subject"] = subject
    if "onset" not in rated:
        rated["onset"] = np.nan
    missing_valence = int(rated["valence"].isna().sum())
    valid = rated["valence"].notna()
    summary = {"subject": subject, "rating_file": str(rated_path),
               "reference_file": str(reference_path) if reference_path else "",
               "n_sentences": len(rated), "missing_text_removed": missing_text,
               "missing_valence_retained": missing_valence,
               "reference_match_rate": match_rate, "reference_used": reference_used,
               "order_verified": order_verified,
               "negative_sentence_prop": float((rated["valence"] < 0).sum() / valid.sum()) if valid.any() else np.nan}
    return rated[["subject", "sentence_index", "text", "valence", "onset"]], summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--subject-info", required=True)
    parser.add_argument("--exclude-file", default="config/imaging_exclusions.txt")
    parser.add_argument("--strict-order", action="store_true")
    parser.add_argument("--out", default="./results")
    args = parser.parse_args()
    out = Path(args.out)
    root = Path(args.behavior)
    if root.is_file() and root.suffix.lower() == ".zip":
        unpacked = out / "source_data"
        if not unpacked.exists():
            unpacked.mkdir(parents=True)
            with zipfile.ZipFile(root) as archive:
                archive.extractall(unpacked)
        root = unpacked
    info = subject_info(args.subject_info)
    excluded = {subject_id(line.strip()) for line in Path(args.exclude_file).read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")}
    metadata_ids = set(info["subject"])
    eligible_ids = metadata_ids - excluded
    rated = find_behavior_files(root, ["sentence_level_ratings", "gpt_generated"])
    references = find_behavior_files(root, ["transcripts_and_timestamps", "sentence_level"])
    candidates = prepare_participants(root, info, eligible_ids)
    complete_ids = set(candidates.dropna(subset=["age", "sex_M", "AutomaticThought"])["subject"])
    selected = complete_ids & set(rated)
    participants = candidates.loc[candidates["subject"].isin(selected)].copy()
    sentences = []
    summaries = []
    for subject in sorted(selected & set(rated)):
        block, summary = prepare_sentences(subject, rated[subject], references.get(subject), args.strict_order)
        sentences.append(block)
        summaries.append(summary)
    if not sentences:
        raise RuntimeError(f"No sentence-level ratings found under {root}")
    sentences = pd.concat(sentences, ignore_index=True)
    qc = pd.DataFrame(summaries)
    available = set(qc["subject"])
    participants = participants.loc[participants["subject"].isin(available)]
    export(participants, out / "inputs" / "participants.csv")
    export(sentences, out / "inputs" / "sentences.csv")
    export(qc, out / "tables" / "source_qc.csv")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "selected_subjects.txt").write_text("\n".join(participants["subject"]) + "\n", encoding="utf-8")
    report = {"subject_info": str(args.subject_info), "behavior": str(args.behavior),
              "exclude_file": str(args.exclude_file), "metadata_subjects": len(metadata_ids),
              "imaging_exclusions_in_metadata": len(excluded & metadata_ids),
              "post_imaging_qc_subjects": len(eligible_ids),
              "complete_behavior_subjects": len(complete_ids),
              "selected_subjects": len(participants),
              "rated_subjects": len(available),
              "reference_subjects": len(set(references) & available),
              "missing_ratings": sorted(complete_ids - set(rated)),
              "missing_questionnaire_data": sorted(eligible_ids - complete_ids),
              "unverified_sentence_order": sorted(qc.loc[~qc["order_verified"], "subject"].tolist())}
    (out / "tables" / "source_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
