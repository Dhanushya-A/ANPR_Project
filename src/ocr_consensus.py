import os
import re
import csv
from collections import Counter

BASE_DIR = r"D:\ANPR_Project"

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "ocr_results", "ocr_results.txt"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "ocr_results", "consensus_ocr_results.csv"
)

# Common Indian state / UT registration prefixes
STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA",
    "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB"
}

# OCR substitutions that are relatively common.
# These are used only in appropriate positions.
DIGIT_FIX = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "7",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8"
}

LETTER_FIX = {
    "0": "O",
    "1": "I",
    "5": "S",
    "6": "G",
    "8": "B"
}


def clean_text(text):
    """Keep only alphanumeric OCR characters."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def extract_blocks(text):
    """
    Split OCR TXT file into plate blocks.

    Expected structure begins with lines such as:
        00000037_plate_1.jpg
        Best OCR: ...
        Confidence: ...
        Candidates:
    """

    lines = text.splitlines()

    blocks = []
    current = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # A plate image filename marks a new block.
        if re.match(r"^.+_plate_\d+\.(jpg|jpeg|png|bmp)$", line, re.I):
            if current:
                blocks.append(current)

            current = {
                "filename": line,
                "candidates": []
            }
            continue

        if current is None:
            continue

        # Candidate lines look approximately like:
        # TEXT | variant | PSM 6 | conf=...
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]

            if len(parts) >= 1:
                text_candidate = clean_text(parts[0])

                if text_candidate:
                    confidence = 0.0

                    for part in parts:
                        m = re.search(
                            r"conf\s*=\s*([0-9.]+)",
                            part,
                            re.I
                        )

                        if m:
                            try:
                                confidence = float(m.group(1))
                            except ValueError:
                                confidence = 0.0

                    current["candidates"].append({
                        "text": text_candidate,
                        "confidence": confidence
                    })

    if current:
        blocks.append(current)

    return blocks


def normalize_candidate(text):
    """
    Safely normalize obvious OCR formatting mistakes.

    We do not invent missing characters.
    """

    text = clean_text(text)

    if len(text) < 4:
        return text

    # If OCR added one character before a known state code,
    # remove that leading character.
    for state in STATE_CODES:
        if text.startswith("X" + state):
            text = text[1:]
            break

        if len(text) >= 3 and text[1:3] == state:
            text = text[1:]
            break

    # Standard Indian plate pattern:
    # STATE + 1/2 digit RTO + 1/3 letters + 1/4 digits
    #
    # We only perform positional substitutions if the structure
    # is already reasonably clear.

    for state in STATE_CODES:

        if not text.startswith(state):
            continue

        remainder = text[2:]

        # RTO code: 2 digits
        if len(remainder) >= 2:

            rto = remainder[:2]

            rto_fixed = "".join(
                DIGIT_FIX.get(ch, ch) for ch in rto
            )

            if rto_fixed.isdigit():
                text = state + rto_fixed + remainder[2:]

        break

    return text


def structure_score(text):
    """
    Score how closely a candidate resembles an Indian
    registration number.

    This is NOT ground-truth accuracy.
    """

    text = normalize_candidate(text)

    score = 0

    if len(text) < 7 or len(text) > 12:
        return score

    state = text[:2]

    if state in STATE_CODES:
        score += 4
    else:
        return score

    remainder = text[2:]

    # Look for a 2-digit RTO section.
    if len(remainder) >= 2:
        rto = remainder[:2]

        if rto.isdigit():
            score += 3

    # General standard format
    pattern = r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"

    if re.match(pattern, text):
        score += 6

    return score


def candidate_score(candidate, support_count):
    """
    Combined score:
    - OCR confidence
    - repeated support
    - Indian plate structure
    - reasonable length
    """

    text = candidate["text"]
    confidence = candidate["confidence"]

    normalized = normalize_candidate(text)

    score = 0

    # Confidence
    score += min(confidence, 100) * 0.45

    # Repeated OCR agreement
    score += min(support_count, 5) * 8

    # Structure
    score += structure_score(normalized) * 4

    # Reasonable plate length
    if 8 <= len(normalized) <= 10:
        score += 8
    elif 7 <= len(normalized) <= 12:
        score += 3

    return score


def process_block(block):

    candidates = block["candidates"]

    if not candidates:
        return {
            "filename": block["filename"],
            "recommended_plate": "",
            "confidence": 0,
            "support_count": 0,
            "structure_score": 0,
            "decision": "REJECT",
            "method": "NO_CANDIDATES"
        }

    # Normalize candidates.
    normalized_candidates = []

    for candidate in candidates:

        normalized = normalize_candidate(candidate["text"])

        if not normalized:
            continue

        normalized_candidates.append({
            "text": normalized,
            "confidence": candidate["confidence"]
        })

    if not normalized_candidates:
        return {
            "filename": block["filename"],
            "recommended_plate": "",
            "confidence": 0,
            "support_count": 0,
            "structure_score": 0,
            "decision": "REJECT",
            "method": "EMPTY"
        }

    # Count repeated candidate readings.
    counts = Counter(
        c["text"] for c in normalized_candidates
    )

    # Keep highest-confidence occurrence for each candidate.
    best_by_text = {}

    for candidate in normalized_candidates:

        text = candidate["text"]

        if (
            text not in best_by_text
            or candidate["confidence"]
            > best_by_text[text]["confidence"]
        ):
            best_by_text[text] = candidate

    ranked = []

    for text, candidate in best_by_text.items():

        support = counts[text]

        score = candidate_score(
            candidate,
            support
        )

        ranked.append({
            "text": text,
            "confidence": candidate["confidence"],
            "support": support,
            "structure": structure_score(text),
            "score": score
        })

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = ranked[0]

    # Decision rules.
    #
    # ACCEPT requires BOTH reasonable OCR confidence
    # and repeated/strong structural evidence.
    #
    # REVIEW remains the safe default.

        # Strong consensus:
    # repeated OCR agreement is important because each plate
    # was processed using multiple image variants and PSM modes.

    if (
        best["structure"] >= 10
        and best["support"] >= 5
        and best["confidence"] >= 45
    ):
        decision = "ACCEPT"

    elif (
        best["structure"] >= 10
        and best["support"] >= 3
        and best["confidence"] >= 60
    ):
        decision = "ACCEPT"

    elif (
        best["structure"] >= 10
        and best["support"] >= 2
        and best["confidence"] >= 75
    ):
        decision = "ACCEPT"

    else:
        decision = "REVIEW"

    return {
        "filename": block["filename"],
        "recommended_plate": best["text"],
        "confidence": round(best["confidence"], 2),
        "support_count": best["support"],
        "structure_score": best["structure"],
        "consensus_score": round(best["score"], 2),
        "decision": decision,
        "method": "OCR_CONSENSUS"
    }


def main():

    if not os.path.exists(INPUT_FILE):
        print("ERROR: OCR results file not found:")
        print(INPUT_FILE)
        return

    print("Reading OCR candidates...")
    print(INPUT_FILE)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        text = f.read()

    blocks = extract_blocks(text)

    print("Plate blocks found:", len(blocks))

    results = []

    for block in blocks:
        results.append(
            process_block(block)
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    fieldnames = [
        "filename",
        "recommended_plate",
        "confidence",
        "support_count",
        "structure_score",
        "consensus_score",
        "decision",
        "method"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    accepted = sum(
        r["decision"] == "ACCEPT"
        for r in results
    )

    review = sum(
        r["decision"] == "REVIEW"
        for r in results
    )

    rejected = sum(
        r["decision"] == "REJECT"
        for r in results
    )

    print()
    print("========== OCR CONSENSUS ==========")
    print("Total records :", len(results))
    print("Accepted      :", accepted)
    print("Review        :", review)
    print("Rejected      :", rejected)
    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()