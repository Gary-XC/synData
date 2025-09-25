import re
import pandas as pd
import numpy as np
from collections import defaultdict

# -------- DGUID map (unchanged)
CENSUS_CODES = {
    "NL":"2021A000210","PE":"2021A000211","NS":"2021A000212","NB":"2021A000213",
    "QC":"2021A000224","ON":"2021A000235","MB":"2021A000246","SK":"2021A000247",
    "AB":"2021A000248","BC":"2021A000259","YT":"2021A000260","NT":"2021A000261","NU":"2021A000262",
}

def _to_num(x):
    return pd.to_numeric(str(x).replace(",", "").strip(), errors="coerce")

def peek_labels(path, prov):
    """Print useful rows so we can see exact wording in your CSV."""
    dguid = CENSUS_CODES[prov]
    df = pd.read_csv(path, low_memory=False)
    df = df[df["DGUID"] == dguid].copy()
    s = df["CHARACTERISTIC_NAME"].astype(str)

    print("\n--- SAMPLE: rows containing 'years' (age) ---")
    print(df[s.str.contains(r"\byears\b|\bans\b", case=False, regex=True)].head(30)[["CHARACTERISTIC_NAME","C1_COUNT_TOTAL"]])

    print("\n--- SAMPLE: rows containing tenure/owner/renter ---")
    pat_ten = r"tenure|ménage|household|owner|propriét|renter|locataire"
    print(df[s.str.contains(pat_ten, case=False, regex=True)].head(30)[["CHARACTERISTIC_NAME","C1_COUNT_TOTAL"]])

    print("\n--- SAMPLE: rows containing income ---")
    print(df[s.str.contains(r"income|revenu", case=False, regex=True)].head(30)[["CHARACTERISTIC_NAME","C1_COUNT_TOTAL"]])

def parse_census_profile(path: str, prov: str, debug=False):
    """
    Robust parser for 2021 Census Profile CSV (English/French variants).
    Returns dict: {age_mix, renter_rate, owner_rate, income_mu, income_sigma}
    """
    dguid = CENSUS_CODES[prov]
    df = pd.read_csv(path, low_memory=False)
    df = df[df["DGUID"] == dguid].copy()
    if df.empty:
        raise ValueError(f"DGUID {dguid} not found in file: {path}")

    # normalize numeric
    df["C1_COUNT_TOTAL"] = df["C1_COUNT_TOTAL"].apply(_to_num)

    # ---------- AGE BANDS ----------
    # handles "0 to 14 years" / "0 à 14 ans" / etc.
    age_patterns = {
        "0-14":  r"(?:\b0\s*(?:to|à)\s*14\s*(?:years|ans)\b)",
        "15-24": r"(?:\b15\s*(?:to|à)\s*24\s*(?:years|ans)\b)",
        "25-64": r"(?:\b25\s*(?:to|à)\s*64\s*(?:years|ans)\b)",
        "65+":   r"(?:\b65\s*(?:years|ans)\s*and\s*over\b|\b65\s*ans\s*et\s*plus\b)",
    }
    age_counts = defaultdict(float)
    for k, pat in age_patterns.items():
        sel = df["CHARACTERISTIC_NAME"].astype(str).str.contains(pat, case=False, regex=True, na=False)
        age_counts[k] = float(df.loc[sel, "C1_COUNT_TOTAL"].sum())

    total_pop = sum(age_counts.values())
    age_mix = {k: (v/total_pop) for k, v in age_counts.items()} if total_pop > 0 else {}

    if debug and not age_mix:
        print("[DEBUG] Age bands not found — check wording:")
        peek_labels(path, prov)

    # ---------- TENURE (OWNER / RENTER) ----------
    # Count households by tenure; accept English/French variants.
    s = df["CHARACTERISTIC_NAME"].astype(str).str.lower()
    # rows that look like households (not percentages)
    is_hh = s.str.contains(r"household|ménages|private households|ménages privés", regex=True)
    is_pct = s.str.contains(r"percent|%|pourcentage", regex=True)

    owner_rows  = df[ is_hh & ~is_pct & s.str.contains(r"owner|propriét", regex=True) ]
    renter_rows = df[ is_hh & ~is_pct & s.str.contains(r"renter|locataire", regex=True) ]

    owner_count  = float(owner_rows["C1_COUNT_TOTAL"].sum())
    renter_count = float(renter_rows["C1_COUNT_TOTAL"].sum())
    hh_total = owner_count + renter_count
    owner_rate  = (owner_count / hh_total) if hh_total > 0 else None
    renter_rate = (renter_count / hh_total) if hh_total > 0 else None

    if debug and renter_rate is None:
        print("[DEBUG] Tenure not matched — check wording:")
        peek_labels(path, prov)

    # ---------- INCOME (fit lognormal to income groups) ----------
    # We prefer banded totals like "Total income groups in 2020 constant dollars".
    # Fall back to "Median total income (dollars)" if needed.
    income_rows = df[s.str.contains(r"total income group|income groups|groupes de revenu", regex=True)]
    mids, weights = [], []

    for _, r in income_rows.iterrows():
        name = str(r["CHARACTERISTIC_NAME"])
        cnt = float(r["C1_COUNT_TOTAL"] or 0.0)
        if cnt <= 0:
            continue

        # "$a to $b" or "$a à $b"
        m = re.search(r"\$([\d,]+)\s*(?:to|-|à)\s*\$([\d,]+)", name)
        if m:
            lo = int(m.group(1).replace(",",""))
            hi = int(m.group(2).replace(",",""))
            mids.append((lo+hi)/2.0); weights.append(cnt); continue

        # "$a and over" / "$a et plus" / "$a+"
        m2 = re.search(r"\$([\d,]+)\s*(?:and\s*over|et\s*plus|\+)", name)
        if m2:
            lo = int(m2.group(1).replace(",",""))
            # heuristic for open-top bin width
            hi = int(lo * 1.5)
            mids.append((lo+hi)/2.0); weights.append(cnt); continue

    income_mu = income_sigma = None
    if mids:
        mids = np.array(mids, dtype=float)
        weights = np.array(weights, dtype=float)
        w = weights / (weights.sum() if weights.sum() else 1.0)
        logm = np.log(mids + 1)
        income_mu = float((w * logm).sum())
        income_sigma = float(np.sqrt(max((w * (logm - income_mu)**2).sum(), 0.0)))
    else:
        # Fallback: use median total income
        median_row = df[s.str.contains(r"\bmedian total income\b|\brevenu total médian\b", regex=True)]
        if not median_row.empty:
            med = float(median_row["C1_COUNT_TOTAL"].iloc[0] or 0.0)
            if med > 0:
                income_mu = float(np.log(med + 1))  # approximate
                income_sigma = 0.55                 # conservative default if bands absent
        if debug and income_mu is None:
            print("[DEBUG] Income bands/median not matched — check wording:")
            peek_labels(path, prov)

    return {
        "province": prov,
        "age_mix": age_mix,
        "renter_rate": renter_rate,
        "owner_rate": owner_rate,
        "income_mu": income_mu,
        "income_sigma": income_sigma,
    }

# ---- quick test
if __name__ == "__main__":
    path = "data/Canada_Demog.csv"  # adjust if needed
    print(parse_census_profile(path, "ON", debug=True))
