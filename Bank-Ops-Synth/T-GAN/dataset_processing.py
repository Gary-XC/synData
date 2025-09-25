import pandas as pd
import numpy as np
from collections import defaultdict

# Data
# https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=001

# ---- 2021 Census DGUIDs for provinces/territories ----
CENSUS_CODES = {
    "NL": "2021A000210",
    "PE": "2021A000211",
    "NS": "2021A000212",
    "NB": "2021A000213",
    "QC": "2021A000224",
    "ON": "2021A000235",
    "MB": "2021A000246",
    "SK": "2021A000247",
    "AB": "2021A000248",
    "BC": "2021A000259",
    "YT": "2021A000260",
    "NT": "2021A000261",
    "NU": "2021A000262",
}

# ---- parser ----
def parse_census_profile(path: str, prov: str):
    """
    Parse a 2021 Census Profile CSV for one province.
    Returns: dict with age_mix, renter_rate, and (mu, sigma) for lognormal income fit.
    """
    dguid = CENSUS_CODES[prov]
    df = pd.read_csv(path)

    # Filter province
    prov_df = df[df["DGUID"] == dguid].copy()

    # --- AGE distribution (broad bands) ---
    age_bands = defaultdict(int)
    for _, row in prov_df.iterrows():
        name = str(row["CHARACTERISTIC_NAME"])
        count = row["C1_COUNT_TOTAL"]
        if pd.isna(count): 
            continue

        # Example categories from Census: "0 to 14 years", "15 to 24 years", etc.
        if "0 to 14 years" in name:
            age_bands["0-14"] += int(count)
        elif "15 to 24 years" in name:
            age_bands["15-24"] += int(count)
        elif "25 to 64 years" in name:
            age_bands["25-64"] += int(count)
        elif "65 years and over" in name:
            age_bands["65+"] += int(count)

    total_pop = sum(age_bands.values())
    age_mix = {k: v / total_pop for k, v in age_bands.items() if total_pop > 0}

    # --- Tenure (owner vs renter) ---
    renter_rate, owner_rate = None, None
    for _, row in prov_df.iterrows():
        name = str(row["CHARACTERISTIC_NAME"])
        count = row["C1_COUNT_TOTAL"]
        if pd.isna(count): 
            continue
        if "Owner" in name and "households" in name:
            owner_count = int(count)
        elif "Renter" in name and "households" in name:
            renter_count = int(count)
        elif "Private households" in name and "total" in name.lower():
            total_households = int(count)

    if "owner_count" in locals() and "renter_count" in locals() and total_households > 0:
        owner_rate = owner_count / total_households # type: ignore
        renter_rate = renter_count / total_households # type: ignore

    # --- Income distribution (fit lognormal to bands) ---
    income_vals, income_counts = [], []
    for _, row in prov_df.iterrows():
        name = str(row["CHARACTERISTIC_NAME"])
        count = row["C1_COUNT_TOTAL"]
        if pd.isna(count): 
            continue
        if "Total income groups" in name and "$" in name:
            # Example: "Total income $20,000 to $29,999"
            try:
                parts = name.replace(",", "").split()
                lo = int(parts[2].replace("$", ""))
                hi = int(parts[4].replace("$", "")) if parts[4].replace("$", "").isdigit() else lo + 10000
                mid = (lo + hi) / 2
                income_vals.append(mid)
                income_counts.append(int(count))
            except:
                continue

    income_array = np.repeat(income_vals, income_counts)
    log_income = np.log(income_array + 1)
    mu, sigma = log_income.mean(), log_income.std()

    return {
        "province": prov,
        "age_mix": age_mix,
        "renter_rate": renter_rate,
        "owner_rate": owner_rate,
        "income_mu": mu,
        "income_sigma": sigma,
    }

# ---- Example usage ----
if __name__ == "__main__":
    # Replace with the actual Census Profile CSV path
    path = "98-401-X2021006_English_CSV_data.csv"
    priors_on = parse_census_profile(path, "ON")
    print(priors_on)
