import pandas as pd


def normalize_toi(input_path: str) -> None:
    # Read, preserving only data (comment lines are ignored by the app anyway)
    df = pd.read_csv(input_path, comment="#")

    # Ensure disposition column exists
    if "tfopwg_disp" in df.columns:
        disp = df["tfopwg_disp"].astype(str).str.strip().str.upper()
        # Map common variants to accepted labels
        disp = disp.replace({
            "APC": "PC",   # Ambiguous Planet Candidate -> treat as PC
            "KP": "CP",    # Known Planet -> treat as Confirmed Planet
            "CONFIRMED": "CP",
            "CANDIDATE": "PC",
            "FALSE POSITIVE": "FP",
        })
        df["tfopwg_disp"] = disp

        # Keep only allowed values
        allowed = {"CP", "PC", "FP"}
        df = df[df["tfopwg_disp"].isin(allowed)].copy()

    # Best-effort RA/DEC: if missing, try to infer from likely alternatives
    if "ra" not in df.columns:
        for cand in ["ra_deg", "ra_degrees", "ra_gaia", "raJ2000", "raJ2000_deg"]:
            if cand in df.columns:
                df["ra"] = df[cand]
                break
    if "dec" not in df.columns:
        for cand in ["dec_deg", "dec_degrees", "dec_gaia", "decJ2000", "decJ2000_deg"]:
            if cand in df.columns:
                df["dec"] = df[cand]
                break

    # Write back (without comments)
    df.to_csv(input_path, index=False)


def normalize_k2(input_path: str) -> None:
    df = pd.read_csv(input_path, comment="#")

    # Normalize disposition
    if "disposition" in df.columns:
        disp = df["disposition"].astype(str).str.strip().str.upper()
        disp = disp.replace({
            "CONFIRMED": "CONFIRMED",
            "CANDIDATE": "CANDIDATE",
            "FALSE POSITIVE": "FALSE POSITIVE",
            "FP": "FALSE POSITIVE",
            "PC": "CANDIDATE",
            "CP": "CONFIRMED",
        })
        df["disposition"] = disp
        allowed = {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
        df = df[df["disposition"].isin(allowed)].copy()

    # Best-effort RA/DEC
    if "ra" not in df.columns:
        for cand in ["ra_deg", "ra_degrees", "ra_gaia", "raJ2000", "raJ2000_deg"]:
            if cand in df.columns:
                df["ra"] = df[cand]
                break
    if "dec" not in df.columns:
        for cand in ["dec_deg", "dec_degrees", "dec_gaia", "decJ2000", "decJ2000_deg"]:
            if cand in df.columns:
                df["dec"] = df[cand]
                break

    df.to_csv(input_path, index=False)


def main():
    normalize_toi("TESS Objects of Interest (TOI) copy.csv")
    normalize_k2("K2 Planets and Candidates copy.csv")


if __name__ == "__main__":
    main()


