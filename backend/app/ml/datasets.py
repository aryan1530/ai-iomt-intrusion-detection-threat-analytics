"""Load CICIoMT2024, NSL-KDD, and WUSTL-EHMS-2020, with synthetic fallbacks for demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "worm", "mailbomb"}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {"guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"}
U2R = {"buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"}


def map_nsl_category(label: str) -> str:
    l = str(label).strip().lower().rstrip(".")
    if l in {"normal"}:
        return "normal"
    if l in DOS:
        return "DoS"
    if l in PROBE:
        return "Probe"
    if l in R2L:
        return "R2L"
    if l in U2R:
        return "U2R"
    return "malicious"


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def _label_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in {"label", "class", "attack", "attack_cat", "attack_type", "target"}:
            return c
    return df.columns[-1]


def generate_synthetic(dataset: str, n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_mal = n // 3
    n_norm = n - n_mal
    proto = rng.choice(["tcp", "udp", "icmp"], n)
    service = rng.choice(["http", "ftp", "smtp", "private", "eco_i", "other"], n)
    flag = rng.choice(["SF", "S0", "REJ", "RSTR"], n)
    src = rng.integers(0, 50_000, n).astype(float)
    dst = rng.integers(0, 50_000, n).astype(float)
    count = rng.integers(1, 300, n).astype(float)
    serror = rng.random(n)
    same_srv = rng.random(n)
    duration = rng.integers(0, 5000, n).astype(float)
    dst_host = rng.integers(1, 255, n).astype(float)
    labels_bin = np.array(["normal"] * n_norm + ["malicious"] * n_mal)
    cats = np.array(["normal"] * n_norm + list(rng.choice(["DoS", "Probe", "R2L", "U2R", "MITM", "Spoofing"], n_mal)))
    rng.shuffle(labels_bin)
    # correlate malicious with high serror / count
    mal_mask = rng.random(n) < 0.33
    serror[mal_mask] = rng.uniform(0.6, 1.0, mal_mask.sum())
    count[mal_mask] = rng.integers(120, 511, mal_mask.sum())
    labels_bin = np.where(mal_mask, "malicious", "normal")
    cats = np.where(mal_mask, rng.choice(["DoS", "Probe", "R2L", "U2R", "MITM", "Spoofing"], n), "normal")
    if dataset == "nsl_kdd":
        df = pd.DataFrame({
            "duration": duration, "protocol_type": proto, "service": service, "flag": flag,
            "src_bytes": src, "dst_bytes": dst, "land": rng.integers(0, 2, n),
            "wrong_fragment": rng.integers(0, 4, n), "urgent": rng.integers(0, 2, n),
            "hot": rng.integers(0, 10, n), "num_failed_logins": rng.integers(0, 5, n),
            "logged_in": rng.integers(0, 2, n), "num_compromised": rng.integers(0, 8, n),
            "root_shell": rng.integers(0, 2, n), "su_attempted": rng.integers(0, 2, n),
            "num_root": rng.integers(0, 5, n), "num_file_creations": rng.integers(0, 5, n),
            "num_shells": rng.integers(0, 3, n), "num_access_files": rng.integers(0, 3, n),
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": rng.integers(0, 2, n),
            "count": count, "srv_count": rng.integers(1, 300, n),
            "serror_rate": serror, "srv_serror_rate": serror * rng.uniform(0.8, 1.0, n),
            "rerror_rate": rng.random(n) * 0.4, "srv_rerror_rate": rng.random(n) * 0.4,
            "same_srv_rate": same_srv, "diff_srv_rate": 1 - same_srv,
            "srv_diff_host_rate": rng.random(n), "dst_host_count": dst_host,
            "dst_host_srv_count": rng.integers(1, 255, n),
            "dst_host_same_srv_rate": rng.random(n), "dst_host_diff_srv_rate": rng.random(n),
            "dst_host_same_src_port_rate": rng.random(n), "dst_host_srv_diff_host_rate": rng.random(n),
            "dst_host_serror_rate": serror * rng.uniform(0.7, 1.0, n),
            "dst_host_srv_serror_rate": serror * rng.uniform(0.7, 1.0, n),
            "dst_host_rerror_rate": rng.random(n) * 0.3, "dst_host_srv_rerror_rate": rng.random(n) * 0.3,
            "label": cats, "difficulty": rng.integers(0, 21, n),
        })
        df["binary_label"] = labels_bin
        df["attack_category"] = cats
        return df
    if dataset == "wustl_ehms":
        df = pd.DataFrame({
            "SrcBytes": src, "DstBytes": dst, "SrcLoad": rng.random(n) * 1000,
            "DstLoad": rng.random(n) * 1000, "SrcGap": rng.integers(0, 100, n),
            "DstGap": rng.integers(0, 100, n), "SIntPkt": rng.random(n) * 10,
            "DIntPkt": rng.random(n) * 10, "SIntPktAct": rng.random(n),
            "Dur": duration, "Trans": rng.integers(1, 50, n),
            "TotPkts": count, "TotBytes": src + dst, "Load": rng.random(n) * 500,
            "Loss": np.where(mal_mask, rng.uniform(0.1, 0.8, n), rng.uniform(0, 0.1, n)),
            "pLoss": serror, "Rate": rng.random(n) * 100,
            "SrcJitter": rng.random(n), "DstJitter": rng.random(n),
            "sMeanPktSz": rng.uniform(40, 1500, n), "dMeanPktSz": rng.uniform(40, 1500, n),
            "SAppBytes": src * 0.8, "DAppBytes": dst * 0.8,
            "HeartRate": rng.integers(50, 140, n), "RespRate": rng.integers(10, 30, n),
            "SpO2": rng.integers(88, 100, n), "Pulse": rng.integers(50, 140, n),
            "Temp": rng.uniform(36, 39, n),
            "label": cats,
        })
        df["binary_label"] = labels_bin
        df["attack_category"] = cats
        return df
    # CICIoMT2024-style flow features
    df = pd.DataFrame({
        "Flow_Duration": duration, "Header_Length": rng.integers(20, 80, n),
        "Protocol_Type": proto, "Duration": duration,
        "Rate": rng.random(n) * 200, "Srate": rng.random(n) * 200, "Drate": rng.random(n) * 200,
        "fin_flag_number": rng.integers(0, 5, n), "syn_flag_number": np.where(mal_mask, rng.integers(5, 40, n), rng.integers(0, 4, n)),
        "rst_flag_number": rng.integers(0, 5, n), "psh_flag_number": rng.integers(0, 8, n),
        "ack_flag_number": rng.integers(0, 20, n), "ece_flag_number": rng.integers(0, 2, n),
        "cwr_flag_number": rng.integers(0, 2, n),
        "ack_count": rng.integers(0, 50, n), "syn_count": count,
        "fin_count": rng.integers(0, 10, n), "urg_count": rng.integers(0, 3, n),
        "rst_count": rng.integers(0, 8, n),
        "HTTP": rng.integers(0, 2, n), "HTTPS": rng.integers(0, 2, n),
        "DNS": rng.integers(0, 2, n), "Telnet": rng.integers(0, 2, n),
        "SMTP": rng.integers(0, 2, n), "SSH": rng.integers(0, 2, n),
        "IRC": rng.integers(0, 2, n), "TCP": (proto == "tcp").astype(int),
        "UDP": (proto == "udp").astype(int), "DHCP": rng.integers(0, 2, n),
        "ARP": rng.integers(0, 2, n), "ICMP": (proto == "icmp").astype(int),
        "IGMP": rng.integers(0, 2, n), "IPv": 1, "LLC": rng.integers(0, 2, n),
        "Tot_sum": src + dst, "Min": rng.uniform(20, 80, n), "Max": rng.uniform(200, 1500, n),
        "AVG": rng.uniform(80, 600, n), "Std": rng.uniform(1, 200, n),
        "Tot_size": src + dst, "IAT": rng.random(n) * 2, "Number": count,
        "Magnitude": rng.random(n) * 10, "Radius": rng.random(n) * 5,
        "Covariance": rng.random(n), "Variance": rng.random(n), "Weight": rng.random(n) * 10,
        "label": cats,
    })
    df["binary_label"] = labels_bin
    df["attack_category"] = cats
    return df


def load_nsl_kdd() -> pd.DataFrame:
    raw = settings.data_raw_dir / "nsl_kdd"
    train = _first_existing([
        raw / "KDDTrain+.txt", raw / "KDDTrain+.csv", raw / "NSL-KDD" / "KDDTrain+.txt",
    ])
    test = _first_existing([
        raw / "KDDTest+.txt", raw / "KDDTest+.csv", raw / "NSL-KDD" / "KDDTest+.txt",
    ])
    frames = []
    for p in [train, test]:
        if p is None:
            continue
        df = pd.read_csv(p, header=None)
        if df.shape[1] == len(NSL_KDD_COLUMNS):
            df.columns = NSL_KDD_COLUMNS
        elif df.shape[1] == len(NSL_KDD_COLUMNS) - 1:
            df.columns = NSL_KDD_COLUMNS[:-1]
        else:
            df = _read_csv_flexible(p)
            if "label" not in df.columns:
                df.rename(columns={df.columns[-1]: "label"}, inplace=True)
        frames.append(df)
    if not frames:
        return generate_synthetic("nsl_kdd")
    df = pd.concat(frames, ignore_index=True)
    lab = _label_column(df)
    df["attack_category"] = df[lab].map(map_nsl_category)
    df["binary_label"] = np.where(df["attack_category"].eq("normal"), "normal", "malicious")
    return df


def load_ciciomt2024() -> pd.DataFrame:
    raw = settings.data_raw_dir / "ciciomt2024"
    files = list(raw.rglob("*.csv")) if raw.exists() else []
    if not files:
        return generate_synthetic("ciciomt2024")
    df = pd.concat([_read_csv_flexible(f) for f in files[:8]], ignore_index=True)
    lab = _label_column(df)
    labels = df[lab].astype(str).str.strip()
    df["attack_category"] = labels.where(~labels.str.lower().isin(["benign", "normal", "0"]), "normal")
    df.loc[df["attack_category"].str.lower().isin(["benign", "normal"]), "attack_category"] = "normal"
    df["binary_label"] = np.where(df["attack_category"].str.lower().eq("normal"), "normal", "malicious")
    return df


def load_wustl_ehms() -> pd.DataFrame:
    raw = settings.data_raw_dir / "wustl_ehms"
    files = list(raw.rglob("*.csv")) if raw.exists() else []
    if not files:
        return generate_synthetic("wustl_ehms")
    df = pd.concat([_read_csv_flexible(f) for f in files[:4]], ignore_index=True)
    lab = _label_column(df)
    labels = df[lab].astype(str).str.strip()
    df["attack_category"] = np.where(labels.str.lower().isin(["0", "normal", "benign"]), "normal", labels)
    df["binary_label"] = np.where(df["attack_category"].astype(str).str.lower().eq("normal"), "normal", "malicious")
    return df


LOADERS = {
    "nsl_kdd": load_nsl_kdd,
    "ciciomt2024": load_ciciomt2024,
    "wustl_ehms": load_wustl_ehms,
}


def load_dataset(name: str) -> pd.DataFrame:
    key = name.lower().replace("-", "_")
    if key not in LOADERS:
        raise ValueError(f"Unknown dataset {name}. Use {list(LOADERS)}")
    df = LOADERS[key]()
    df["dataset"] = key
    return df


def list_available() -> dict:
    out = {}
    for name in LOADERS:
        d = settings.data_raw_dir / name
        files = list(d.rglob("*")) if d.exists() else []
        real = [str(f.relative_to(d)) for f in files if f.is_file()]
        out[name] = {"raw_files": real, "using_synthetic": len(real) == 0}
    return out
