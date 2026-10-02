from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    database_url: str = f"sqlite:///{(ROOT / 'backend' / 'iomt.db').as_posix()}"
    artifacts_dir: Path = ROOT / "artifacts"
    data_raw_dir: Path = ROOT / "data" / "raw"
    data_processed_dir: Path = ROOT / "data" / "processed"
    feature_mode: str = "union"
    mi_threshold: float = 0.01
    top_k_per_model: int = 20
    log_level: str = "INFO"


settings = Settings()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
