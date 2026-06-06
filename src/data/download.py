# src/data/download.py
"""
Baixa o dataset Home Credit Default Risk do Kaggle.
Pré-requisito: arquivo ~/.kaggle/kaggle.json com credenciais,
ou variáveis KAGGLE_USERNAME e KAGGLE_KEY no .env.
"""

import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
COMPETITION = "home-credit-default-risk"
TARGET_FILE = DATA_RAW / "application_train.csv"


def download_dataset(force: bool = False) -> Path:
    """
    Baixa e descompacta o dataset. Pula se TARGET_FILE já existir (a não ser que force=True).
    Retorna o path do arquivo principal.
    """
    if TARGET_FILE.exists() and not force:
        print(f"Dataset já existe em {TARGET_FILE}. Use force=True para re-baixar.")
        return TARGET_FILE

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle

        kaggle.api.authenticate()
        print(f"Baixando competição '{COMPETITION}'...")
        kaggle.api.competition_download_files(
            COMPETITION,
            path=str(DATA_RAW),
            quiet=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Erro ao baixar dataset: {e}\n"
            "Verifique se KAGGLE_USERNAME e KAGGLE_KEY estão no .env "
            "ou em ~/.kaggle/kaggle.json"
        ) from e

    # Descompactar todos os .zip baixados
    for zip_path in DATA_RAW.glob("*.zip"):
        print(f"Descompactando {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_RAW)
        zip_path.unlink()

    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Arquivo esperado não encontrado após download: {TARGET_FILE}")

    print(f"Dataset pronto em {DATA_RAW}/")
    return TARGET_FILE


if __name__ == "__main__":
    path = download_dataset()
    print(f"Arquivo principal: {path}")
