"""Inferencia de sentimiento para las reseñas de WeLoveReviews.

Lee data/raw/reviews.csv, predice el sentimiento de cada reseña con un modelo
preentrenado de Hugging Face (fijado a un commit concreto) y escribe
data/processed/reviews_with_sentiment.csv.

Uso (desde cualquier carpeta):
    python src/app.py
"""

import os
from pathlib import Path

# Salida limpia: sin barras de progreso de descarga ni avisos de symlinks en Windows
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from transformers import pipeline

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"  # commit fijado en el Hub
BATCH_SIZE = 16

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

BANDS = ["Positivo", "Neutral", "Negativo"]


def stars_to_band(stars: int) -> str:
    """Mapea estrellas (1–5) a banda: 1–2 Negativo, 3 Neutral, 4–5 Positivo."""
    if stars <= 2:
        return "Negativo"
    if stars == 3:
        return "Neutral"
    return "Positivo"


def load_reviews(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model():
    """Carga el pipeline de sentimiento una sola vez, en CPU."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
        device=-1,
    )


def predict(reviews: pd.DataFrame, classifier) -> pd.DataFrame:
    """Agrega pred_stars, pred_score, sentiment_band y rating_band a cada reseña."""
    predictions = classifier(
        reviews["review_text"].tolist(), batch_size=BATCH_SIZE, truncation=True
    )
    enriched = reviews.assign(
        pred_stars=[int(p["label"].split()[0]) for p in predictions],
        pred_score=[round(p["score"], 4) for p in predictions],
    )
    enriched["sentiment_band"] = enriched["pred_stars"].map(stars_to_band)
    enriched["rating_band"] = enriched["rating"].map(stars_to_band)
    return enriched


def summarize(enriched: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "% modelo (texto)": enriched["sentiment_band"].value_counts(normalize=True),
            "% rating humano": enriched["rating_band"].value_counts(normalize=True),
        })
        .reindex(BANDS)
        .fillna(0)
        .mul(100)
        .round(1)
    )


def main() -> None:
    reviews = load_reviews()
    print(f"Reseñas leídas: {len(reviews)} ({RAW_PATH.relative_to(ROOT)})")

    classifier = load_model()
    print(f"Modelo: {MODEL_NAME} @ {MODEL_REVISION[:7]}")

    enriched = predict(reviews, classifier)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Archivo escrito: {OUTPUT_PATH.relative_to(ROOT)} ({len(enriched)} filas)\n")

    print(summarize(enriched).to_string())
    print(
        f"\nPromedio de estrellas — rating humano: {enriched['rating'].mean():.2f}"
        f" | modelo: {enriched['pred_stars'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
