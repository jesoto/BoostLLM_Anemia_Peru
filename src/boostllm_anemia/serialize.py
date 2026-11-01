"""Serialización TabLLM: una fila de ENDES -> viñeta clínica en español."""
from __future__ import annotations

import pandas as pd

TEMPLATE = (
    "Niño/a de {HC1:.0f} meses, residente a {HV040:.0f} m s. n. m. "
    "{zona}. La madre tiene {mother_age:.0f} años y tuvo {anc_visits:.0f} controles prenatales. "
    "¿Tiene anemia? Responde solo Sí o No."
)


def serialize_row(row: pd.Series) -> str:
    zona = "Zona rural" if row.get("urban", 1) == 0 else "Zona urbana"
    d = {"HC1": row["HC1"], "HV040": row["HV040"], "zona": zona,
         "mother_age": row.get("mother_age", float("nan")),
         "anc_visits": row.get("anc_visits", float("nan"))}
    return TEMPLATE.format(**d)


def serialize(df: pd.DataFrame) -> list[str]:
    return [serialize_row(r) for _, r in df.iterrows()]
