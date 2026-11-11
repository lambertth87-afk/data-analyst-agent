"""Outils fixes de l'agent analyste de données.

Chaque outil est une fonction déterministe : charger un CSV, décrire ses
colonnes, filtrer des lignes, calculer des agrégats par groupe. Aucune
exécution de code arbitraire : le modèle ne peut que composer ces outils.
"""

import base64
import io
import json
import uuid
from collections import OrderedDict
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

MAX_DATASETS = 8
PREVIEW_ROWS = 5
MAX_FILTER_ROWS = 100
MAX_AGGREGATE_GROUPS = 500

_DATASETS: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
_META: dict[str, dict[str, Any]] = {}

FilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "in",
    "is_null",
    "not_null",
]

AggFunction = Literal["count", "nunique", "sum", "mean", "median", "min", "max"]


class Filter(BaseModel):
    """Une condition de filtre appliquée aux lignes."""

    column: Annotated[str, Field(description="Nom exact de la colonne.")]
    operator: Annotated[FilterOperator, Field(description="Opérateur de comparaison.")]
    value: Annotated[
        Any,
        Field(
            description=(
                "Valeur de comparaison. Ignorée pour is_null/not_null. "
                "Pour `in`, fournir une liste de valeurs."
            )
        ),
    ] = None


class Metric(BaseModel):
    """Un agrégat calculé par groupe."""

    column: Annotated[
        str, Field(description="Colonne à agréger (ignorée pour `count`).")
    ]
    function: Annotated[AggFunction, Field(description="Fonction d'agrégation.")]
    alias: Annotated[
        str | None, Field(description="Nom de la colonne résultat (optionnel).")
    ] = None


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False, date_format="iso"))


def _get_dataset(dataset_id: str) -> pd.DataFrame:
    df = _DATASETS.get(dataset_id)
    if df is None:
        raise agents.ToolError(
            f"Dataset '{dataset_id}' introuvable (expiré ou session recyclée). "
            "Retéléchargez le fichier depuis Google Drive avec download_file_content "
            "puis rechargez-le avec load_csv avant de continuer."
        )
    _DATASETS.move_to_end(dataset_id)
    return df


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise agents.ToolError(
            f"Colonnes introuvables: {missing}. Colonnes disponibles: {list(df.columns)}."
        )


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    result = item() if callable(item) else value
    if isinstance(result, float) and result != result:
        return None
    return result


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def _compare_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    if operator == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False)
    if pd.api.types.is_numeric_dtype(series):
        try:
            ref = float(value)
        except (TypeError, ValueError):
            raise agents.ToolError(
                f"La colonne '{series.name}' est numérique ; la valeur '{value}' n'est pas un nombre."
            ) from None
        target = series
    else:
        ref = str(value)
        target = series.astype(str)
    if operator == "eq":
        return target == ref
    if operator == "ne":
        return target != ref
    if operator == "gt":
        return target > ref
    if operator == "gte":
        return target >= ref
    if operator == "lt":
        return target < ref
    if operator == "lte":
        return target <= ref
    raise agents.ToolError(f"Opérateur inconnu: '{operator}'.")


def _as_filters(filters: list[Filter | dict[str, Any]] | None) -> list[Filter]:
    return [f if isinstance(f, Filter) else Filter.model_validate(f) for f in filters or []]


def _as_metrics(metrics: list[Metric | dict[str, Any]]) -> list[Metric]:
    return [m if isinstance(m, Metric) else Metric.model_validate(m) for m in metrics]


def _apply_filters(df: pd.DataFrame, filters: list[Filter] | None) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for f in filters:
        _require_columns(df, [f.column])
        series = df[f.column]
        if f.operator == "is_null":
            mask &= series.isna()
        elif f.operator == "not_null":
            mask &= series.notna()
        elif f.operator == "in":
            values = list(f.value) if isinstance(f.value, (list, tuple, set)) else [f.value]
            if pd.api.types.is_numeric_dtype(series):
                try:
                    values = [float(v) for v in values]
                except (TypeError, ValueError):
                    raise agents.ToolError(
                        f"La colonne '{f.column}' est numérique ; les valeurs de `in` doivent être des nombres."
                    ) from None
            else:
                values = [str(v) for v in values]
                mask &= series.astype(str).isin(values)
                continue
            mask &= series.isin(values)
        else:
            mask &= _compare_mask(series, f.operator, f.value)
    return df[mask]


def _profile_column(name: str, series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    profile: dict[str, Any] = {
        "name": name,
        "kind": _column_kind(series),
        "dtype": str(series.dtype),
        "non_null": int(series.notna().sum()),
        "nulls": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
        "sample": json.loads(non_null.head(5).to_json(orient="values", date_format="iso")),
    }
    if _column_kind(series) == "numeric" and len(non_null):
        profile["min"] = _scalar(non_null.min())
        profile["max"] = _scalar(non_null.max())
        profile["mean"] = round(float(non_null.mean()), 6)
        profile["median"] = _scalar(non_null.median())
        profile["sum"] = _scalar(non_null.sum())
    else:
        counts = series.astype(str).value_counts(dropna=True).head(5)
        profile["top_values"] = json.loads(counts.to_json(force_ascii=False))
    return profile


@agents.tool(
    annotations=agents.ToolAnnotations(read_only_hint=True),
    title="Charger un CSV",
)
def load_csv(
    csv_base64: Annotated[
        str | None,
        Field(description="Contenu du fichier CSV encodé en base64 (sortie de download_file_content)."),
    ] = None,
    csv_text: Annotated[
        str | None,
        Field(description="Contenu du fichier CSV en texte brut (alternative à csv_base64)."),
    ] = None,
    name: Annotated[str, Field(description="Nom lisible du jeu de données.")] = "dataset",
    source: Annotated[
        str | None,
        Field(description="Source du fichier (lien ou identifiant Google Drive), pour s'en souvenir."),
    ] = None,
) -> dict[str, Any]:
    """Charge un CSV et renvoie son identifiant de dataset, ses dimensions et un aperçu.

    Appeler cet outil une fois par fichier ; conserver le `dataset_id` renvoyé pour
    les appels suivants (describe_columns, filter_rows, aggregate_by_group).
    """
    if (csv_base64 is None) == (csv_text is None):
        raise agents.ToolError("Fournir exactement un des deux champs: csv_base64 ou csv_text.")
    if csv_base64 is not None:
        try:
            raw = base64.b64decode(csv_base64, validate=False)
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        except Exception:
            raise agents.ToolError("csv_base64 n'est pas du base64 valide.") from None
    else:
        text = csv_text or ""
    if not text.strip():
        raise agents.ToolError("Le fichier CSV est vide.")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise agents.ToolError(f"Lecture CSV impossible: {exc}") from None
    if df.empty:
        raise agents.ToolError("Le CSV ne contient aucune ligne de données.")
    dataset_id = uuid.uuid4().hex[:12]
    _DATASETS[dataset_id] = df
    _META[dataset_id] = {"name": name, "source": source}
    while len(_DATASETS) > MAX_DATASETS:
        old_id, _ = _DATASETS.popitem(last=False)
        _META.pop(old_id, None)
    columns = [_profile_column(c, df[c]) for c in df.columns]
    return {
        "dataset_id": dataset_id,
        "name": name,
        "source": source,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": columns,
        "preview": _json_records(df.head(PREVIEW_ROWS)),
    }


@agents.tool(
    annotations=agents.ToolAnnotations(read_only_hint=True),
    title="Décrire les colonnes",
)
def describe_columns(
    dataset_id: Annotated[str, Field(description="Identifiant renvoyé par load_csv.")],
) -> dict[str, Any]:
    """Décris chaque colonne : type, valeurs manquantes, unicités, stats de base."""
    df = _get_dataset(dataset_id)
    meta = _META.get(dataset_id, {})
    return {
        "dataset_id": dataset_id,
        "name": meta.get("name"),
        "n_rows": int(len(df)),
        "columns": [_profile_column(c, df[c]) for c in df.columns],
    }


@agents.tool(
    annotations=agents.ToolAnnotations(read_only_hint=True),
    title="Filtrer des lignes",
)
def filter_rows(
    dataset_id: Annotated[str, Field(description="Identifiant renvoyé par load_csv.")],
    filters: Annotated[list[Filter], Field(description="Conditions combinées en ET.")],
    order_by: Annotated[
        str | None, Field(description="Colonne de tri (optionnel).")
    ] = None,
    ascending: Annotated[bool, Field(description="Tri croissant.")] = True,
    limit: Annotated[
        int, Field(description="Nombre maximal de lignes renvoyées (1-100).")
    ] = 20,
) -> dict[str, Any]:
    """Filtre les lignes du dataset et renvoie le nombre de correspondances et un échantillon."""
    df = _get_dataset(dataset_id)
    filters = _as_filters(filters)
    limit = max(1, min(int(limit), MAX_FILTER_ROWS))
    filtered = _apply_filters(df, filters)
    if order_by:
        _require_columns(filtered, [order_by])
        filtered = filtered.sort_values(order_by, ascending=ascending)
    return {
        "dataset_id": dataset_id,
        "matched_rows": int(len(filtered)),
        "returned": int(min(len(filtered), limit)),
        "rows": _json_records(filtered.head(limit)),
    }


@agents.tool(
    annotations=agents.ToolAnnotations(read_only_hint=True),
    title="Agrégats par groupe",
)
def aggregate_by_group(
    dataset_id: Annotated[str, Field(description="Identifiant renvoyé par load_csv.")],
    group_by: Annotated[list[str], Field(description="Colonnes de regroupement.")],
    metrics: Annotated[list[Metric], Field(description="Agrégats à calculer par groupe.")],
    filters: Annotated[
        list[Filter] | None, Field(description="Filtres appliqués avant agrégation (optionnel).")
    ] = None,
    sort_by: Annotated[
        str | None,
        Field(description="Colonne résultat pour le tri (groupe ou alias de métrique)."),
    ] = None,
    ascending: Annotated[bool, Field(description="Tri croissant.")] = False,
    limit: Annotated[
        int, Field(description="Nombre maximal de groupes renvoyés (1-500).")
    ] = 50,
) -> dict[str, Any]:
    """Calcule des agrégats (somme, moyenne, compte...) par groupe, avec filtres optionnels."""
    df = _get_dataset(dataset_id)
    metrics = _as_metrics(metrics)
    if not group_by:
        raise agents.ToolError("group_by doit contenir au moins une colonne.")
    if not metrics:
        raise agents.ToolError("metrics doit contenir au moins un agrégat.")
    _require_columns(df, group_by)
    limit = max(1, min(int(limit), MAX_AGGREGATE_GROUPS))
    filtered = _apply_filters(df, _as_filters(filters))

    aggregations: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for metric in metrics:
        alias = metric.alias or (
            f"{metric.column}_{metric.function}" if metric.function != "count" else "count"
        )
        if alias in aggregations:
            raise agents.ToolError(f"Doublon d'alias de métrique: '{alias}'.")
        if metric.function != "count":
            _require_columns(df, [metric.column])
        aggregations[alias] = (metric.column, metric.function)
        aliases[alias] = alias

    try:
        named = {
            alias: pd.NamedAgg(column=column, aggfunc=function)
            for alias, (column, function) in aggregations.items()
        }
        grouped = filtered.groupby(group_by, dropna=False, observed=True).agg(**named)
    except Exception as exc:
        raise agents.ToolError(f"Agrégation impossible: {exc}") from None
    grouped = grouped.reset_index()
    if sort_by:
        if sort_by not in grouped.columns:
            raise agents.ToolError(
                f"Colonne de tri '{sort_by}' inconnue. Colonnes résultat: {list(grouped.columns)}."
            )
        grouped = grouped.sort_values(sort_by, ascending=ascending)
    return {
        "dataset_id": dataset_id,
        "n_groups": int(len(grouped)),
        "returned": int(min(len(grouped), limit)),
        "rows": _json_records(grouped.head(limit)),
    }
