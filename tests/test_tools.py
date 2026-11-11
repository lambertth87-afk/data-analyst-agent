import base64
import json

import pytest

from tools import aggregate_by_group, describe_columns, filter_rows, load_csv

CSV = """region,produit,montant,quantite,date
Nord,A,100.5,2,2024-01-15
Sud,A,50.0,1,2024-01-20
Nord,B,200.0,3,2024-02-01
Est,B,,4,2024-02-10
Sud,C,75.25,5,2024-03-05
Nord,A,300.0,6,2024-03-15
"""


@pytest.fixture()
def dataset_id():
    result = load_csv(csv_text=CSV, name="ventes", source="https://drive.google.com/file/d/XYZ/view")
    return result["dataset_id"]


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def test_load_csv_returns_schema_and_preview():
    result = load_csv(csv_base64=b64(CSV), name="ventes", source="drive://XYZ")
    assert result["name"] == "ventes"
    assert result["n_rows"] == 6
    assert result["n_columns"] == 5
    kinds = {c["name"]: c["kind"] for c in result["columns"]}
    assert kinds["region"] == "text"
    assert kinds["montant"] == "numeric"
    assert kinds["quantite"] == "numeric"
    assert kinds["date"] == "text"
    assert len(result["preview"]) == 5
    assert result["columns"][2]["nulls"] == 1


def test_load_csv_rejects_bad_input():
    with pytest.raises(Exception, match="exactement un"):
        load_csv(csv_text=CSV, csv_base64=b64(CSV))
    with pytest.raises(Exception, match="base64"):
        load_csv(csv_base64="!!!pas-base64!!!")


def test_describe_columns(dataset_id):
    result = describe_columns(dataset_id)
    assert result["n_rows"] == 6
    montant = next(c for c in result["columns"] if c["name"] == "montant")
    assert montant["non_null"] == 5
    assert montant["nulls"] == 1
    assert montant["sum"] == pytest.approx(725.75)
    assert montant["min"] == pytest.approx(50.0)
    assert montant["max"] == pytest.approx(300.0)
    region = next(c for c in result["columns"] if c["name"] == "region")
    assert region["top_values"]["Nord"] == 3


def test_filter_rows(dataset_id):
    result = filter_rows(
        dataset_id,
        filters=[{"column": "region", "operator": "eq", "value": "Nord"}],
    )
    assert result["matched_rows"] == 3
    assert all(row["region"] == "Nord" for row in result["rows"])

    result = filter_rows(
        dataset_id,
        filters=[{"column": "montant", "operator": "gte", "value": 100}],
        order_by="montant",
        ascending=False,
    )
    assert result["matched_rows"] == 3
    assert result["rows"][0]["montant"] == 300.0

    result = filter_rows(
        dataset_id,
        filters=[{"column": "produit", "operator": "in", "value": ["A", "C"]}],
    )
    assert result["matched_rows"] == 4

    result = filter_rows(
        dataset_id,
        filters=[{"column": "montant", "operator": "is_null", "value": None}],
    )
    assert result["matched_rows"] == 1


def test_filter_unknown_column_raises_tool_error(dataset_id):
    from mistralai import agents

    with pytest.raises(agents.ToolError, match="introuvables"):
        filter_rows(dataset_id, filters=[{"column": "nope", "operator": "eq", "value": 1}])


def test_unknown_dataset_raises_tool_error():
    from mistralai import agents

    with pytest.raises(agents.ToolError, match="introuvable"):
        describe_columns("deadbeef")


def test_aggregate_by_group(dataset_id):
    result = aggregate_by_group(
        dataset_id,
        group_by=["region"],
        metrics=[
            {"column": "montant", "function": "sum", "alias": "total"},
            {"column": "montant", "function": "mean", "alias": "moyenne"},
            {"column": "produit", "function": "count", "alias": "ventes"},
        ],
        sort_by="total",
        ascending=False,
    )
    assert result["n_groups"] == 3
    rows = {r["region"]: r for r in result["rows"]}
    assert rows["Nord"]["total"] == pytest.approx(600.5)
    assert rows["Nord"]["ventes"] == 3
    assert rows["Sud"]["total"] == pytest.approx(125.25)
    assert result["rows"][0]["region"] == "Nord"


def test_aggregate_with_filters(dataset_id):
    result = aggregate_by_group(
        dataset_id,
        group_by=["produit"],
        metrics=[{"column": "quantite", "function": "sum", "alias": "unites"}],
        filters=[{"column": "region", "operator": "eq", "value": "Nord"}],
    )
    rows = {r["produit"]: r for r in result["rows"]}
    assert rows["A"]["unites"] == 8
    assert rows["B"]["unites"] == 3
    assert "C" not in rows


def test_aggregate_validates_inputs(dataset_id):
    from mistralai import agents

    with pytest.raises(agents.ToolError, match="group_by"):
        aggregate_by_group(dataset_id, group_by=[], metrics=[{"column": "montant", "function": "sum"}])
    with pytest.raises(agents.ToolError, match="introuvables"):
        aggregate_by_group(
            dataset_id, group_by=["nope"], metrics=[{"column": "montant", "function": "sum"}]
        )


def test_dataset_eviction():
    from tools import MAX_DATASETS, _DATASETS

    ids = []
    for i in range(MAX_DATASETS + 2):
        result = load_csv(csv_text=CSV, name=f"ds-{i}")
        ids.append(result["dataset_id"])
    assert ids[0] not in _DATASETS
    assert ids[-1] in _DATASETS
    assert len(_DATASETS) <= MAX_DATASETS


def test_json_records_are_json_serializable(dataset_id):
    result = filter_rows(dataset_id, filters=[], limit=100)
    json.dumps(result["rows"])
