"""Descriptions the model writes in the background for what the Data Dictionary
leaves out, with the fake local summary model or the fake company model, seen through
Turns and the upload reply."""

import json
from dataclasses import replace

import httpx
import pytest
from fakes import FakeSummaryModel, fake_provider
from test_turns_analysis import ask_to_describe, ask_to_list, upload_csv

from perry import dataset_descriptions, models, usage

SALES = "region,amount\nNorth,100\nSouth,250\nNorth,50\n"


def guesses(dataset: str | None = None, **columns: str) -> str:
    """The model's reply: one sentence for the Dataset and one per column."""
    return json.dumps({"dataset": dataset, "columns": columns})


@pytest.fixture
def local_model(perry):
    """The local summary model the background job calls, beside the scripted chat model."""
    model = FakeSummaryModel()
    with models.use(fake_provider(perry.model, summary=model)):
        yield model


def test_the_models_descriptions_reach_the_agent_marked_as_guesses(perry, local_model, tmp_path):
    upload_csv(tmp_path, "sales.csv", SALES)
    local_model.replies.append(
        guesses("One row per sale.", region="Sales region.", amount="Sale amount.")
    )

    dataset_descriptions.describe_pending()

    assert ask_to_describe(perry, "sales").splitlines() == [
        "sales (3 rows, from sales.csv)",
        "One row per sale. (the model's guess)",
        (
            "- region (text): Sales region. (the model's guess) [no empty values; "
            "values 'North', 'South']"
        ),
        "- amount (bigint): Sale amount. (the model's guess) [no empty values; min 50, max 250]",
    ]
    assert ask_to_list(perry) == (
        "sales (3 rows, from sales.csv): One row per sale. (the model's guess) "
        "Columns: region (text), amount (bigint)"
    )


def test_the_model_never_overwrites_the_owners_descriptions_and_the_owner_replaces_its_guesses(
    perry, local_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)
    upload_csv(tmp_path, "sales.dictionary.csv", "column,description\namount,Sale amount in VND\n")
    local_model.replies.append(
        guesses("One row per sale.", region="Sales region.", amount="Amount in dollars.")
    )

    dataset_descriptions.describe_pending()

    asked = local_model.calls[0][-1].text
    assert "region (text)" in asked
    assert "amount" not in asked.split("Describe these columns:")[1].split("Its other")[0]
    assert ask_to_describe(perry, "sales").splitlines()[1:] == [
        "One row per sale. (the model's guess)",
        (
            "- region (text): Sales region. (the model's guess) [no empty values; "
            "values 'North', 'South']"
        ),
        "- amount (bigint): Sale amount in VND. [no empty values; min 50, max 250]",
    ]

    upload_csv(
        tmp_path,
        "sales.dictionary.csv",
        "column,description\n,One sale.\nregion,Where it was sold\namount,Sale amount in VND\n",
    )

    assert ask_to_describe(perry, "sales").splitlines()[1:3] == [
        "One sale.",
        "- region (text): Where it was sold. [no empty values; values 'North', 'South']",
    ]


def test_a_reupload_sends_its_new_columns_and_the_guessed_dataset_description_to_the_model(
    perry, local_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)
    local_model.replies.append(
        guesses("One row per sale.", region="Sales region.", amount="Sale amount.")
    )
    dataset_descriptions.describe_pending()

    upload_csv(tmp_path, "sales.csv", "region,amount,units\nNorth,100,2\nEast,70,1\n")

    assert ask_to_describe(perry, "sales").splitlines()[1:] == [
        "No description yet.",
        (
            "- region (text): Sales region. (the model's guess) [no empty values; "
            "values 'East', 'North']"
        ),
        "- amount (bigint): Sale amount. (the model's guess) [no empty values; min 70, max 100]",
        "- units (bigint): No description yet. [no empty values; min 1, max 2]",
    ]

    local_model.replies.append(guesses("One sale of some units.", units="Units sold."))
    dataset_descriptions.describe_pending()

    asked = local_model.calls[1][-1].text
    assert "- units (bigint): no empty values; min 1, max 2" in asked
    assert "- region" not in asked and "- amount" not in asked
    assert "Also say what one row of the table is." in asked
    described = ask_to_describe(perry, "sales")
    assert described.splitlines()[1] == "One sale of some units. (the model's guess)"
    assert "- units (bigint): Units sold. (the model's guess)" in described


def test_for_a_dataset_the_owner_described_only_columns_are_asked_for_and_a_bare_reply_counts(
    perry, local_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)
    upload_csv(tmp_path, "sales.dictionary.csv", "column,description\n,One row per sale.\n")
    # Just the columns' object, as models asked for one key often reply.
    local_model.replies.append(json.dumps({"region": "Sales region.", "amount": "Sale amount."}))

    dataset_descriptions.describe_pending()

    asked = local_model.calls[0][-1].text
    assert "null" not in asked and "one row" not in asked.lower()
    assert ask_to_describe(perry, "sales").splitlines()[1:] == [
        "One row per sale.",
        (
            "- region (text): Sales region. (the model's guess) [no empty values; "
            "values 'North', 'South']"
        ),
        "- amount (bigint): Sale amount. (the model's guess) [no empty values; min 50, max 250]",
    ]


def test_a_reupload_keeps_the_owners_dataset_description(perry, local_model, tmp_path):
    upload_csv(tmp_path, "sales.csv", SALES)
    upload_csv(tmp_path, "sales.dictionary.csv", "column,description\n,One row per sale.\n")

    upload_csv(tmp_path, "sales.csv", "region,amount\nNorth,100\n")

    assert ask_to_describe(perry, "sales").splitlines()[1] == "One row per sale."


def test_a_failing_model_leaves_descriptions_pending_and_the_next_start_retries(
    perry, local_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)
    local_model.replies.append(httpx.ConnectError("Ollama isn't running"))

    dataset_descriptions.describe_pending()

    described = ask_to_describe(perry, "sales")
    assert "- amount (bigint): No description yet. [no empty values; min 50, max 250]" in (
        described
    )
    assert perry.model.steps == []  # the Turn ran: the Dataset stays usable

    dataset_descriptions.retry_failed()  # what the next start does before describing again
    local_model.replies.append(
        guesses("One row per sale.", region="Sales region.", amount="Sale amount.")
    )
    dataset_descriptions.describe_pending()

    assert ask_to_describe(perry, "sales").splitlines()[1:] == [
        "One row per sale. (the model's guess)",
        (
            "- region (text): Sales region. (the model's guess) [no empty values; "
            "values 'North', 'South']"
        ),
        "- amount (bigint): Sale amount. (the model's guess) [no empty values; min 50, max 250]",
    ]


def test_a_column_the_models_reply_leaves_out_stays_pending_and_the_next_start_retries_it(
    perry, local_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)
    local_model.replies.append(guesses("One row per sale.", amount="Sale amount."))

    dataset_descriptions.describe_pending()

    assert ask_to_describe(perry, "sales").splitlines()[1:] == [
        "One row per sale. (the model's guess)",
        "- region (text): No description yet. [no empty values; values 'North', 'South']",
        "- amount (bigint): Sale amount. (the model's guess) [no empty values; min 50, max 250]",
    ]

    dataset_descriptions.retry_failed()
    local_model.replies.append(guesses(region="Sales region."))
    dataset_descriptions.describe_pending()

    asked = local_model.calls[1][-1].text
    assert "- region (text)" in asked and "- amount" not in asked
    assert "- region (text): Sales region. (the model's guess)" in ask_to_describe(perry, "sales")


def test_local_description_calls_are_in_the_usage_ledger_at_zero_cost(
    perry, local_model, tmp_path, caplog
):
    upload_csv(tmp_path, "sales.csv", SALES)
    local_model.replies.append(guesses("One row per sale.", region="Region.", amount="Amount."))

    dataset_descriptions.describe_pending()

    # The fake reports 100 tokens in and 20 out, as qwen3:8b, priced by prices.toml.
    assert usage.today(kind="dataset_description") == usage.Usage(tokens=120, cost_usd=0.0)
    assert "'qwen3:8b'" not in caplog.text  # priced at zero, not just unpriced


@pytest.fixture
def company_model(perry, local_model, monkeypatch):
    """The company's model (inet-qwen), chosen to describe Datasets
    (PERRY_DATASET_DESCRIPTION_MODEL=inet-qwen), beside the local model."""
    model = FakeSummaryModel(model="knowledge")  # named as inet-qwen's server names it
    monkeypatch.setattr(models, "DATASET_DESCRIPTION_MODEL", models.INET_QWEN)
    provider = replace(fake_provider(perry.model, summary=local_model), company_chat=lambda: model)
    with models.use(provider):
        yield model


def test_on_the_company_model_it_describes_datasets_and_the_local_model_isnt_asked(
    perry, local_model, company_model, tmp_path, caplog
):
    upload_csv(tmp_path, "sales.csv", SALES)
    company_model.replies.append(
        guesses("One row per sale.", region="Sales region.", amount="Sale amount.")
    )

    dataset_descriptions.describe_pending()

    assert local_model.calls == []
    assert ask_to_describe(perry, "sales").splitlines()[1:] == [
        "One row per sale. (the model's guess)",
        (
            "- region (text): Sales region. (the model's guess) [no empty values; "
            "values 'North', 'South']"
        ),
        "- amount (bigint): Sale amount. (the model's guess) [no empty values; min 50, max 250]",
    ]
    assert usage.today(kind="dataset_description") == usage.Usage(tokens=120, cost_usd=0.0)
    assert "'knowledge'" not in caplog.text  # priced at zero, not just unpriced


def test_one_setting_chooses_the_model_that_describes_datasets_by_default_the_company_one(
    monkeypatch,
):
    monkeypatch.setattr(models, "INET_QWEN_BASE_URL", "https://inet.example/v1")
    assert models._dataset_description_model("") == "inet-qwen"
    assert models._dataset_description_model(" Local ") == "local"

    monkeypatch.setattr(models, "INET_QWEN_BASE_URL", "")
    assert models._dataset_description_model("") == "local"
    with pytest.raises(
        ValueError, match="PERRY_DATASET_DESCRIPTION_MODEL.*PERRY_INET_QWEN_BASE_URL"
    ):
        models._dataset_description_model("inet-qwen")
    with pytest.raises(ValueError, match="PERRY_DATASET_DESCRIPTION_MODEL is qwen3:8b"):
        models._dataset_description_model("qwen3:8b")


def test_an_unreachable_company_model_leaves_the_dataset_to_the_next_start_not_the_local_model(
    perry, local_model, company_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)
    company_model.replies.append(httpx.ConnectError("inet-qwen isn't reachable"))

    dataset_descriptions.describe_pending()

    assert local_model.calls == []
    assert ask_to_describe(perry, "sales").splitlines()[1] == "No description yet."

    dataset_descriptions.retry_failed()  # what the next start does before describing again
    company_model.replies.append(
        guesses("One row per sale.", region="Sales region.", amount="Sale amount.")
    )
    dataset_descriptions.describe_pending()

    assert ask_to_describe(perry, "sales").splitlines()[1] == (
        "One row per sale. (the model's guess)"
    )


def test_the_upload_reply_says_what_waits_for_the_model_and_the_panel_marks_its_guesses(
    perry, tmp_path
):
    from logins import request_as

    from perry import app

    data = tmp_path / "sales.csv"
    data.write_text(SALES, encoding="utf-8")
    dictionary = tmp_path / "sales.dictionary.csv"
    dictionary.write_text("column,description\namount,Sale amount in VND\n", encoding="utf-8")

    # The perry fixture's provider has no local model, so nothing is described meanwhile.
    status, shown = app.upload_datasets(request_as(perry.user), [str(data), str(dictionary)])
    reply, _ = app.upload_datasets(request_as(perry.user), [str(data)])

    assert status.splitlines()[0] == (
        "- sales: created, 3 rows, columns region, amount; "
        "left for the model to describe: the Dataset and 2 columns"
    )
    assert reply == (
        "- sales: replaced, 3 rows, columns region, amount; the Owner's descriptions on "
        "1 column; left for the model to describe: the Dataset and 1 column"
    )
    assert shown == "- **sales** (3 rows): No description yet. Columns: region, amount"

    local_model = FakeSummaryModel(replies=[guesses("One row per sale.", region="Sales region.")])
    with models.use(fake_provider(perry.model, summary=local_model)):
        dataset_descriptions.describe_pending()

    assert app.dataset_list() == (
        "- **sales** (3 rows): One row per sale. (the model's guess) Columns: region, amount; "
        "the model guessed the descriptions of region"
    )


def test_until_the_model_has_described_a_dataset_its_descriptions_read_as_pending(
    perry, local_model, tmp_path
):
    upload_csv(tmp_path, "sales.csv", SALES)  # the upload doesn't wait for the model

    assert local_model.calls == []
    assert ask_to_describe(perry, "sales").splitlines() == [
        "sales (3 rows, from sales.csv)",
        "No description yet.",
        "- region (text): No description yet. [no empty values; values 'North', 'South']",
        "- amount (bigint): No description yet. [no empty values; min 50, max 250]",
    ]
    assert ask_to_list(perry) == (
        "sales (3 rows, from sales.csv): No description yet. "
        "Columns: region (text), amount (bigint)"
    )
