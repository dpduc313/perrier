"""Data Dictionaries through the upload seam: recognising one, matching its names, refusing
a broken one, and the upload reply."""

import pandas as pd
import pytest

from perry import datasets
from perry.data_dictionary import DictionaryRefused

SALES = "region,amount\nNorth,100\nSouth,250\n"


def write(folder, name: str, text: str):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def upload(folder, name: str, text: str) -> list[str]:
    return [r.describe() for r in datasets.upload(write(folder, name, text))]


def descriptions(name: str) -> tuple[str | None, dict[str, str | None]]:
    """A Dataset's own description and its columns', as the registry holds them."""
    (dataset,) = [d for d in datasets.list_datasets() if d.name == name]
    return _text(dataset.description), {
        c.name: _text(c.description) for c in dataset.column_details
    }


def _text(description: datasets.Description | None) -> str | None:
    return description and description.text


def test_a_dictionary_file_describes_the_dataset_named_like_it_and_isnt_a_dataset(
    database, tmp_path
):
    upload(tmp_path, "sales.csv", SALES)

    reply = upload(
        tmp_path,
        "sales.dictionary.csv",
        "column,description,references\n"
        ",One row per sale.,\n"
        "region,Where the sale was made.,\n"
        "amount,Sale amount in VND.,\n",
    )

    assert reply == [
        "sales.dictionary.csv: Data Dictionary, described sales (the Dataset and 2 columns)"
    ]
    assert [d.name for d in datasets.list_datasets()] == ["sales"]
    assert descriptions("sales") == (
        "One row per sale.",
        {"region": "Where the sale was made.", "amount": "Sale amount in VND."},
    )


def test_a_dictionary_can_describe_several_datasets_by_their_files_own_names(database, tmp_path):
    upload(tmp_path, "Sales Q3.csv", "Region,Base Amount\nNorth,100\n")
    upload(tmp_path, "staff.csv", "name,team\nAn,Ops\n")

    reply = upload(
        tmp_path,
        "company.dictionary.csv",
        "dataset,column,description\n"
        "Sales Q3,,One row per sale.\n"
        "Sales Q3,Base Amount,Before tax.\n"
        "staff,Team,The team someone works in.\n",
    )

    assert reply == [
        (
            "company.dictionary.csv: Data Dictionary, described "
            "sales_q3 (the Dataset and 1 column), staff (1 column)"
        )
    ]
    assert descriptions("sales_q3") == (
        "One row per sale.",
        {"region": None, "base_amount": "Before tax."},
    )
    assert descriptions("staff") == (None, {"name": None, "team": "The team someone works in."})


def test_a_dictionary_replaces_the_descriptions_of_the_datasets_it_names_only(database, tmp_path):
    upload(tmp_path, "sales.csv", SALES)
    upload(tmp_path, "staff.csv", "name,team\nAn,Ops\n")
    upload(
        tmp_path,
        "company.dictionary.csv",
        "dataset,column,description\n"
        "sales,,One row per sale.\n"
        "sales,region,Where the sale was made.\n"
        "staff,team,The team someone works in.\n",
    )

    upload(tmp_path, "sales.dictionary.csv", "column,description\namount,In VND.\n")

    assert descriptions("sales") == (None, {"region": None, "amount": "In VND."})
    assert descriptions("staff") == (None, {"name": None, "team": "The team someone works in."})


def test_rows_naming_a_missing_column_or_dataset_are_reported_and_the_rest_attach(
    database, tmp_path
):
    upload(tmp_path, "sales.csv", SALES)

    reply = upload(
        tmp_path,
        "sales.dictionary.csv",
        "dataset,column,description\n"
        ",region,Where the sale was made.\n"
        ",bonus,A typo for amount.\n"
        ",amount,\n",
    )

    assert reply == [
        (
            "sales.dictionary.csv: Data Dictionary, described sales (1 column); not attached: "
            "row 3 (sales has no column bonus), row 4 (sales.amount has no description)"
        )
    ]
    assert descriptions("sales") == (None, {"region": "Where the sale was made.", "amount": None})


def test_rows_for_a_dataset_not_uploaded_yet_wait_for_it(database, tmp_path):
    upload(tmp_path, "sales.csv", SALES)

    reply = upload(
        tmp_path,
        "company.dictionary.csv",
        "dataset,column,description\n"
        "sales,region,Where the sale was made.\n"
        "refunds,,One row per refund.\n"
        "refunds,amount,Refunded amount.\n",
    )

    assert reply == [
        (
            "company.dictionary.csv: Data Dictionary, described sales (1 column); "
            "waiting for refunds (the Dataset and 1 column) to be uploaded"
        )
    ]
    assert [d.name for d in datasets.list_datasets()] == ["sales"]


def test_uploading_a_dataset_attaches_the_rows_waiting_for_it(database, tmp_path):
    upload(
        tmp_path,
        "sales.dictionary.csv",
        "column,description\n,One row per sale.\nregion,Where the sale was made.\n"
        "amount,Sale amount in VND.\nbonus,A column the file doesn't have.\n",
    )

    reply = upload(tmp_path, "sales.csv", SALES)

    assert reply == [
        (
            "sales: created, 2 rows, columns region, amount; the Owner's descriptions on "
            "the Dataset and 2 columns; not attached: sales.dictionary.csv row 5 "
            "(sales has no column bonus)"
        )
    ]
    assert descriptions("sales") == (
        "One row per sale.",
        {"region": "Where the sale was made.", "amount": "Sale amount in VND."},
    )


def test_replacing_a_dataset_keeps_the_descriptions_of_columns_that_survive(database, tmp_path):
    upload(tmp_path, "sales.csv", SALES)
    upload(
        tmp_path,
        "sales.dictionary.csv",
        "column,description\n,One row per sale.\nregion,Where the sale was made.\n"
        "amount,Sale amount in VND.\n",
    )

    reply = upload(tmp_path, "sales.csv", "Region,Total\nNorth,100\n")

    assert reply == [
        (
            "sales: replaced, 1 rows, columns region, total; "
            "the Owner's descriptions on the Dataset and 1 column; "
            "left for the model to describe: 1 column"
        )
    ]
    assert descriptions("sales") == (
        "One row per sale.",
        {"region": "Where the sale was made.", "total": None},
    )


def test_a_replaced_dataset_keeps_its_descriptions_beside_its_inferred_join_hints(
    database, tmp_path
):
    upload(tmp_path, "staff.csv", "staff_id,name\n1,An\n2,Binh\n")
    upload(tmp_path, "sales.csv", "staff_id,amount\n1,100\n")
    upload(tmp_path, "sales.dictionary.csv", "column,description\nstaff_id,Who made the sale.\n")

    upload(tmp_path, "sales.csv", "staff_id,amount\n2,250\n")

    (sales,) = [d for d in datasets.list_datasets() if d.name == "sales"]
    assert descriptions("sales")[1]["staff_id"] == "Who made the sale."
    assert "references staff.staff_id (inferred)" in sales.column_details[0].describe()


def test_a_dictionary_without_a_description_column_is_refused_whole(database, tmp_path):
    upload(tmp_path, "sales.csv", SALES)

    with pytest.raises(DictionaryRefused, match="no description column"):
        upload(tmp_path, "sales.dictionary.csv", "column,meaning\nregion,Where.\n")

    assert descriptions("sales") == (None, {"region": None, "amount": None})


def test_a_dictionary_describing_a_column_twice_is_refused_naming_both_rows(database, tmp_path):
    upload(tmp_path, "sales.csv", SALES)

    with pytest.raises(DictionaryRefused, match="rows 2 and 4 both describe sales.region"):
        upload(
            tmp_path,
            "sales.dictionary.csv",
            'column,description\nregion,Where.\namount,How much.\nRegion,"Where, again."\n',
        )

    assert descriptions("sales") == (None, {"region": None, "amount": None})


def test_a_column_described_twice_is_refused_even_when_one_copy_is_blank(database, tmp_path):
    upload(tmp_path, "sales.csv", SALES)

    with pytest.raises(DictionaryRefused, match="rows 2 and 3 both describe sales.region"):
        upload(tmp_path, "sales.dictionary.csv", "column,description\nregion,\nRegion,Where.\n")

    assert descriptions("sales") == (None, {"region": None, "amount": None})


def test_a_workbooks_dictionary_sheet_describes_its_other_sheets_by_sheet_name(database, tmp_path):
    path = tmp_path / "budget.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Team": ["Ops"], "Spend": [1.5]}).to_excel(
            writer, sheet_name="2025", index=False
        )
        pd.DataFrame(
            {
                "dataset": ["2025", "2025"],
                "column": [None, "Spend"],
                "description": ["One row per team.", "In millions of VND."],
            }
        ).to_excel(writer, sheet_name="Dictionary", index=False)

    reply = [r.describe() for r in datasets.upload(path)]

    assert reply == [
        (
            "budget_2025: created, 1 rows, columns team, spend; the Owner's descriptions on the "
            "Dataset and 1 column; left for the model to describe: 1 column"
        ),
        "budget.xlsx: Data Dictionary, described budget_2025 (the Dataset and 1 column)",
    ]
    assert [d.name for d in datasets.list_datasets()] == ["budget_2025"]
    assert descriptions("budget_2025") == (
        "One row per team.",
        {"team": None, "spend": "In millions of VND."},
    )


def budget_workbook(tmp_path, dictionary: dict):
    path = tmp_path / "budget.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Team": ["Ops"], "Spend": [1.5]}).to_excel(
            writer, sheet_name="2025", index=False
        )
        pd.DataFrame(dictionary).to_excel(writer, sheet_name="dictionary", index=False)
    return path


def test_a_refused_dictionary_sheet_refuses_only_the_dictionary(database, tmp_path):
    path = budget_workbook(tmp_path, {"dataset": ["2025"], "column": ["Spend"], "meaning": ["x"]})

    reply = [r.describe() for r in datasets.upload(path)]

    assert reply == [
        (
            "budget_2025: created, 1 rows, columns team, spend; "
            "left for the model to describe: the Dataset and 2 columns"
        ),
        (
            "budget.xlsx: Data Dictionary refused: it has no description column "
            "(its columns are dataset, column, description and references)"
        ),
    ]
    assert [d.name for d in datasets.list_datasets()] == ["budget_2025"]


def test_a_dictionary_sheet_without_a_dataset_column_is_refused_saying_so(database, tmp_path):
    path = budget_workbook(tmp_path, {"column": ["Spend"], "description": ["In millions."]})

    reply = [r.describe() for r in datasets.upload(path)]

    assert reply[1] == (
        "budget.xlsx: Data Dictionary refused: it has no dataset column "
        "(its columns are dataset, column, description and references)"
    )
    assert descriptions("budget_2025") == (None, {"team": None, "spend": None})


def test_a_dataset_whose_name_ends_in_dictionary_is_still_a_dataset(database, tmp_path):
    reply = upload(tmp_path, "dictionary.csv", "word,meaning\nperry,a helper\n")

    assert reply == [
        (
            "dictionary: created, 1 rows, columns word, meaning; "
            "left for the model to describe: the Dataset and 2 columns"
        )
    ]
    assert [d.name for d in datasets.list_datasets()] == ["dictionary"]


def test_the_datasets_panel_shows_each_datasets_description(perry, tmp_path):
    # perry: fake models with no local model, so no describing starts in the background
    from logins import request_as

    from perry import app

    data = write(tmp_path, "sales.csv", SALES)
    dictionary = write(tmp_path, "sales.dictionary.csv", "column,description\n,One row per sale\n")

    status, shown = app.upload_datasets(request_as(perry.user), [str(data), str(dictionary)])

    assert status.splitlines() == [
        (
            "- sales: created, 2 rows, columns region, amount; "
            "left for the model to describe: the Dataset and 2 columns"
        ),
        "- sales.dictionary.csv: Data Dictionary, described sales (the Dataset)",
    ]
    assert shown == "- **sales** (2 rows): One row per sale. Columns: region, amount"


def test_the_upload_reply_says_why_a_dictionary_was_refused(perry, tmp_path):
    from logins import request_as

    from perry import app

    dictionary = write(tmp_path, "sales.dictionary.csv", "column\nregion\n")

    status, _ = app.upload_datasets(request_as(perry.user), [str(dictionary)])

    assert status == (
        "- sales.dictionary.csv: failed (Data Dictionary refused: it has no description column "
        "(its columns are dataset, column, description and references))"
    )


STAFF = "employee_id,name,manager_id\n1,An,\n2,Bình,1\n3,Chi,1\n"


def references(name: str) -> dict[str, str]:
    """A Dataset's join hints as describe_datasets shows them, by column."""
    (dataset,) = [d for d in datasets.list_datasets() if d.name == name]
    return {c.name: c.join_hint.describe() for c in dataset.column_details if c.join_hint}


def test_a_reference_to_a_missing_column_is_refused_and_the_description_kept(database, tmp_path):
    upload(tmp_path, "staff.csv", STAFF)

    reply = upload(
        tmp_path,
        "staff.dictionary.csv",
        "column,description,references\n"
        "manager_id,Who they report to.,staff.boss_id\n"
        "employee_id,Their number.,people.employee_id\n"
        "name,Their name.,full_name\n",
    )

    assert reply == [
        (
            "staff.dictionary.csv: Data Dictionary, described staff (3 columns); "
            "references refused: row 2 (staff has no column boss_id), "
            "row 4 (full_name isn't written as dataset.column); "
            "references waiting: row 3 (people isn't uploaded yet)"
        )
    ]
    assert references("staff") == {}
    assert descriptions("staff")[1]["manager_id"] == "Who they report to."


def test_a_reference_to_a_dataset_not_uploaded_yet_waits_and_attaches_when_it_arrives(
    database, tmp_path
):
    upload(tmp_path, "reviews.csv", "review_id,reviewer_id\n7,1\n8,5\n")

    reply = upload(
        tmp_path,
        "reviews.dictionary.csv",
        "column,description,references\nreviewer_id,Who wrote it.,staff.employee_id\n",
    )
    waiting = references("reviews")
    arrived = upload(tmp_path, "staff.csv", STAFF)

    assert reply == [
        (
            "reviews.dictionary.csv: Data Dictionary, described reviews (1 column); "
            "references waiting: row 2 (staff isn't uploaded yet)"
        )
    ]
    assert waiting == {}
    assert arrived == [
        (
            "staff: created, 3 rows, columns employee_id, name, manager_id; "
            "left for the model to describe: the Dataset and 3 columns; "
            "values not found: 1 value of reviews.reviewer_id not found in staff.employee_id"
        )
    ]
    assert references("reviews") == {
        "reviewer_id": "references staff.employee_id (declared, 1 value not found there)"
    }


def test_a_dictionary_and_the_datasets_it_joins_can_come_in_any_order(database, tmp_path):
    dictionary = write(
        tmp_path,
        "company.dictionary.csv",
        "dataset,column,description,references\n"
        "reviews,reviewer_id,Who wrote it.,staff.employee_id\n",
    )
    reviews = write(tmp_path, "reviews.csv", "review_id,reviewer_id\n7,1\n8,2\n")
    staff = write(tmp_path, "staff.csv", STAFF)

    replies = [r.describe() for f in (dictionary, reviews, staff) for r in datasets.upload(f)]

    assert replies[1] == (
        "reviews: created, 2 rows, columns review_id, reviewer_id; "
        "the Owner's descriptions on 1 column; "
        "left for the model to describe: the Dataset and 1 column; "
        "declared references waiting: reviews.reviewer_id → staff.employee_id "
        "(staff isn't uploaded yet)"
    )
    assert references("reviews") == {"reviewer_id": "references staff.employee_id (declared)"}


def test_a_reference_none_of_whose_values_appear_is_refused_and_one_with_some_is_kept(
    database, tmp_path
):
    upload(tmp_path, "staff.csv", "employee_id,name\n1,An\n2,Bình\n")
    upload(tmp_path, "reviews.csv", "review_id,reviewer_id,author_id\n7,1,8\n8,5,9\n9,6,9\n")

    reply = upload(
        tmp_path,
        "reviews.dictionary.csv",
        "column,description,references\n"
        "reviewer_id,Who wrote the review.,staff.employee_id\n"
        "author_id,Who was reviewed.,staff.employee_id\n",
    )

    assert reply == [
        (
            "reviews.dictionary.csv: Data Dictionary, described reviews (2 columns); "
            "references refused: row 3 (none of reviews.author_id's values appear in "
            "staff.employee_id); values not found: row 2 (2 values of reviews.reviewer_id "
            "not found in staff.employee_id)"
        )
    ]
    assert references("reviews") == {
        "reviewer_id": "references staff.employee_id (declared, 2 values not found there)"
    }


def test_an_upload_that_leaves_a_declared_reference_unsupported_drops_it_and_says_so(
    database, tmp_path
):
    upload(tmp_path, "staff.csv", STAFF)
    upload(
        tmp_path,
        "staff.dictionary.csv",
        "column,description,references\nmanager_id,Who they report to.,staff.employee_id\n",
    )

    reply = upload(tmp_path, "staff.csv", "employee_id,name,manager_id\n5,Em,4\n")

    assert reply == [
        (
            "staff: replaced, 1 rows, columns employee_id, name, manager_id; "
            "the Owner's descriptions on 1 column; "
            "left for the model to describe: the Dataset and 2 columns; "
            "declared references dropped: staff.manager_id → staff.employee_id "
            "(none of staff.manager_id's values appear in staff.employee_id)"
        )
    ]
    assert references("staff") == {}


def test_a_reference_no_longer_declared_gives_way_to_the_inferred_one(database, tmp_path):
    upload(tmp_path, "staff.csv", STAFF)
    upload(tmp_path, "reviews.csv", "review_id,employee_id\n7,1\n8,2\n")
    upload(
        tmp_path,
        "reviews.dictionary.csv",
        "column,description,references\nemployee_id,Who wrote it.,staff.manager_id\n",
    )
    declared = references("reviews")

    upload(tmp_path, "reviews.dictionary.csv", "column,description\nemployee_id,Who wrote it.\n")

    assert declared == {
        "employee_id": "references staff.manager_id (declared, 1 value not found there)"
    }
    assert references("reviews") == {"employee_id": "references staff.employee_id (inferred)"}


def test_a_reference_waiting_for_its_dataset_is_checked_when_the_dataset_arrives(
    database, tmp_path
):
    upload(tmp_path, "staff.csv", STAFF)
    upload(
        tmp_path,
        "reviews.dictionary.csv",
        "column,description,references\n"
        "reviewer_id,Who wrote it.,staff.employee_id\n"
        "author_id,Who was reviewed.,staff.employee_id\n",
    )

    reply = upload(tmp_path, "reviews.csv", "review_id,reviewer_id,author_id\n7,1,8\n8,5,9\n")

    assert reply == [
        (
            "reviews: created, 2 rows, columns review_id, reviewer_id, author_id; "
            "the Owner's descriptions on 2 columns; "
            "left for the model to describe: the Dataset and 1 column; "
            "declared references dropped: reviews.author_id → staff.employee_id "
            "(none of reviews.author_id's values appear in staff.employee_id); "
            "values not found: 1 value of reviews.reviewer_id not found in staff.employee_id"
        )
    ]
    assert references("reviews") == {
        "reviewer_id": "references staff.employee_id (declared, 1 value not found there)"
    }


def test_an_upload_that_changes_a_declared_references_orphans_reports_them(database, tmp_path):
    upload(tmp_path, "staff.csv", STAFF)
    upload(tmp_path, "reviews.csv", "review_id,reviewer_id\n7,1\n8,2\n")
    upload(
        tmp_path,
        "reviews.dictionary.csv",
        "column,description,references\nreviewer_id,Who wrote it.,staff.employee_id\n",
    )

    reply = upload(tmp_path, "staff.csv", "employee_id,name,manager_id\n1,An,\n")

    assert reply == [
        (
            "staff: replaced, 1 rows, columns employee_id, name, manager_id; "
            "left for the model to describe: the Dataset and 3 columns; "
            "values not found: 1 value of reviews.reviewer_id not found in staff.employee_id"
        )
    ]
