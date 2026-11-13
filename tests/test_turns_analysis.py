"""Turns that involve the Analysis Agent, with a scripted fake chat model and real Datasets."""

from pathlib import Path

from fakes import call, last_message

from perry import datasets

TOTALS_SQL = "SELECT region, sum(amount) AS total FROM sales GROUP BY region ORDER BY region"


def add_sales_dataset(tmp_path) -> None:
    path = tmp_path / "sales.csv"
    path.write_text("region,amount\nNorth,100\nSouth,250\nNorth,50\n", encoding="utf-8")
    datasets.upload(path)


def test_data_question_is_answered_by_the_analysis_agent_showing_the_query_used(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="What are total sales per region?"),  # Supervisor
        call("run_sql", query=TOTALS_SQL),  # Analysis Agent
        lambda messages: f"Totals: {messages[-1].text}",  # Analysis Agent reports
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("What were total sales per region?")

    assert perry.consulted() == ["ask_analysis_agent"]
    assert "North | 150" in reply
    assert "South | 250" in reply
    assert TOTALS_SQL in reply


def test_the_analysis_agent_sees_which_datasets_exist_and_their_columns(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="Which Datasets do I have?"),
        call("list_datasets"),
        last_message,
        last_message,
    )

    reply = perry.send("Which Datasets do I have?")

    # Without a local model (the fake provider has none), nothing is described yet.
    assert reply == (
        "sales (3 rows, from sales.csv): No description yet. Columns: region (text), amount (bigint)"
    )


def test_a_failed_query_gives_a_clear_message_instead_of_crashing(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="Total sales?"),
        call("run_sql", query="DELETE FROM sales"),
        last_message,  # Analysis Agent reports the failure
        last_message,
    )

    reply = perry.send("Delete my sales data and tell me the total")

    assert reply.startswith("The query failed: ")
    assert datasets.run_query("SELECT count(*) FROM sales").rows == [(3,)]


FAKE_PNG = b"\x89PNG\r\n\x1a\nfake chart"

CHART_CODE = """import pandas as pd
sales = pd.read_csv("sales.csv")
sales.groupby("region")["amount"].sum().plot.bar().figure.savefig("totals.png")"""


def test_a_computation_question_runs_python_and_the_answer_shows_the_chart(
    perry, tmp_path, monkeypatch
):
    from perry import charts, sandbox

    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    add_sales_dataset(tmp_path)
    ran = []

    def fake_sandbox(code, files):
        ran.append((code, files))
        return sandbox.Result(output="North 150\nSouth 250\n", charts=[FAKE_PNG])

    monkeypatch.setattr(sandbox, "run", fake_sandbox)
    perry.script(
        call("ask_analysis_agent", question="Chart total sales per region"),  # Supervisor
        call("run_python", code=CHART_CODE, inputs={"sales.csv": "SELECT * FROM sales"}),
        "Totals: North 150, South 250.",  # Analysis Agent, not mentioning the chart
        "North sold 150 and South 250.",  # Supervisor, not mentioning it either
    )

    reply = perry.send("Plot my total sales per region")

    assert perry.consulted() == ["ask_analysis_agent"]
    code, files = ran[0]
    assert code == CHART_CODE
    assert files["sales.csv"].decode().splitlines() == [
        "region,amount",
        "North,100",
        "South,250",
        "North,50",
    ]
    shown = charts.chat_messages(reply)
    assert shown[-1] == {"role": "assistant", "content": "North sold 150 and South 250."}
    chart_files = [m["content"]["path"] for m in shown[:-1]]
    assert len(chart_files) == 1
    assert Path(chart_files[0]).read_bytes() == FAKE_PNG
    # Reopening the Thread shows the chart again.
    assert charts.chat_messages(perry.history()[-1]["content"]) == shown


def test_a_failing_script_is_reported_so_the_user_knows_to_rephrase(perry, tmp_path, monkeypatch):
    from perry import sandbox

    add_sales_dataset(tmp_path)
    monkeypatch.setattr(
        sandbox,
        "run",
        lambda code, files: sandbox.Result(output="", error="The script failed:\nKeyError: 'x'"),
    )
    perry.script(
        call("ask_analysis_agent", question="Forecast sales"),
        call("run_python", code="print(1)", inputs={"sales.csv": "SELECT * FROM sales"}),
        last_message,
        last_message,
    )

    reply = perry.send("Forecast my sales")

    assert "KeyError: 'x'" in reply


def test_an_input_named_without_an_extension_is_saved_as_a_csv_file(perry, tmp_path, monkeypatch):
    from perry import sandbox

    add_sales_dataset(tmp_path)
    ran = []
    monkeypatch.setattr(
        sandbox, "run", lambda code, files: ran.append(files) or sandbox.Result(output="ok")
    )
    perry.script(
        call("ask_analysis_agent", question="Chart sales"),
        # Models often name an input without ".csv", then read "<name>.csv".
        call("run_python", code="print(1)", inputs={"sales": "SELECT * FROM sales"}),
        last_message,
        last_message,
    )

    perry.send("Chart my sales")

    assert list(ran[0]) == ["sales.csv"]


def test_a_failing_script_is_told_which_input_files_it_had(perry, tmp_path, monkeypatch):
    from perry import sandbox

    add_sales_dataset(tmp_path)
    monkeypatch.setattr(
        sandbox,
        "run",
        lambda code, files: sandbox.Result(
            output="", error="The script failed:\nFileNotFoundError: 'totals.csv'"
        ),
    )
    perry.script(
        call("ask_analysis_agent", question="Chart sales"),
        call("run_python", code="print(1)", inputs={"sales.csv": "SELECT * FROM sales"}),
        last_message,  # Analysis Agent echoes the error
        last_message,
    )

    reply = perry.send("Chart my sales")

    assert "Its input files: sales.csv." in reply


def ask_to_describe(perry, *names: str) -> str:
    """Run a Turn in which the Analysis Agent describes `names`; the reply is the text the
    model received from describe_datasets."""
    perry.script(
        call("ask_analysis_agent", question="Describe my Datasets"),  # Supervisor
        call("describe_datasets", names=list(names)),  # Analysis Agent
        last_message,  # Analysis Agent echoes the tool result
        last_message,  # Supervisor passes it on
    )
    return perry.send("Describe my Datasets")


def test_describing_a_dataset_gives_each_columns_type_and_value_profile(perry, tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(
        "region,amount,note\nNorth,100,\nSouth,250,late\nNorth,50,\nEast,,\n", encoding="utf-8"
    )
    datasets.upload(path)

    described = ask_to_describe(perry, "sales")

    assert described.splitlines() == [
        "sales (4 rows, from sales.csv)",
        "No description yet.",
        "- region (text): No description yet. [no empty values; values 'East', 'North', 'South']",
        "- amount (double precision): No description yet. [25% empty; min 50, max 250]",
        "- note (text): No description yet. [75% empty; values 'late']",
    ]


def test_a_text_column_with_more_than_20_values_is_described_by_three_examples(perry, tmp_path):
    twenty = [f"Team {n:02}" for n in range(1, 21)]
    rows = [f"{team},Person {n:02}" for n, team in enumerate(twenty, 1)] + ["Team 01,Person 21"]
    path = tmp_path / "staff.csv"
    path.write_text("team,person\n" + "\n".join(rows) + "\n", encoding="utf-8")
    datasets.upload(path)

    described = ask_to_describe(perry, "staff")

    listed = ", ".join(f"'{t}'" for t in twenty)
    assert f"- team (text): No description yet. [no empty values; values {listed}]" in described
    assert (
        "- person (text): No description yet. [no empty values; 21 distinct values, "
        "e.g. 'Person 01', 'Person 02', 'Person 03']"
    ) in described


def test_a_short_lists_values_are_shown_in_full_however_long(perry, tmp_path):
    status = "Đang làm việc, trong thời gian thử việc hai tháng theo hợp đồng lao động"
    path = tmp_path / "staff.csv"
    path.write_text(f'name,status\nAn,"{status}"\nBình,Nghỉ việc\n', encoding="utf-8")
    datasets.upload(path)

    described = ask_to_describe(perry, "staff")

    assert len(status) > 60
    assert (
        f"- status (text): No description yet. [no empty values; values 'Nghỉ việc', '{status}']"
    ) in described.splitlines()


def test_text_that_reads_as_iso_dates_is_described_with_its_date_range(perry, tmp_path):
    path = tmp_path / "hires.csv"
    path.write_text(
        "name,hired,pay_period\n"
        "An,2025-11-03,2026-01\nBình,2024-02-29,2026-02\nChi,2026-01-15,2025-12\nDũng,,2026-01\n",
        encoding="utf-8",
    )
    datasets.upload(path)

    described = ask_to_describe(perry, "hires")

    assert "- name (text): No description yet. [no empty values; values 'An', 'Bình'," in (
        described
    )
    assert (
        "- hired (text): No description yet. [25% empty; min 2024-02-29, max 2026-01-15; "
        "values '2024-02-29', '2025-11-03', '2026-01-15']"
    ) in described
    assert (
        "- pay_period (text): No description yet. [no empty values; min 2025-12, max 2026-02;"
        in (described)
    )


def test_describing_an_unknown_dataset_suggests_the_closest_names(perry, tmp_path):
    add_sales_dataset(tmp_path)
    (tmp_path / "salaries.csv").write_text("name,amount\nAn,10\n", encoding="utf-8")
    datasets.upload(tmp_path / "salaries.csv")

    described = ask_to_describe(perry, "sale", "Sales")

    assert described.startswith("No Dataset is called 'sale'. Did you mean sales or salaries?\n")
    assert "sales (3 rows, from sales.csv)" in described  # names as typed in the file work


def test_describing_a_dataset_that_matches_nothing_lists_the_datasets(perry, tmp_path):
    add_sales_dataset(tmp_path)

    described = ask_to_describe(perry, "employees")

    assert described == "No Dataset is called 'employees'. The Datasets are: sales."


def test_uploading_a_dataset_again_replaces_its_profile(perry, tmp_path):
    add_sales_dataset(tmp_path)
    path = tmp_path / "sales.csv"
    path.write_text("region,amount\nWest,7\n", encoding="utf-8")
    datasets.upload(path)

    described = ask_to_describe(perry, "sales")

    assert described.splitlines() == [
        "sales (1 rows, from sales.csv)",
        "No description yet.",
        "- region (text): No description yet. [no empty values; values 'West']",
        "- amount (bigint): No description yet. [no empty values; min 7, max 7]",
    ]


def test_the_analysis_agent_is_asked_to_describe_datasets_before_writing_sql(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="Total sales?"),
        "Done.",  # Analysis Agent
        last_message,
    )

    perry.send("Total sales?")

    agent_prompt = perry.model.calls[1][0].text
    assert "call describe_datasets for the Datasets you will query" in agent_prompt


def add_described_sales_dataset(tmp_path) -> None:
    add_sales_dataset(tmp_path)
    dictionary = tmp_path / "sales.dictionary.csv"
    dictionary.write_text(
        "column,description\n,One row per sale.\namount,Sale amount in VND\n", encoding="utf-8"
    )
    datasets.upload(dictionary)


def test_the_analysis_agent_sees_each_datasets_description_in_the_list(perry, tmp_path):
    add_described_sales_dataset(tmp_path)

    reply = ask_to_list(perry)

    assert reply == (
        "sales (3 rows, from sales.csv): One row per sale. Columns: region (text), amount (bigint)"
    )


def test_describing_a_dataset_gives_the_owners_descriptions_of_it_and_its_columns(perry, tmp_path):
    add_described_sales_dataset(tmp_path)

    described = ask_to_describe(perry, "sales")

    assert described.splitlines() == [
        "sales (3 rows, from sales.csv)",
        "One row per sale.",
        "- region (text): No description yet. [no empty values; values 'North', 'South']",
        "- amount (bigint): Sale amount in VND. [no empty values; min 50, max 250]",
    ]


def upload_csv(tmp_path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    datasets.upload(path)


EMPLOYEES = "employee_id,name\n1,An\n2,Bình\n3,Chi\n"
PAYROLL = "payroll_id,employee_id,amount\n10,1,500\n11,3,700\n12,,0\n"


def ask_to_list(perry) -> str:
    """Run a Turn in which the Analysis Agent lists the Datasets; the reply is the text the
    model received from list_datasets."""
    perry.script(
        call("ask_analysis_agent", question="Which Datasets do I have?"),
        call("list_datasets"),
        last_message,
        last_message,
    )
    return perry.send("Which Datasets do I have?")


def test_an_id_column_whose_values_all_appear_in_another_datasets_key_gets_a_join_hint(
    perry, tmp_path
):
    upload_csv(tmp_path, "employees.csv", EMPLOYEES)
    upload_csv(tmp_path, "payroll.csv", PAYROLL)  # employee_id has a blank: double precision

    listed = ask_to_list(perry)
    described = ask_to_describe(perry, "payroll")

    assert listed.splitlines() == [
        (
            "employees (3 rows, from employees.csv): No description yet. "
            "Columns: employee_id (bigint), name (text)"
        ),
        (
            "payroll (3 rows, from payroll.csv): No description yet. Columns: payroll_id (bigint), "
            "employee_id (double precision), amount (bigint); "
            "join hints: employee_id → employees.employee_id"
        ),
    ]
    assert described.splitlines()[2:4] == [
        "- payroll_id (bigint): No description yet. [no empty values; min 10, max 12]",
        (
            "- employee_id (double precision): No description yet. [33% empty; min 1, max 3; "
            "references employees.employee_id (inferred)]"
        ),
    ]


def test_a_dataset_uploaded_earlier_gets_a_join_hint_to_a_new_dataset(perry, tmp_path):
    upload_csv(tmp_path, "payroll.csv", PAYROLL)
    upload_csv(tmp_path, "employees.csv", EMPLOYEES)

    described = ask_to_describe(perry, "payroll")

    assert (
        "- employee_id (double precision): No description yet. [33% empty; min 1, max 3; "
        "references employees.employee_id (inferred)]"
    ) in described.splitlines()


def test_a_join_hint_disappears_when_a_reupload_leaves_a_value_unmatched(perry, tmp_path):
    upload_csv(tmp_path, "employees.csv", EMPLOYEES)
    upload_csv(tmp_path, "payroll.csv", PAYROLL)
    upload_csv(tmp_path, "employees.csv", "employee_id,name\n1,An\n2,Bình\n")  # 3 has left

    listed = ask_to_list(perry)
    described = ask_to_describe(perry, "payroll")

    assert "join hints" not in listed
    assert (
        "- employee_id (double precision): No description yet. [33% empty; min 1, max 3]"
    ) in described.splitlines()

    upload_csv(tmp_path, "employees.csv", EMPLOYEES)  # 3 is back: the hint returns
    assert "join hints: employee_id → employees.employee_id" in ask_to_list(perry)
    upload_csv(tmp_path, "payroll.csv", PAYROLL + "13,4,900\n")  # but payroll now pays 4

    assert "join hints" not in ask_to_list(perry)


def test_a_described_column_keeps_its_description_beside_its_join_hint(perry, tmp_path):
    upload_csv(tmp_path, "employees.csv", EMPLOYEES)
    upload_csv(tmp_path, "payroll.csv", PAYROLL)
    upload_csv(
        tmp_path,
        "payroll.dictionary.csv",
        "column,description\n,One row per payment.\nemployee_id,Who was paid\n",
    )

    listed = ask_to_list(perry)
    described = ask_to_describe(perry, "payroll")

    assert listed.splitlines()[1] == (
        "payroll (3 rows, from payroll.csv): One row per payment. Columns: payroll_id (bigint), "
        "employee_id (double precision), amount (bigint); "
        "join hints: employee_id → employees.employee_id"
    )
    assert (
        "- employee_id (double precision): Who was paid. [33% empty; min 1, max 3; "
        "references employees.employee_id (inferred)]"
    ) in described.splitlines()


STAFF = "employee_id,name,manager_id\n1,An,\n2,Bình,1\n3,Chi,1\n"


def test_a_declared_self_reference_reaches_the_agent_as_declared(perry, tmp_path):
    upload_csv(tmp_path, "staff.csv", STAFF)
    upload_csv(
        tmp_path,
        "staff.dictionary.csv",
        "column,description,references\nmanager_id,Who they report to,staff.employee_id\n",
    )

    listed = ask_to_list(perry)
    described = ask_to_describe(perry, "staff")

    assert listed.endswith("; join hints: manager_id → staff.employee_id")
    assert (
        "- manager_id (double precision): Who they report to. [33% empty; min 1, max 1; "
        "references staff.employee_id (declared)]"
    ) in described.splitlines()


def test_a_declared_reference_wins_over_the_inferred_one_for_the_same_column(perry, tmp_path):
    upload_csv(tmp_path, "employees.csv", EMPLOYEES)
    upload_csv(tmp_path, "people.csv", "employee_id,name\n1,An\n3,Chi\n4,Dũng\n")
    upload_csv(tmp_path, "payroll.csv", PAYROLL)  # infers employees.employee_id
    upload_csv(
        tmp_path,
        "payroll.dictionary.csv",
        "column,description,references\nemployee_id,Who was paid,people.employee_id\n",
    )
    upload_csv(tmp_path, "employees.csv", EMPLOYEES)  # checks inferred hints again

    listed = ask_to_list(perry)
    described = ask_to_describe(perry, "payroll")

    assert listed.splitlines()[1].endswith("; join hints: employee_id → people.employee_id")
    assert (
        "- employee_id (double precision): Who was paid. [33% empty; min 1, max 3; "
        "references people.employee_id (declared)]"
    ) in described.splitlines()


def test_a_declared_reference_is_checked_again_when_either_dataset_is_uploaded(perry, tmp_path):
    upload_csv(tmp_path, "staff.csv", STAFF)
    upload_csv(tmp_path, "reviews.csv", "review_id,reviewer_id\n7,1\n8,2\n")
    upload_csv(
        tmp_path,
        "reviews.dictionary.csv",
        "column,description,references\nreviewer_id,Who wrote it,staff.employee_id\n",
    )

    upload_csv(tmp_path, "reviews.csv", "review_id,reviewer_id\n7,1\n8,2\n9,5\n")
    after_reviews = ask_to_describe(perry, "reviews")
    upload_csv(tmp_path, "staff.csv", "employee_id,name,manager_id\n1,An,\n")
    after_staff = ask_to_describe(perry, "reviews")

    assert (
        "references staff.employee_id (declared, 1 value not found there)"
        in (after_reviews.splitlines()[3])  # after the line saying reviews is undescribed
    )
    assert (
        "references staff.employee_id (declared, 2 values not found there)"
        in (after_staff.splitlines()[3])
    )
