"""`python -m scripts.company_data`: the synthetic company sample, whose tables must join up
and whose payroll must follow from the other tables and the tax and insurance rules."""

from datetime import date

import pandas as pd
import pytest

from perry import datasets
from scripts import company_data


@pytest.fixture(scope="module", params=[100, 1000], ids=["100 employees", "1000 employees"])
def tables(request, tmp_path_factory) -> dict[str, pd.DataFrame]:
    """The tables as their CSVs read back, which is what an upload sees."""
    folder = tmp_path_factory.mktemp("company")
    paths = company_data.write(company_data.generate(employees=request.param), folder)
    return {path.stem: pd.read_csv(path) for path in paths if "." not in path.stem}


def ids_in(tables, table: str, column: str, target: str, key: str) -> bool:
    referenced = tables[table][column].dropna()
    return bool(referenced.isin(tables[target][key]).all())


def test_every_reference_points_at_a_row_that_exists(tables):
    references = [
        ("departments", "parent_department_id", "departments", "department_id"),
        ("positions", "department_id", "departments", "department_id"),
        ("positions", "grade_id", "job_grades", "grade_id"),
        ("employees", "position_id", "positions", "position_id"),
        ("employees", "manager_id", "employees", "employee_id"),
        ("salary_history", "position_id", "positions", "position_id"),
        ("performance_reviews", "reviewer_id", "employees", "employee_id"),
        ("payroll", "position_id", "positions", "position_id"),
    ]
    references += [
        (table, "employee_id", "employees", "employee_id")
        for table in (
            "education",
            "salary_history",
            "performance_reviews",
            "bonuses",
            "penalties",
            "payroll",
        )
    ]

    broken = [ref for ref in references if not ids_in(tables, *ref)]

    assert broken == []
    for table, frame in tables.items():
        key = frame.columns[0]
        if key.endswith("_id"):
            assert frame[key].is_unique, table


def test_only_the_chief_executive_has_no_manager(tables):
    employees = tables["employees"]
    assert employees[employees.manager_id.isna()].employee_id.tolist() == [1]


def test_salaries_stay_within_their_grade_band_outside_probation(tables):
    salaries = tables["salary_history"].merge(tables["positions"], on="position_id")
    salaries = salaries.merge(tables["job_grades"], on="grade_id")
    full = salaries[salaries.reason != "Tuyển dụng - thử việc"]

    outside = full[(full.base_salary < full.min_salary) | (full.base_salary > full.max_salary)]

    assert outside.empty


def test_headcount_stays_near_the_size_asked_for(tables):
    size = len(tables["employees"][tables["employees"].hire_date < "2026-01-01"])
    monthly = tables["payroll"].groupby("pay_period").employee_id.nunique()

    assert len(monthly) == 12
    assert monthly.between(size * 0.97, size * 1.03).all()


def test_tax_on_a_30_million_salary_under_the_2026_rules():
    # Gross 30 million, 3 million insurance, no dependents: 11.5 million taxable.
    taxable = 30_000_000 - 3_000_000 - 15_500_000

    assert company_data.income_tax(taxable, date(2026, 3, 1)) == 650_000  # 500k + 150k
    assert company_data.income_tax(taxable, date(2025, 3, 1)) == 975_000  # 7 brackets
    assert company_data.income_tax(0, date(2026, 3, 1)) == 0


def test_each_payslip_adds_up(tables):
    pay = tables["payroll"]
    insurance = pay.social_insurance + pay.health_insurance + pay.unemployment_insurance

    assert (
        pay.gross_income == pay.salary_pay + pay.allowances + pay.overtime_pay + pay.bonus_total
    ).all()
    expected_taxable = pay.gross_income - insurance - pay.personal_deduction
    expected_taxable = (expected_taxable - pay.dependent_deduction).clip(lower=0)
    assert (pay.taxable_income == expected_taxable).all()
    deducted = insurance + pay.personal_income_tax + pay.penalty_total
    assert (pay.net_pay == pay.gross_income - deducted).all()
    assert (pay.total_employer_cost == pay.gross_income + pay.employer_insurance).all()


def test_bonuses_and_deductions_reach_the_payslip_of_their_month(tables):
    pay = tables["payroll"].set_index(["employee_id", "pay_period"])
    bonuses = tables["bonuses"].groupby(["employee_id", "pay_period"]).amount.sum()
    deductions = tables["penalties"].groupby(["employee_id", "pay_period"]).deduction_amount.sum()

    assert (pay.loc[bonuses.index, "bonus_total"] == bonuses).all()
    assert (pay.loc[deductions.index, "penalty_total"] == deductions).all()
    assert pay.bonus_total.sum() == bonuses.sum()


def test_insurance_never_goes_over_its_cap(tables):
    pay = tables["payroll"]
    first_half = pay.pay_period <= "2026-06"
    social_cap = first_half.map({True: 46_800_000, False: 50_600_000})

    assert (pay.social_insurance <= (social_cap * 0.08).round()).all()
    assert (pay.health_insurance <= (social_cap * 0.015).round()).all()
    assert (pay.unemployment_insurance <= round(106_200_000 * 0.01)).all()
    assert (pay[pay.worked_days < 14].insurance_salary == 0).all()


def test_only_employees_on_the_books_are_paid(tables):
    pay = tables["payroll"].merge(tables["employees"], on="employee_id")
    month_start = pay.pay_period + "-01"

    assert (pay.hire_date <= month_start.str[:7] + "-31").all()
    left = pay[pay.termination_date.notna()]
    assert (left.termination_date >= left.pay_period + "-01").all()


def test_the_same_seed_writes_the_same_files(tmp_path):
    first = company_data.write(company_data.generate(seed=7), tmp_path / "a")
    second = company_data.write(company_data.generate(seed=7), tmp_path / "b")
    other = company_data.write(company_data.generate(seed=8), tmp_path / "c")

    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]
    assert [p.read_bytes() for p in first] != [p.read_bytes() for p in other]


def test_the_committed_sample_is_what_the_defaults_generate(tmp_path):
    fresh = company_data.write(company_data.generate(), tmp_path)

    for path in fresh:
        committed = company_data.OUT_DIR / path.name
        assert committed.read_bytes() == path.read_bytes(), path.name


def test_the_sample_uploads_with_a_data_dictionary_for_every_dataset(database, tmp_path):
    paths = company_data.write(company_data.generate(), tmp_path)

    replies = [result.describe() for path in paths for result in datasets.upload(path)]

    described = {d.name: d for d in datasets.list_datasets()}
    dictionaries = [p for p in paths if p.name.endswith(".dictionary.csv")]
    assert sorted(p.name.split(".")[0] for p in dictionaries) == sorted(described)
    for name, dataset in described.items():
        assert dataset.description, f"{name} has no description of what one row is"
    for problem in ("not attached", "refused", "not found", "waiting", "dropped"):
        assert [r for r in replies if problem in r] == [], problem
    declared = [
        ("employees", "employees.employee_id"),  # manager_id
        ("performance_reviews", "employees.employee_id"),  # reviewer_id
        ("departments", "departments.department_id"),  # parent_department_id
    ]
    for name, target in declared:
        assert f"references {target} (declared)" in described[name].describe_columns(), name


def test_uploaded_tables_join_up_for_the_analysis_agent(database, tmp_path):
    for path in company_data.write(company_data.generate(), tmp_path):
        datasets.upload(path)

    result = datasets.run_query(
        """
        SELECT d.department_name, sum(p.total_employer_cost) AS cost
        FROM payroll p
        JOIN positions pos ON pos.position_id = p.position_id
        JOIN departments d ON d.department_id = pos.department_id
        WHERE p.pay_period BETWEEN '2026-04' AND '2026-06'
        GROUP BY d.department_name
        ORDER BY cost DESC
        """
    )

    assert result.columns == ["department_name", "cost"]
    assert result.rows[0][0] == "Phòng Phát triển phần mềm"
    assert len(result.rows) == 9
