"""Generate the company sample: a fictional Vietnamese software company, Công ty Cổ phần
Công nghệ Perry, with about 100 employees and a year of payroll, as one CSV per table in
sample_data/company/, each with its Data Dictionary (its README.md describes the tables).

    uv run python -m scripts.company_data                         # write the files
    uv run python -m scripts.company_data --employees 1000 --years 2
    uv run python -m scripts.company_data --load                  # and upload them as Datasets

The same arguments always write the same files. Payroll is computed from the other tables
(salary history, bonuses, penalties, and the tax and insurance parameters), so its numbers
agree with them. The rules are Vietnam's (VND, progressive personal income tax, social,
health and unemployment insurance), simplified where the README says so.
"""

import argparse
import calendar
import random
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "company"
EMAIL_DOMAIN = "perrytech.example"
FOUNDED = date(2016, 3, 1)

# --- Reference tables -------------------------------------------------------------------

# department_id, department_name, parent_department_id, cost_center
DEPARTMENTS = [
    (1, "Ban Giám đốc", None, "CC-100"),
    (2, "Khối Kỹ thuật", 1, "CC-200"),
    (3, "Phòng Phát triển phần mềm", 2, "CC-210"),
    (4, "Phòng Hạ tầng và Vận hành", 2, "CC-220"),
    (5, "Phòng Kinh doanh", 1, "CC-300"),
    (6, "Phòng Marketing", 1, "CC-400"),
    (7, "Phòng Chăm sóc khách hàng", 1, "CC-500"),
    (8, "Phòng Nhân sự", 1, "CC-600"),
    (9, "Phòng Tài chính - Kế toán", 1, "CC-700"),
]
TECH_DEPARTMENTS = {3, 4}
SALES_DEPARTMENT = 5

# grade_id, grade_code, grade_name, min_salary, max_salary (monthly base, VND)
GRADES = [
    (1, "G1", "Nhân viên mới (Fresher)", 9_000_000, 14_000_000),
    (2, "G2", "Nhân viên (Junior)", 13_000_000, 20_000_000),
    (3, "G3", "Nhân viên chính (Middle)", 19_000_000, 30_000_000),
    (4, "G4", "Chuyên viên cao cấp (Senior)", 28_000_000, 45_000_000),
    (5, "G5", "Trưởng nhóm", 40_000_000, 60_000_000),
    (6, "G6", "Trưởng phòng", 55_000_000, 85_000_000),
    (7, "G7", "Giám đốc", 90_000_000, 150_000_000),
]
BAND = {g[0]: (g[3], g[4]) for g in GRADES}
STAFF_GRADES = {1: "Fresher", 2: "Junior", 3: "Middle", 4: "Senior"}
STAFF_GRADE_WEIGHTS = [15, 35, 35, 15]
POSITION_ALLOWANCE = {5: 3_000_000, 6: 5_000_000, 7: 10_000_000}

# The leaders: key, department, title, grade, manager's key. They stay all period, so the
# reporting lines hold still while staff come and go.
LEADERS = [
    ("ceo", 1, "Tổng Giám đốc", 7, None),
    ("cto", 2, "Giám đốc Kỹ thuật", 7, "ceo"),
    ("assistant", 1, "Trợ lý Tổng Giám đốc", 3, "ceo"),
    (3, 3, "Trưởng phòng Phát triển phần mềm", 6, "cto"),
    (4, 4, "Trưởng phòng Hạ tầng và Vận hành", 6, "cto"),
    (5, 5, "Trưởng phòng Kinh doanh", 6, "ceo"),
    (6, 6, "Trưởng phòng Marketing", 6, "ceo"),
    (7, 7, "Trưởng phòng Chăm sóc khách hàng", 6, "ceo"),
    (8, 8, "Trưởng phòng Nhân sự", 6, "ceo"),
    (9, 9, "Kế toán trưởng", 6, "ceo"),
]
TEAM_LEAD_TITLES = {
    3: "Trưởng nhóm phát triển",
    4: "Trưởng nhóm vận hành",
    5: "Trưởng nhóm kinh doanh",
    6: "Trưởng nhóm marketing",
    7: "Trưởng nhóm chăm sóc khách hàng",
    8: "Trưởng nhóm nhân sự",
    9: "Trưởng nhóm kế toán",
}
STAFF_PER_TEAM_LEAD = 10

# department, role, share of the staff, where in its grade's band the pay sits (+ higher)
STAFF_ROLES = [
    (3, "Lập trình viên", 24, 0.15),
    (3, "Kiểm thử viên", 6, 0.0),
    (4, "Kỹ sư hệ thống", 7, 0.1),
    (5, "Chuyên viên kinh doanh", 16, 0.0),
    (6, "Chuyên viên marketing", 7, -0.05),
    (7, "Nhân viên chăm sóc khách hàng", 11, -0.2),
    (8, "Chuyên viên nhân sự", 4, -0.1),
    (9, "Kế toán viên", 5, -0.05),
]

# Tax and insurance rules, each in force from its effective_from until a later row of the
# same kind replaces it: a new rule is a new row, not a code change.
# bracket_no, lower_bound, upper_bound (None: no limit), rate, effective_from
TAX_BRACKETS = [
    (1, 0, 5_000_000, 0.05, "2009-01-01"),
    (2, 5_000_000, 10_000_000, 0.10, "2009-01-01"),
    (3, 10_000_000, 18_000_000, 0.15, "2009-01-01"),
    (4, 18_000_000, 32_000_000, 0.20, "2009-01-01"),
    (5, 32_000_000, 52_000_000, 0.25, "2009-01-01"),
    (6, 52_000_000, 80_000_000, 0.30, "2009-01-01"),
    (7, 80_000_000, None, 0.35, "2009-01-01"),
    # Luật Thuế TNCN 109/2025/QH15: five brackets for salaries from the 2026 tax period.
    (1, 0, 10_000_000, 0.05, "2026-01-01"),
    (2, 10_000_000, 30_000_000, 0.10, "2026-01-01"),
    (3, 30_000_000, 60_000_000, 0.20, "2026-01-01"),
    (4, 60_000_000, 100_000_000, 0.30, "2026-01-01"),
    (5, 100_000_000, None, 0.35, "2026-01-01"),
]
# insurance_type, employee_rate, employer_rate, salary_cap, effective_from, effective_to.
# BHXH and BHYT are capped at 20 × the base reference salary, BHTN at 20 × the region I
# minimum wage.
INSURANCE_RATES = [
    ("BHXH", 0.08, 0.175, 46_800_000, "2024-07-01", "2026-06-30"),
    ("BHXH", 0.08, 0.175, 50_600_000, "2026-07-01", None),
    ("BHYT", 0.015, 0.03, 46_800_000, "2024-07-01", "2026-06-30"),
    ("BHYT", 0.015, 0.03, 50_600_000, "2026-07-01", None),
    ("BHTN", 0.01, 0.01, 99_200_000, "2024-07-01", "2025-12-31"),
    ("BHTN", 0.01, 0.01, 106_200_000, "2026-01-01", None),
]
# code, value, unit, effective_from, note
PAYROLL_PARAMETERS = [
    ("personal_deduction", 11_000_000, "VND/tháng", "2020-07-01", "NQ 954/2020/UBTVQH14"),
    ("personal_deduction", 15_500_000, "VND/tháng", "2026-01-01", "NQ 110/2025/UBTVQH15"),
    ("dependent_deduction", 4_400_000, "VND/tháng", "2020-07-01", "NQ 954/2020/UBTVQH14"),
    ("dependent_deduction", 6_200_000, "VND/tháng", "2026-01-01", "NQ 110/2025/UBTVQH15"),
    ("base_reference_salary", 2_340_000, "VND/tháng", "2024-07-01", "Lương cơ sở"),
    ("base_reference_salary", 2_530_000, "VND/tháng", "2026-07-01", "NĐ 161/2026/NĐ-CP"),
    ("regional_minimum_wage_1", 4_960_000, "VND/tháng", "2024-07-01", "Lương tối thiểu vùng I"),
    ("regional_minimum_wage_1", 5_310_000, "VND/tháng", "2026-01-01", "NĐ 293/2025/NĐ-CP"),
    ("meal_allowance", 730_000, "VND/tháng", "2016-03-01", "Chính sách công ty"),
    ("overtime_rate", 1.5, "hệ số", "2016-03-01", "Làm thêm ngày thường"),
    ("probation_salary_rate", 0.85, "hệ số", "2016-03-01", "Lương thử việc"),
    ("probation_days", 60, "ngày", "2016-03-01", "Thử việc trình độ cao đẳng trở lên"),
    ("insurance_min_worked_days", 14, "ngày", "2016-03-01", "Dưới mức này không đóng BH"),
]

# Tết (first day of the lunar year) and the Hùng Kings' day, for the public holidays.
TET = {
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
    2027: date(2027, 2, 6),
    2028: date(2028, 1, 26),
}
HUNG_KINGS = {
    2025: date(2025, 4, 7),
    2026: date(2026, 4, 26),
    2027: date(2027, 4, 16),
    2028: date(2028, 4, 4),
}

# --- People -----------------------------------------------------------------------------

SURNAMES = [
    ("Nguyễn", 31), ("Trần", 11), ("Lê", 9), ("Phạm", 7), ("Hoàng", 5), ("Huỳnh", 4),
    ("Phan", 4), ("Vũ", 4), ("Võ", 3), ("Đặng", 2), ("Bùi", 2), ("Đỗ", 2), ("Hồ", 2),
    ("Ngô", 2), ("Dương", 1), ("Lý", 1), ("Đinh", 1), ("Trịnh", 1), ("Mai", 1), ("Tạ", 1),
]  # fmt: skip
MIDDLE_NAMES = {
    "Nam": ["Văn", "Đức", "Minh", "Quang", "Hữu", "Thanh", "Tuấn", "Hoàng", "Ngọc", "Trọng"],
    "Nữ": ["Thị", "Ngọc", "Thu", "Thanh", "Minh", "Phương", "Hồng", "Mai", "Thùy", "Bảo"],
}
GIVEN_NAMES = {
    "Nam": [
        "An", "Bình", "Cường", "Dũng", "Duy", "Đạt", "Hải", "Hiếu", "Hoàng", "Hùng", "Huy",
        "Khang", "Khánh", "Kiên", "Long", "Lâm", "Mạnh", "Minh", "Nam", "Nghĩa", "Phong",
        "Phúc", "Quân", "Sơn", "Tài", "Thắng", "Thành", "Toàn", "Trung", "Tú", "Tuấn", "Việt",
    ],
    "Nữ": [
        "Anh", "Chi", "Dung", "Giang", "Hà", "Hạnh", "Hằng", "Hiền", "Hoa", "Hương", "Lan",
        "Linh", "Ly", "Mai", "My", "Nga", "Ngân", "Ngọc", "Nhung", "Oanh", "Phương", "Quyên",
        "Thảo", "Thủy", "Trang", "Uyên", "Vân", "Vy", "Yến", "Yên",
    ],
}  # fmt: skip
# The share of women in each department.
WOMEN_SHARE = {1: 0.4, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.45, 6: 0.6, 7: 0.7, 8: 0.75, 9: 0.75}
PROVINCES = [
    "Hà Nội", "TP. Hồ Chí Minh", "Hải Phòng", "Đà Nẵng", "Cần Thơ", "Huế", "Nghệ An",
    "Thanh Hóa", "Ninh Bình", "Hưng Yên", "Bắc Ninh", "Quảng Ninh", "Phú Thọ", "Thái Nguyên",
    "Lào Cai", "Hà Tĩnh", "Quảng Trị", "Quảng Ngãi", "Gia Lai", "Đắk Lắk", "Khánh Hòa",
    "Lâm Đồng", "Đồng Nai", "Tây Ninh", "An Giang", "Đồng Tháp", "Vĩnh Long", "Cà Mau",
]  # fmt: skip
NORTH = {
    "Hà Nội", "Hải Phòng", "Nghệ An", "Thanh Hóa", "Ninh Bình", "Hưng Yên", "Bắc Ninh",
    "Quảng Ninh", "Phú Thọ", "Thái Nguyên", "Lào Cai", "Hà Tĩnh",
}  # fmt: skip
# Age range when holding each grade.
AGES = {1: (22, 25), 2: (23, 29), 3: (26, 34), 4: (29, 40), 5: (31, 42), 6: (34, 48), 7: (40, 55)}

MAJORS = {
    1: ["Quản trị kinh doanh", "Kinh tế đối ngoại"],
    2: ["Công nghệ thông tin", "Khoa học máy tính"],
    3: ["Công nghệ thông tin", "Khoa học máy tính", "Kỹ thuật phần mềm", "Hệ thống thông tin"],
    4: ["Mạng máy tính và truyền thông dữ liệu", "Công nghệ thông tin", "Kỹ thuật điện tử"],
    5: ["Quản trị kinh doanh", "Kinh doanh quốc tế", "Marketing", "Kinh tế"],
    6: ["Marketing", "Truyền thông đa phương tiện", "Quan hệ công chúng"],
    7: ["Quản trị kinh doanh", "Ngôn ngữ Anh", "Quản trị dịch vụ du lịch và lữ hành"],
    8: ["Quản trị nhân lực", "Luật", "Tâm lý học"],
    9: ["Kế toán", "Tài chính - Ngân hàng", "Kiểm toán"],
}
TECH_SCHOOLS = {
    "north": [
        "Đại học Bách khoa Hà Nội", "Trường Đại học Công nghệ - ĐHQG Hà Nội",
        "Học viện Công nghệ Bưu chính Viễn thông", "Trường Đại học FPT",
    ],
    "south": [
        "Trường Đại học Bách khoa - ĐHQG TP.HCM",
        "Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM",
        "Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM", "Trường Đại học FPT",
    ],
}  # fmt: skip
BUSINESS_SCHOOLS = {
    "north": [
        "Đại học Kinh tế Quốc dân", "Trường Đại học Ngoại thương", "Học viện Tài chính",
        "Trường Đại học Thương mại", "Trường Đại học Luật Hà Nội",
    ],
    "south": [
        "Đại học Kinh tế TP.HCM", "Trường Đại học Ngân hàng TP.HCM",
        "Trường Đại học Kinh tế - Luật - ĐHQG TP.HCM", "Trường Đại học Luật TP.HCM",
    ],
}  # fmt: skip
COLLEGES = ["Trường Cao đẳng FPT Polytechnic", "Trường Cao đẳng Kinh tế Đối ngoại"]
VOCATIONAL = ["Trường Trung cấp Kinh tế - Kỹ thuật", "Trường Trung cấp Du lịch và Khách sạn"]
# department -> (certificate, issuer, chance)
CERTIFICATES = {
    3: [
        ("AWS Certified Solutions Architect - Associate", "Amazon Web Services", 0.15),
        ("ISTQB Certified Tester Foundation Level", "ISTQB", 0.1),
    ],
    4: [
        ("CCNA", "Cisco", 0.35),
        ("AWS Certified SysOps Administrator - Associate", "Amazon Web Services", 0.25),
    ],
    9: [("Chứng chỉ Kế toán trưởng", "Bộ Tài chính", 0.3), ("ACCA", "ACCA", 0.1)],
    8: [("Chứng chỉ Quản trị nhân sự chuyên nghiệp", "PACE", 0.25)],
}
ENGLISH = [
    ("IELTS 6.5", "IDP Việt Nam"),
    ("IELTS 7.0", "British Council"),
    ("TOEIC 750", "IIG Việt Nam"),
]

PROJECTS = [
    "Ứng dụng di động cho khách hàng",
    "Hệ thống CRM nội bộ",
    "Chuyển hạ tầng lên cloud",
    "Cổng thanh toán trực tuyến",
    "Nâng cấp hệ thống giám sát",
    "Tích hợp hóa đơn điện tử",
    "Trợ lý ảo chăm sóc khách hàng",
]
REPRIMANDS = [
    "Đi muộn nhiều lần trong tháng",
    "Vi phạm quy định bảo mật thông tin",
    "Nghỉ việc không báo trước",
    "Không tuân thủ quy trình phê duyệt",
]
DAMAGES = [
    ("Làm hỏng laptop công ty", 3_000_000, 8_000_000),
    ("Làm mất thẻ ra vào và thiết bị", 500_000, 2_000_000),
    ("Làm hỏng màn hình làm việc", 1_500_000, 4_000_000),
]
REPRIMAND = "Khiển trách"
DELAYED_RAISE = "Kéo dài thời hạn nâng lương"
DAMAGE = "Bồi thường thiệt hại tài sản"

ACTIVE, LEFT = "Đang làm việc", "Đã nghỉ việc"
PERMANENT, PROBATION = "Chính thức", "Thử việc"

# --- Data Dictionary ----------------------------------------------------------------------

# Per table: what one row is, then the columns whose meaning isn't obvious, each with the
# join hint Perry can't infer when there is one (a self-reference or a renamed key). Perry
# infers the rest: an `x_id` column whose values all appear in the Dataset whose first
# column is `x_id`. Money is whole VND, rates are fractions (0.08 is 8%).
MONEY = "VND"
DICTIONARY: dict[str, tuple[str, dict[str, str | tuple[str, str]]]] = {
    "departments": (
        "One department or division of the company; departments nest in a tree.",
        {
            "parent_department_id": (
                (
                    "The department this one sits under; empty for Ban Giám đốc, the top. "
                    "Khối Kỹ thuật's children are the engineering departments."
                ),
                "departments.department_id",
            ),
            "cost_center": "The accounting cost centre the department's costs are booked to.",
        },
    ),
    "job_grades": (
        "One job grade, G1 (Fresher) to G7 (Giám đốc), with its monthly base salary band.",
        {
            "min_salary": f"Lowest monthly base salary of the grade, {MONEY}.",
            "max_salary": f"Highest monthly base salary of the grade, {MONEY}.",
        },
    ),
    "positions": (
        "One job title in one department at one job grade.",
        {"position_title": "The title, with the level in brackets for staff roles."},
    ),
    "tax_brackets": (
        (
            "One bracket of the progressive personal income tax on monthly taxable income. "
            "A set of brackets applies from its effective_from until a later set starts."
        ),
        {
            "bracket_no": "The bracket's order within its set, from 1 (lowest).",
            "lower_bound": f"Monthly taxable income where the bracket starts, {MONEY}.",
            "upper_bound": f"Where the bracket ends, {MONEY}; empty for the top bracket.",
            "rate": "Tax rate on the income within the bracket, as a fraction.",
            "effective_from": "First day the set of brackets applies (ISO date text).",
        },
    ),
    "insurance_rates": (
        (
            "The rates of one compulsory insurance (BHXH social, BHYT health, BHTN "
            "unemployment) over one period."
        ),
        {
            "employee_rate": "Share of the insurance salary deducted from the employee.",
            "employer_rate": "Share of the insurance salary the company pays on top.",
            "salary_cap": f"Monthly insurance salary above which nothing more is paid, {MONEY}.",
            "effective_from": "First day the rates apply (ISO date text).",
            "effective_to": "Last day the rates apply; empty while still in force.",
        },
    ),
    "payroll_parameters": (
        (
            "One value of one payroll parameter from a date on; a later row for the same "
            "code replaces it."
        ),
        {
            "code": "Which parameter: deductions, reference salaries, allowances and rates.",
            "value": "The parameter's value, in its unit.",
            "unit": "VND/tháng is VND per month, hệ số a multiplier, ngày days.",
            "effective_from": "First day the value applies (ISO date text).",
            "note": "The legal source or policy behind the value.",
        },
    ),
    "employees": (
        "One person ever employed, including those who left.",
        {
            "employee_code": "The HR code shown on documents (NV0001).",
            "hometown_province": "The province the person comes from, not where they work.",
            "termination_date": "Last day employed; empty while still employed.",
            "status": f"{ACTIVE} (still employed) or {LEFT} (left the company).",
            "employment_type": f"{PERMANENT} (permanent) or {PROBATION} (still on probation).",
            "position_id": "The latest position held (the one at leaving, for leavers).",
            "manager_id": (
                "The employee's line manager; empty only for the Tổng Giám đốc.",
                "employees.employee_id",
            ),
            "work_location": "The office the person works in.",
            "num_dependents": "Dependents registered for the personal income tax deduction.",
        },
    ),
    "education": (
        "One degree or certificate held by one employee.",
        {
            "record_type": "Bằng cấp (a degree) or Chứng chỉ (a certificate).",
            "degree_level": "Level of the degree (Thạc sĩ is a master's); Chứng chỉ for a "
            "certificate.",
            "major": "The field studied, or the certificate's name.",
            "graduation_year": "Year the degree or certificate was obtained.",
        },
    ),
    "salary_history": (
        "One change of an employee's pay or position, effective from a date until their next row.",
        {
            "effective_date": "First day the pay applies (ISO date text).",
            "position_id": "The position held from that date.",
            "base_salary": f"Monthly base salary, {MONEY}.",
            "position_allowance": f"Monthly allowance of the position, {MONEY}.",
            "meal_allowance": f"Monthly meal allowance, {MONEY}.",
            "reason": "Why the pay changed: hiring, end of probation, the April review, or "
            "a promotion.",
        },
    ),
    "performance_reviews": (
        "One performance review of one employee.",
        {
            "review_period": "The period reviewed: a year (2025) for the January review, "
            "or a half (2026-H1) for the July review.",
            "review_date": "Day the review took place (ISO date text).",
            "rating": "From 1 (poor) to 5 (outstanding).",
            "reviewer_id": (
                "The employee who gave the review; empty for the Tổng Giám đốc's own review.",
                "employees.employee_id",
            ),
        },
    ),
    "bonuses": (
        "One bonus paid to one employee.",
        {
            "pay_period": "Month the bonus is paid in (YYYY-MM); it adds to that month's "
            "payroll bonus_total.",
            "amount": f"Bonus paid, {MONEY}, before tax.",
            "note": "What the bonus is for, such as the quarter or project.",
        },
    ),
    "penalties": (
        "One disciplinary action against one employee.",
        {
            "incident_date": "Day of the incident (ISO date text).",
            "pay_period": "Month any deduction is taken from pay (YYYY-MM).",
            "penalty_type": f"{REPRIMAND} (a reprimand), {DELAYED_RAISE} (cancels an April "
            f"raise due within six months) or {DAMAGE} (compensation for damaged property).",
            "deduction_amount": f"Taken from net pay, {MONEY}; 0 except for {DAMAGE}.",
        },
    ),
    "payroll": (
        "One employee's payslip for one month worked; all amounts in VND.",
        {
            "pay_period": "The month paid (YYYY-MM).",
            "position_id": "The position held that month.",
            "standard_days": "Working days in the month (weekdays less public holidays).",
            "worked_days": "Days employed in the month, less unpaid leave.",
            "base_salary": "Monthly base salary in force that month, before proration.",
            "salary_pay": "base_salary prorated by worked_days / standard_days.",
            "allowances": "Position and meal allowances, prorated by days worked.",
            "overtime_pay": "Pay for overtime_hours at 1.5 times the hourly rate.",
            "bonus_total": "Sum of the employee's bonuses paid that month.",
            "gross_income": "salary_pay + allowances + overtime_pay + bonus_total.",
            "insurance_salary": "Salary insurance is computed on, before each insurance's cap; "
            "0 when worked_days is under 14.",
            "social_insurance": "Employee's BHXH (social insurance) deduction.",
            "health_insurance": "Employee's BHYT (health insurance) deduction.",
            "unemployment_insurance": "Employee's BHTN (unemployment insurance) deduction.",
            "personal_deduction": "Personal income tax deduction for the employee.",
            "dependent_deduction": "Tax deduction for the employee's dependents.",
            "taxable_income": "gross_income less the employee's insurance and the deductions, "
            "never below 0.",
            "personal_income_tax": "Personal income tax withheld that month.",
            "penalty_total": "Deductions for damaged property taken that month.",
            "net_pay": "Paid to the employee: gross_income less the employee's insurance, "
            "personal_income_tax and penalty_total.",
            "employer_insurance": "Insurance the company pays on top of gross_income.",
            "total_employer_cost": "What the employee costs the company that month: "
            "gross_income + employer_insurance.",
        },
    ),
}


# --- Rules ------------------------------------------------------------------------------


def in_force(rows: list[tuple], on: date, *, kind=None, date_at: int, kind_at: int = 0):
    """The rows in force on `on`: those with the latest effective_from not after it (for
    one `kind`, when given)."""
    rows = [r for r in rows if kind is None or r[kind_at] == kind]
    latest = max(r[date_at] for r in rows if r[date_at] <= on.isoformat())
    return [r for r in rows if r[date_at] == latest]


def parameter(code: str, on: date) -> float:
    return in_force(PAYROLL_PARAMETERS, on, kind=code, date_at=3)[0][1]


def insurance(kind: str, on: date) -> tuple[float, float, int]:
    """(employee rate, employer rate, salary cap) for BHXH, BHYT or BHTN on a date."""
    row = in_force(INSURANCE_RATES, on, kind=kind, date_at=4)[0]
    return row[1], row[2], row[3]


def income_tax(taxable: float, on: date) -> int:
    """Personal income tax on a month's taxable income: each bracket's rate on the slice of
    income inside it."""
    tax = 0.0
    for _, lower, upper, rate, _ in in_force(TAX_BRACKETS, on, date_at=4):
        top = taxable if upper is None else min(taxable, upper)
        if top > lower:
            tax += (top - lower) * rate
    return round(tax)


def holidays(year: int) -> set[date]:
    """Public holidays on working days. One that falls on a weekend moves to the next working
    day. The lunar dates are approximate (see the README)."""
    days = [date(year, 1, 1), date(year, 4, 30), date(year, 5, 1), date(year, 9, 1)]
    days += [date(year, 9, 2), HUNG_KINGS[year]]
    days += [TET[year] + timedelta(days=d) for d in range(-1, 4)]
    off: set[date] = set()
    for day in sorted(days):
        while day.weekday() >= 5 or day in off:
            day += timedelta(days=1)
        off.add(day)
    return off


def working_days(start: date, end: date) -> int:
    off = set().union(*(holidays(y) for y in range(start.year, end.year + 1)))
    return sum(
        1
        for n in range((end - start).days + 1)
        if (d := start + timedelta(days=n)).weekday() < 5 and d not in off
    )


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def plain(text: str) -> str:
    """Lowercase ASCII, for e-mail addresses: "Đặng Thùy" -> "dang thuy"."""
    text = text.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def round_to(amount: float, step: int) -> int:
    return int(round(amount / step) * step)


# --- The simulation ---------------------------------------------------------------------


@dataclass
class Seat:
    """A place in the organisation chart, held by one employee at a time."""

    position_id: int
    department_id: int
    grade_id: int
    role: str | None  # the staff role, for promotions; None for leaders and team leads
    manager: "Seat | None"
    band_shift: float = 0.0
    leader: bool = False
    holder: "Employee | None" = None


@dataclass
class Employee:
    employee_id: int
    full_name: str
    gender: str
    birth_date: date
    hometown: str
    email: str
    hire_date: date
    seat: Seat
    location: str
    dependents: int
    troublemaker: bool
    termination_date: date | None = None
    probation_end: date | None = None
    degree: str = "Đại học"
    salaries: list[dict] = field(default_factory=list)
    ratings: dict[str, int] = field(default_factory=dict)

    def active_on(self, day: date) -> bool:
        return self.hire_date <= day and (
            self.termination_date is None or self.termination_date >= day
        )

    def active_in(self, start: date, end: date) -> bool:
        return self.hire_date <= end and (
            self.termination_date is None or self.termination_date >= start
        )

    def salary_on(self, day: date) -> dict:
        return [s for s in self.salaries if s["effective_date"] <= day][-1]


class Company:
    def __init__(self, employees: int, start_year: int, years: int, seed: int):
        if employees < 30:
            raise ValueError("The organisation chart needs at least 30 employees.")
        last_year = start_year + years - 1
        if start_year < min(TET) or last_year > max(TET):
            raise ValueError(f"Payroll years must lie within {min(TET)}-{max(TET)}.")
        self.rng = random.Random(seed)
        self.start = date(start_year, 1, 1)
        self.end = date(last_year, 12, 31)
        self.positions: list[tuple] = []
        self.position_ids: dict[tuple, int] = {}
        self.seats = self._organisation(employees)
        self.people: list[Employee] = []
        self.emails: set[str] = set()
        self.education: list[dict] = []
        self.reviews: list[dict] = []
        self.bonuses: list[dict] = []
        self.penalties: list[dict] = []
        self.payroll: list[dict] = []
        self.referrals: dict[int, Employee] = {}  # new hire's id -> the employee who referred
        for seat in self.seats:
            self._staff_initially(seat)

    # The organisation chart.

    def _position(self, title: str, department_id: int, grade_id: int) -> int:
        key = (title, department_id, grade_id)
        if key not in self.position_ids:
            self.position_ids[key] = len(self.positions) + 1
            self.positions.append((len(self.positions) + 1, title, department_id, grade_id))
        return self.position_ids[key]

    def _organisation(self, headcount: int) -> list[Seat]:
        seats: dict = {}
        for key, department, title, grade, manager in LEADERS:
            seats[key] = Seat(
                self._position(title, department, grade),
                department,
                grade,
                None,
                seats.get(manager),
                leader=True,
            )
        for department, role, _, _ in STAFF_ROLES:  # every staff position, for promotions
            for grade, label in STAFF_GRADES.items():
                self._position(f"{role} ({label})", department, grade)
        # Share the remaining seats out by the roles' weights (largest remainder first).
        remaining = headcount - len(LEADERS)
        weights = {d: 0 for d in TEAM_LEAD_TITLES}
        for department, _, weight, _ in STAFF_ROLES:
            weights[department] += weight
        total = sum(weights.values())
        exact = {d: remaining * w / total for d, w in weights.items()}
        counts = {d: int(x) for d, x in exact.items()}
        for d in sorted(exact, key=lambda d: counts[d] - exact[d])[
            : remaining - sum(counts.values())
        ]:
            counts[d] += 1
        all_seats = list(seats.values())
        for department, count in counts.items():
            head = seats[department]
            leads = [
                Seat(
                    self._position(TEAM_LEAD_TITLES[department], department, 5),
                    department,
                    5,
                    None,
                    head,
                    leader=True,
                )
                for _ in range(count // (STAFF_PER_TEAM_LEAD + 1))
            ]
            all_seats += leads
            roles = [r for r in STAFF_ROLES if r[0] == department]
            staff = count - len(leads)
            for n in range(staff):
                role = self.rng.choices(roles, weights=[r[2] for r in roles])[0]
                grade = self.rng.choices(list(STAFF_GRADES), weights=STAFF_GRADE_WEIGHTS)[0]
                manager = leads[n % len(leads)] if leads else head
                all_seats.append(
                    Seat(
                        self.position_ids[
                            (f"{role[1]} ({STAFF_GRADES[grade]})", department, grade)
                        ],
                        department,
                        grade,
                        role[1],
                        manager,
                        band_shift=role[3],
                    )
                )
        return all_seats

    # People.

    def _new_person(self, seat: Seat, hire_date: date, *, troublemaker: bool) -> Employee:
        rng = self.rng
        women = WOMEN_SHARE[seat.department_id]
        gender = "Nữ" if rng.random() < women else "Nam"
        surname = rng.choices([s for s, _ in SURNAMES], weights=[w for _, w in SURNAMES])[0]
        middle = rng.choice(MIDDLE_NAMES[gender])
        given = rng.choice(GIVEN_NAMES[gender])
        low, high = AGES[seat.grade_id]
        age = rng.randint(low, high)
        birth = date(hire_date.year - age, rng.randint(1, 12), rng.randint(1, 28))
        hometown = rng.choice(PROVINCES)
        location = (
            "Hà Nội" if rng.random() < (0.8 if hometown in NORTH else 0.25) else "TP. Hồ Chí Minh"
        )
        age_now = self.end.year - birth.year
        dependents = 0 if age_now < 27 else rng.choices([0, 1, 2, 3], weights=[35, 30, 28, 7])[0]
        email = plain(f"{given}.{surname[0]}{middle[0]}")
        stem, n = email.replace(" ", ""), 1
        email = stem
        while email in self.emails:
            n += 1
            email = f"{stem}{n}"
        self.emails.add(email)
        person = Employee(
            employee_id=len(self.people) + 1,
            full_name=f"{surname} {middle} {given}",
            gender=gender,
            birth_date=birth,
            hometown=hometown,
            email=f"{email}@{EMAIL_DOMAIN}",
            hire_date=hire_date,
            seat=seat,
            location=location,
            dependents=dependents,
            troublemaker=troublemaker,
        )
        self.people.append(person)
        seat.holder = person
        self._educate(person)
        return person

    def _educate(self, person: Employee) -> None:
        rng, seat = self.rng, person.seat
        department = seat.department_id
        graduated = person.birth_date.year + 22
        region = "north" if person.location == "Hà Nội" else "south"
        technical = department in {2, 3, 4}
        if department == 7 and seat.grade_id <= 3:
            degree = rng.choices(["Trung cấp", "Cao đẳng", "Đại học"], weights=[10, 35, 55])[0]
        else:
            degree = "Đại học"
        schools = (TECH_SCHOOLS if technical else BUSINESS_SCHOOLS)[region]
        school = {"Trung cấp": VOCATIONAL, "Cao đẳng": COLLEGES}.get(degree, schools)
        major = rng.choice(MAJORS[department])
        self._record(
            person, "Bằng cấp", degree, major, rng.choice(school), graduated - (degree != "Đại học")
        )
        master = {1: 0, 2: 0.05, 3: 0.15, 4: 0.25, 5: 0.4, 6: 0.45, 7: 0.6}[seat.grade_id]
        if degree == "Đại học" and rng.random() < master:
            degree = "Thạc sĩ"
            self._record(
                person,
                "Bằng cấp",
                degree,
                major,
                rng.choice(schools),
                graduated + rng.randint(2, 6),
            )
        if person.seat.position_id == self.position_ids[("Giám đốc Kỹ thuật", 2, 7)]:
            degree = "Tiến sĩ"
            self._record(
                person,
                "Bằng cấp",
                degree,
                "Khoa học máy tính",
                "Đại học Bách khoa Hà Nội",
                graduated + 9,
            )
        person.degree = degree
        for name, issuer, chance in CERTIFICATES.get(department, []):
            if rng.random() < chance:
                self._record(
                    person, "Chứng chỉ", "Chứng chỉ", name, issuer, graduated + rng.randint(1, 5)
                )
        if rng.random() < 0.3:
            name, issuer = rng.choice(ENGLISH)
            self._record(
                person, "Chứng chỉ", "Chứng chỉ", name, issuer, graduated + rng.randint(-1, 3)
            )

    def _record(self, person, record_type, level, major, institution, year) -> None:
        self.education.append(
            {
                "education_id": len(self.education) + 1,
                "employee_id": person.employee_id,
                "record_type": record_type,
                "degree_level": level,
                "major": major,
                "institution": institution,
                "graduation_year": min(year, self.end.year),
            }
        )

    def _pay_for(self, seat: Seat, degree: str) -> int:
        low, high = BAND[seat.grade_id]
        bonus = {"Trung cấp": -0.1, "Cao đẳng": -0.05, "Thạc sĩ": 0.08, "Tiến sĩ": 0.15}
        where = self.rng.uniform(0.1, 0.6) + seat.band_shift + bonus.get(degree, 0)
        return round_to(low + (high - low) * min(max(where, 0), 1), 100_000)

    def _salary(self, person, day: date, base: int, reason: str) -> None:
        person.salaries.append(
            {
                "employee_id": person.employee_id,
                "effective_date": day,
                "position_id": person.seat.position_id,
                "base_salary": base,
                "position_allowance": POSITION_ALLOWANCE.get(person.seat.grade_id, 0),
                "meal_allowance": int(parameter("meal_allowance", day)),
                "reason": reason,
            }
        )

    def _hire(self, person: Employee) -> None:
        """A new employee's pay: the probation rate first, then the full salary from the
        first of the month after probation."""
        base = self._pay_for(person.seat, person.degree)
        day = person.hire_date
        ends = day + timedelta(days=int(parameter("probation_days", day)))
        person.probation_end = date(ends.year + ends.month // 12, ends.month % 12 + 1, 1)
        rate = parameter("probation_salary_rate", day)
        self._salary(person, day, round_to(base * rate, 100_000), "Tuyển dụng - thử việc")
        self._salary(person, person.probation_end, base, "Hết thử việc")

    def _staff_initially(self, seat: Seat) -> None:
        rng = self.rng
        latest = self.start - timedelta(days=1)
        if seat.leader:
            hired = FOUNDED + timedelta(days=rng.randint(0, 5 * 365))
        elif seat.grade_id == 1:
            hired = latest - timedelta(days=rng.randint(20, 540))
        else:
            hired = latest - timedelta(days=int(rng.triangular(30, (latest - FOUNDED).days, 400)))
        troublemaker = not seat.leader and rng.random() < 0.05
        person = self._new_person(seat, hired, troublemaker=troublemaker)
        if hired > latest - timedelta(days=90):
            self._hire(person)
            return
        last_review = date(self.start.year - 1, 4, 1)
        self._salary(
            person,
            max(hired, last_review),
            self._pay_for(seat, person.degree),
            "Tuyển dụng" if hired > last_review else "Điều chỉnh lương định kỳ",
        )

    # Events, month by month.

    def run(self) -> None:
        day = self.start
        while day <= self.end:
            self._month(day.year, day.month)
            day = month_end(day.year, day.month) + timedelta(days=1)

    def _active(self, day: date) -> list[Employee]:
        return [p for p in self.people if p.active_on(day)]

    def _month(self, year: int, month: int) -> None:
        start, end = date(year, month, 1), month_end(year, month)
        period = f"{year}-{month:02d}"
        if month == 1:
            self._review(
                str(year - 1), date(year - 1, 12, 31), date(year - 1, 10, 1), date(year, 1, 15)
            )
        if month == 4:
            self._raise(start)
        if month == 7:
            self._review(f"{year}-H1", date(year, 6, 30), date(year, 4, 1), date(year, 7, 10))
            self._promote(start)
        tet = TET[year]
        tet_pay = tet if tet.day > 10 else date(tet.year, tet.month, 1) - timedelta(days=1)
        if (tet_pay.year, tet_pay.month) == (year, month):
            self._tet_bonus(start, period)
        if month in (1, 4, 7, 10):
            self._sales_bonus(start, period)
        self._project_bonus(start, end, period)
        self._referral_bonus(start, period)
        self._turnover(start, end)
        self._discipline(start, end, period)
        for person in [p for p in self.people if p.active_in(start, end)]:
            self._pay(person, start, end, period)

    def _review(self, period: str, active_on: date, hired_by: date, reviewed: date) -> None:
        for person in self._active(active_on):
            if person.hire_date > hired_by:
                continue
            weights = [15, 35, 45, 5, 0] if person.troublemaker else [3, 10, 50, 28, 9]
            rating = self.rng.choices([1, 2, 3, 4, 5], weights=weights)[0]
            person.ratings[period] = rating
            manager = person.seat.manager.holder if person.seat.manager else None
            self.reviews.append(
                {
                    "review_id": len(self.reviews) + 1,
                    "employee_id": person.employee_id,
                    "review_period": period,
                    "review_date": reviewed,
                    "rating": rating,
                    "reviewer_id": manager.employee_id if manager else None,
                }
            )

    def _raise(self, day: date) -> None:
        period = str(day.year - 1)
        rises = {1: (0, 0), 2: (0, 0.02), 3: (0.04, 0.06), 4: (0.08, 0.1), 5: (0.12, 0.15)}
        for person in self._active(day):
            if period not in person.ratings or self._raise_delayed(person, day):
                continue
            current = person.salary_on(day)["base_salary"]
            low, high = rises[person.ratings[period]]
            new = min(
                round_to(current * (1 + self.rng.uniform(low, high)), 100_000),
                BAND[person.seat.grade_id][1],
            )
            if new > current:
                self._salary(person, day, new, "Điều chỉnh lương định kỳ")

    def _raise_delayed(self, person: Employee, day: date) -> bool:
        since = day - timedelta(days=183)
        return any(
            p["employee_id"] == person.employee_id
            and p["penalty_type"] == DELAYED_RAISE
            and since <= p["incident_date"] < day
            for p in self.penalties
        )

    def _promote(self, day: date) -> None:
        h1, last_year = f"{day.year}-H1", str(day.year - 1)
        for person in self._active(day):
            seat = person.seat
            if seat.role is None or seat.grade_id >= 4:
                continue
            if person.ratings.get(h1) != 5 or person.ratings.get(last_year, 0) < 4:
                continue
            seat.grade_id += 1
            title = f"{seat.role} ({STAFF_GRADES[seat.grade_id]})"
            seat.position_id = self.position_ids[(title, seat.department_id, seat.grade_id)]
            low, high = BAND[seat.grade_id]
            current = person.salary_on(day)["base_salary"]
            self._salary(
                person, day, min(max(round_to(current * 1.12, 100_000), low), high), "Thăng chức"
            )

    def _bonus(self, person, period: str, kind: str, amount: int, note: str) -> None:
        self.bonuses.append(
            {
                "bonus_id": len(self.bonuses) + 1,
                "employee_id": person.employee_id,
                "pay_period": period,
                "bonus_type": kind,
                "amount": amount,
                "note": note,
            }
        )

    def _tet_bonus(self, start: date, period: str) -> None:
        last_year = start.year - 1  # the year the bonus rewards
        months_by_rating = {1: 0.5, 2: 1.0, 3: 1.0, 4: 1.5, 5: 2.0}
        year_start = date(last_year, 1, 1)
        for person in self._active(start.replace(day=15)):
            if person.hire_date > date(last_year, 12, 31):
                continue
            months = months_by_rating.get(person.ratings.get(str(last_year), 3), 1.0)
            worked = (date(last_year, 12, 31) - max(person.hire_date, year_start)).days + 1
            share = min(worked / 365, 1)
            base = person.salary_on(start)["base_salary"]
            note = f"Thưởng Tết {start.year}" + ("" if share == 1 else " (theo thời gian làm việc)")
            self._bonus(
                person, period, "Thưởng Tết", round_to(base * months * share, 100_000), note
            )

    def _sales_bonus(self, start: date, period: str) -> None:
        quarter_start = date(start.year - (start.month == 1), (start.month - 4) % 12 + 1, 1)
        quarter = (quarter_start.month - 1) // 3 + 1
        for person in self._active(start.replace(day=15)):
            if person.seat.department_id != SALES_DEPARTMENT or person.hire_date > quarter_start:
                continue
            achieved = self.rng.choices([0.0, 0.6, 1.0, 1.4], weights=[20, 30, 35, 15])[0]
            if achieved:
                amount = round_to(person.salary_on(start)["base_salary"] * 0.5 * achieved, 100_000)
                self._bonus(
                    person,
                    period,
                    "Thưởng doanh số quý",
                    amount,
                    f"Doanh số Q{quarter}/{quarter_start.year}",
                )

    def _project_bonus(self, start: date, end: date, period: str) -> None:
        if self.rng.random() > 0.35:
            return
        project = self.rng.choice(PROJECTS)
        team = [
            p
            for p in self._active(end)
            if p.seat.department_id in TECH_DEPARTMENTS and p.hire_date < start
        ]
        for person in self.rng.sample(team, min(len(team), self.rng.randint(3, 6))):
            amount = round_to(self.rng.uniform(3_000_000, 15_000_000), 500_000)
            self._bonus(person, period, "Thưởng dự án", amount, f"Hoàn thành dự án {project}")

    def _referral_bonus(self, start: date, period: str) -> None:
        for hire_id, referrer in self.referrals.items():
            hire = self.people[hire_id - 1]
            if hire.probation_end == start and hire.active_on(start) and referrer.active_on(start):
                self._bonus(
                    referrer,
                    period,
                    "Thưởng giới thiệu ứng viên",
                    5_000_000,
                    f"Giới thiệu {hire.full_name}",
                )

    def _turnover(self, start: date, end: date) -> None:
        """About 8% of the staff leave each year; each seat is filled again within weeks."""
        for person in self._active(start):
            if person.seat.leader or person.termination_date is not None:
                continue
            chance = 0.08 / 12 * (3 if person.troublemaker else 1)
            if self.rng.random() >= chance:
                continue
            left = start + timedelta(days=self.rng.randint(0, (end - start).days))
            while left.weekday() >= 5:
                left -= timedelta(days=1)
            person.termination_date = max(left, person.hire_date)
            hired = person.termination_date + timedelta(days=self.rng.randint(14, 50))
            while hired.weekday() >= 5:
                hired += timedelta(days=1)
            if hired > self.end:
                continue
            newcomer = self._new_person(person.seat, hired, troublemaker=self.rng.random() < 0.05)
            self._hire(newcomer)
            colleagues = [p for p in self._active(start) if p is not person]
            if self.rng.random() < 0.4:
                self.referrals[newcomer.employee_id] = self.rng.choice(colleagues)

    def _discipline(self, start: date, end: date, period: str) -> None:
        for person in self._active(start):
            if person.seat.leader:
                continue
            if self.rng.random() >= (0.15 if person.troublemaker else 0.004):
                continue
            day = start + timedelta(days=self.rng.randint(0, (end - start).days))
            if not person.active_on(day):
                continue
            earlier = [
                p["penalty_type"] for p in self.penalties if p["employee_id"] == person.employee_id
            ]
            kind = self.rng.choices([REPRIMAND, DAMAGE], weights=[70, 30])[0]
            if kind == REPRIMAND and REPRIMAND in earlier:
                kind = DELAYED_RAISE
            amount, note = 0, f"Tái phạm: {self.rng.choice(REPRIMANDS).lower()}"
            if kind == REPRIMAND:
                note = self.rng.choice(REPRIMANDS)
            elif kind == DAMAGE:
                note, low, high = self.rng.choice(DAMAGES)
                cap = person.salary_on(day)["base_salary"] * 0.3  # Bộ luật Lao động, Điều 129
                amount = min(round_to(self.rng.uniform(low, high), 100_000), round_to(cap, 100_000))
            self.penalties.append(
                {
                    "penalty_id": len(self.penalties) + 1,
                    "employee_id": person.employee_id,
                    "incident_date": day,
                    "pay_period": period,
                    "penalty_type": kind,
                    "deduction_amount": amount,
                    "note": note,
                }
            )

    def _pay(self, person: Employee, start: date, end: date, period: str) -> None:
        rng = self.rng
        first = max(start, person.hire_date)
        last = min(end, person.termination_date or end)
        standard = working_days(start, end)
        scheduled = working_days(first, last)
        roll = rng.random()
        if roll < 0.004:
            unpaid = 15  # a long unpaid leave: no insurance that month
        elif roll < (0.3 if person.troublemaker else 0.05):
            unpaid = rng.randint(1, 4 if person.troublemaker else 2)
        else:
            unpaid = 0
        unpaid = min(unpaid, scheduled)
        worked = scheduled - unpaid
        overtime = 0
        if person.seat.grade_id <= 4 and worked:
            tech = person.seat.department_id in TECH_DEPARTMENTS
            if rng.random() < (0.3 if tech else 0.08):
                overtime = rng.randint(4, 24) if tech else rng.randint(2, 8)

        salary = person.salary_on(last)
        base = salary["base_salary"]
        fixed = salary["position_allowance"]
        salary_pay = round(base * worked / standard)
        allowances = round((fixed + salary["meal_allowance"]) * worked / standard)
        overtime_pay = round(base / standard / 8 * parameter("overtime_rate", start) * overtime)
        mine = person.employee_id
        bonus = sum(
            b["amount"]
            for b in self.bonuses
            if b["employee_id"] == mine and b["pay_period"] == period
        )
        penalty = sum(
            p["deduction_amount"]
            for p in self.penalties
            if p["employee_id"] == mine and p["pay_period"] == period
        )
        gross = salary_pay + allowances + overtime_pay + bonus

        insured = worked >= parameter("insurance_min_worked_days", start)
        insurance_salary = base + fixed if insured else 0
        paid, employer = {}, 0
        for kind in ("BHXH", "BHYT", "BHTN"):
            rate, employer_rate, cap = insurance(kind, start)
            paid[kind] = round(min(insurance_salary, cap) * rate)
            employer += round(min(insurance_salary, cap) * employer_rate)
        insurance_total = sum(paid.values())
        personal = int(parameter("personal_deduction", start))
        dependents = int(parameter("dependent_deduction", start)) * person.dependents
        taxable = max(0, gross - insurance_total - personal - dependents)
        tax = income_tax(taxable, start)
        self.payroll.append(
            {
                "payroll_id": len(self.payroll) + 1,
                "employee_id": mine,
                "pay_period": period,
                "position_id": salary["position_id"],
                "standard_days": standard,
                "worked_days": worked,
                "unpaid_leave_days": unpaid,
                "overtime_hours": overtime,
                "base_salary": base,
                "salary_pay": salary_pay,
                "allowances": allowances,
                "overtime_pay": overtime_pay,
                "bonus_total": bonus,
                "gross_income": gross,
                "insurance_salary": insurance_salary,
                "social_insurance": paid["BHXH"],
                "health_insurance": paid["BHYT"],
                "unemployment_insurance": paid["BHTN"],
                "personal_deduction": personal,
                "dependent_deduction": dependents,
                "taxable_income": taxable,
                "personal_income_tax": tax,
                "penalty_total": penalty,
                "net_pay": gross - insurance_total - tax - penalty,
                "employer_insurance": employer,
                "total_employer_cost": gross + employer,
            }
        )

    # Tables.

    def tables(self) -> dict[str, pd.DataFrame]:
        def frame(rows, columns=None, ints=()) -> pd.DataFrame:
            df = pd.DataFrame(rows, columns=columns)
            for column in ints:
                df[column] = df[column].astype("Int64")  # whole numbers, blank when missing
            return df

        employees = [
            {
                "employee_id": p.employee_id,
                "employee_code": f"NV{p.employee_id:04d}",
                "full_name": p.full_name,
                "gender": p.gender,
                "birth_date": p.birth_date,
                "hometown_province": p.hometown,
                "email": p.email,
                "hire_date": p.hire_date,
                "termination_date": p.termination_date,
                "status": LEFT if p.termination_date else ACTIVE,
                "employment_type": PROBATION
                if p.probation_end and p.probation_end > self.end
                else PERMANENT,
                "position_id": p.seat.position_id
                if p.termination_date is None
                else p.salary_on(p.termination_date)["position_id"],
                "manager_id": self._manager_id(p),
                "work_location": p.location,
                "num_dependents": p.dependents,
            }
            for p in self.people
        ]
        salaries = [s for p in self.people for s in p.salaries]
        salaries.sort(key=lambda s: (s["employee_id"], s["effective_date"]))
        salaries = [{"salary_id": n + 1, **s} for n, s in enumerate(salaries)]
        return {
            "departments": frame(
                DEPARTMENTS,
                ["department_id", "department_name", "parent_department_id", "cost_center"],
                ints=["parent_department_id"],
            ),
            "job_grades": frame(
                GRADES, ["grade_id", "grade_code", "grade_name", "min_salary", "max_salary"]
            ),
            "positions": frame(
                self.positions, ["position_id", "position_title", "department_id", "grade_id"]
            ),
            "tax_brackets": frame(
                TAX_BRACKETS,
                ["bracket_no", "lower_bound", "upper_bound", "rate", "effective_from"],
                ints=["upper_bound"],
            ),
            "insurance_rates": frame(
                INSURANCE_RATES,
                [
                    "insurance_type",
                    "employee_rate",
                    "employer_rate",
                    "salary_cap",
                    "effective_from",
                    "effective_to",
                ],
            ),
            "payroll_parameters": frame(
                PAYROLL_PARAMETERS, ["code", "value", "unit", "effective_from", "note"]
            ),
            "employees": frame(employees, ints=["manager_id"]),
            "education": frame(self.education),
            "salary_history": frame(salaries),
            "performance_reviews": frame(self.reviews, ints=["reviewer_id"]),
            "bonuses": frame(self.bonuses),
            "penalties": frame(self.penalties),
            "payroll": frame(self.payroll),
        }

    def _manager_id(self, person: Employee) -> int | None:
        manager = person.seat.manager
        return manager.holder.employee_id if manager and manager.holder else None


def generate(employees: int = 100, start_year: int = 2026, years: int = 1, seed: int = 42):
    """Every table of the company sample, by name."""
    company = Company(employees, start_year, years, seed)
    company.run()
    return company.tables()


def write(tables: dict[str, pd.DataFrame], folder: Path) -> list[Path]:
    """Each table as `<name>.csv` followed by its Data Dictionary, `<name>.dictionary.csv`.
    They upload in any order: a join the dictionary declares to a table not uploaded yet
    waits for it."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = folder / f"{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        dictionary = folder / f"{name}.dictionary.csv"
        data_dictionary(name, table).to_csv(dictionary, index=False, lineterminator="\n")
        paths += [path, dictionary]
    return paths


def data_dictionary(name: str, table: pd.DataFrame) -> pd.DataFrame:
    """The table's Data Dictionary from DICTIONARY, its columns in the table's order.
    Raises ValueError when DICTIONARY describes a column the table doesn't have."""
    about, columns = DICTIONARY[name]
    if unknown := sorted(set(columns) - set(table.columns)):
        raise ValueError(f"DICTIONARY describes columns {name} doesn't have: {unknown}")
    rows = [("", about, "")]
    for column in table.columns:
        entry = columns.get(column)
        if entry is not None:
            description, references = entry if isinstance(entry, tuple) else (entry, "")
            rows.append((column, description, references))
    return pd.DataFrame(rows, columns=["column", "description", "references"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.company_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--employees", type=int, default=100, help="headcount (default 100)")
    parser.add_argument("--year", type=int, default=2026, help="first payroll year")
    parser.add_argument("--years", type=int, default=1, help="how many years of payroll")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="folder for the CSVs")
    parser.add_argument("--load", action="store_true", help="also upload them as Datasets")
    args = parser.parse_args(argv)

    paths = write(generate(args.employees, args.year, args.years, args.seed), args.out)
    for path in paths:
        print(f"{path.name}: {sum(1 for _ in path.open(encoding='utf-8')) - 1} rows")
    if args.load:
        from perry import datasets, db  # importing perry loads .env

        db.init_schema()
        for path in paths:
            for result in datasets.upload(path):
                print(result.describe().split(", columns")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
