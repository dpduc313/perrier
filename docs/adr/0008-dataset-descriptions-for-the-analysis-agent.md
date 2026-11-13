# Datasets carry descriptions, value profiles and join hints, read in two steps

Until now the Analysis Agent knew a Dataset only by its name, row count, and column names and types. So it guessed what columns meant (is `total_employer_cost` in VND? with insurance?), which values to filter on (`'Đang làm việc'` or `'active'`), and how to join. On the company sample it ran `information_schema` and `SELECT *` queries just to find the department tree.

Each Dataset's registry entry now also holds:
- **Descriptions** of the Dataset and its columns, from a Data Dictionary the user uploads. Neither is required: a Dataset or column the Data Dictionary leaves out gets a description written by a model in the background, marked as the model's guess. A Dataset's description says what one row is.
- **A value profile** of each column, computed when the Dataset is uploaded.
- **Join hints**: the ones the Data Dictionary declares, and failing those, ones inferred from column names.

The Agent reads all this in two steps. `list_datasets` stays short, and a new `describe_datasets(names)` tool gives the column detail for the Datasets the Agent is about to query.

## Considered Options

- **Descriptions only from the model**: rejected. The model can only guess meaning from a column name and a few values, and a confident wrong description (gross or net? which currency?) is worse than none, because the Agent trusts it. The model only fills gaps, and its descriptions say they are guesses.
- **Descriptions only from the Data Dictionary**: rejected. A Dataset uploaded without one would gain nothing but the profile, and most uploads won't come with one.
- **Postgres `COMMENT ON` for the descriptions**: rejected. Replacing a Dataset drops and recreates its table, which loses the comments. Everything the Agent reads about Datasets already comes from the registry.
- **Everything in `list_datasets`**: rejected. With descriptions and profiles, every analysis Turn would carry a few thousand tokens for the 13 company-sample Datasets, growing with each upload, whichever Datasets the question needs. The cost is one more tool call per Turn, which the eval gate below measures.
- **One fixed name, `data_dictionary.csv`**: rejected. Among data files it reads as one more table, it stops anyone uploading a Dataset of that name, and per-Dataset dictionaries would all share it and overwrite each other on disk.
- **A non-tabular format (JSON, YAML)**: rejected. It would look unlike data, but the people who write descriptions work in spreadsheets, so a Data Dictionary stays a CSV or a sheet and its name says what it is.
- **Real foreign keys in the `analytics` schema**: rejected. Datasets would have to be uploaded in dependency order, and one orphaned id would refuse a whole file. Join hints guide the Agent's SQL and never refuse data.

## Consequences

- **A Data Dictionary is recognised by the `.dictionary` suffix** and isn't stored as a Dataset. Its columns are `dataset`, `column`, `description` and optional `references` (e.g. `employees.employee_id`).
  - `payroll.dictionary.csv` sits beside `payroll.csv` and describes `payroll`. A blank `dataset` means the name before `.dictionary`, so a one-Dataset dictionary can leave that column out.
  - `company.dictionary.csv` names its Datasets in `dataset` and describes several at once. Any row may name any Dataset, whatever the file is called; one with a blank `dataset` describes the Dataset named like the file, and waits like any other row if there is none.
  - In a spreadsheet, a sheet named `dictionary` describes the other sheets, naming each by its sheet name.
  - A row with an empty `column` describes the Dataset itself.
  - Names go through the same normalisation as uploads, so a Data Dictionary can use the file's own names: `Lương tháng.csv` or `Base Salary` match the Dataset `lương_tháng` and the column `base_salary`.
  - A Dataset and its dictionary can be uploaded together, in either order. Uploading a dictionary replaces the descriptions of the Datasets it names and leaves the others alone.
- **A Data Dictionary is checked when it is uploaded**, and the upload reply lists what it found. The checks can't tell whether a description is true, only whether it fits the data:
  - It is refused whole when it lacks the `description` column (or, for a workbook's `dictionary` sheet, the `dataset` column) or describes the same column twice. A refused `dictionary` sheet leaves the workbook's other sheets to upload.
  - A row for a Dataset not uploaded yet waits for that Dataset.
  - A row for a column the Dataset doesn't have is reported and isn't attached.
  - A declared `references` whose target Dataset isn't uploaded yet waits for it, and is checked when it arrives.
  - A declared `references` whose target column doesn't exist, or none of whose values appear in the target, is refused. One with some orphaned values is kept, and the orphan count is shown to the Agent and in the upload reply.
- **Model descriptions are only as good as a guess.** The model sees just the names and the profile, writes one sentence per column, and its descriptions are marked so in the Datasets panel and to the Agent. The Data Dictionary is how a user corrects one.
- **Replacing a Dataset keeps descriptions** for columns that still exist by name. The model describes only new columns, and the Dataset itself unless the Owner described it.
- **The model writes in the background**, like Summaries (ADR-0003). An upload never waits for it, and until it finishes a column has its profile and no description.
- **Which model writes them**: `PERRY_DATASET_DESCRIPTION_MODEL` names it, `inet-qwen` (the company's model, ADR-0006) or `local` (the local summary model, `qwen3:8b`). It defaults to `inet-qwen` when `PERRY_INET_QWEN_BASE_URL` is set, and to `local` otherwise; any other value stops Perry at start. On the gate for this ADR, `qwen3:8b` returned broken JSON for 3 of the 13 company-sample Datasets; `inet-qwen`, with the request below, described all 13 in about 20 seconds. Either way there is no fallback to another model: a Dataset whose describing fails stays usable, is marked failed and is retried on the next start. Summaries, RAPTOR and Document metadata stay on the local model.
- **The request asks only for what is missing.** A Dataset the Owner already described isn't mentioned in it, because a model told to leave that key null echoed the line instead of replying with the JSON object. A reply that is just the columns' object counts.
- **The profile comes from the data alone, so it is always correct**:
  - the share of empty values
  - min and max for numbers, and for text that reads as ISO dates
  - every distinct value for text columns with at most 20 of them, otherwise three examples
- **A name-inferred join hint**: a column named `x_id` whose values all appear in another Dataset's first column of the same name, checked again whenever either Dataset is uploaded. Declared hints win over inferred ones, and only declared hints cover self-references and renamed keys (`manager_id` → `employees.employee_id`).
- **Profiles send real values to the chat model**, a few per column. Query results already go there in every analysis Turn, so the privacy boundary doesn't move.
- **The company sample ships a Data Dictionary per Dataset** (`payroll.dictionary.csv` beside `payroll.csv`, and so on), the pattern a user follows when adding one table.
- **Eval gate**, the thresholds of ADR-0006 against `main`:
  - the analysis Turns of the eval set
  - five company-sample questions with SQL-checked answers, through `scripts.ask`
  - the cost per Turn before and after
  
  If the extra `describe_datasets` call costs more than it saves in failed queries, the fallback is column descriptions inside `list_datasets` for small libraries only.

  It passed on 2026-10-21 (#74). On the normal flow, the analysis Turns scored 17/18 routing and Citation checks, the same as `main`'s paid reference run, with a judge score 9.1% lower. The company-sample questions went from 3/5 correct on `main` to 5/5. They read about 1.6× the input tokens at the same number of calls, except where `main` explored until the Token Ceiling.
