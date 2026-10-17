# Plausible but Wrong: Silent Analytical Errors in LLM-Generated Charts

*A pilot study. Juliet Uyo Ukwella, October 2026.*

When people ask a chatbot to turn a data question into a chart, the code usually runs and the chart usually looks reasonable. The question this pilot asks is how often that chart is nonetheless **analytically wrong**: built on a defensible-looking choice (totals instead of rates, a running total instead of new cases, a regional row counted as a country) that answers a different question from the one asked, without anything on the screen to say so. I call these **silent analytical errors**.

The pilot tests three things:

1. **How common** are such errors when a chatbot answers ordinary, underspecified data questions?
2. **Prevention:** does asking the model to state its analytical choices *before* writing code reduce them?
3. **Detection:** can a model reliably spot them when asked to check finished code, and can ordinary readers?

## Design

**Data.** The Our World in Data COVID-19 dataset, final release (data to 14 August 2024). The file is not included in this repository; it can be downloaded from
`https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv`.

**Questions.** 28 questions phrased as a typical user would ask them:

- **25 trap questions** (T01–T25), each built around an analytical choice the question leaves open, such as whether to compare totals or rates, whether rows like "World" or "High-income countries" count as countries, or whether a column is daily or weekly.
- **3 control questions** (C01–C03), fully specified, to check that the model can do the basic task.

For every question, an **answer key** describing a valid chart, a misleading chart and what the data actually show was written **before any model output was seen** and was never changed or shown to the model.

**Model.** ChatGPT (free plan, default model, Think and Memory off), each question in a new chat, 15–16 October 2026.

**Procedure.**

| Step | What was done |
|---|---|
| 1. Baseline | Each question sent with a plain prompt; the returned Python code run **exactly as written** and the chart labelled. |
| 2. State choices | Each question sent again, asking the model first to list the choices the question leaves open, say which it will take and why, and flag anything that could mislead. |
| 3. Check afterwards | Each baseline script sent to a new chat with the question, asking whether the chart answers it correctly ("Correct" or "Problem:"). |
| 4. People check | Eight baseline charts (all four silent errors and four valid charts) shown with their questions to five lay readers, each answering alone: "Is this chart a correct answer to the question?" |

**Labels.** Each chart was labelled against the answer key as one of:

- **Valid** / **Valid – choice stated:** an acceptable choice; the second label means the model also said so or added the needed caveat.
- **Flagged analytical error:** a misleading chart, but the model's text names the problem or the better measure.
- **Silent analytical error:** the code runs and the chart looks fine, but it uses a misleading choice and nothing in the code or text says so.
- **Visible error:** the code crashes, or the chart is empty or unreadable.

Model code was never corrected; crashes were recorded as results, and where a question was re-run the new result replaced the old one, as decided in advance. Where a case did not fit the answer key cleanly, the reasoning is written in the Notes column, and judgement calls are marked as such.

## Results

**Steps 1 and 2: baseline vs. stating choices first (25 trap questions)**

| Label | Baseline | State choices |
|---|---|---|
| Valid (incl. choice stated) | 12 (48%) | 20 (80%) |
| Flagged analytical error | 8 (32%) | 4 (16%) |
| **Silent analytical error** | **4 (16%)** | **0 (0%)** |
| Visible error | 1 (4%) | 1 (4%) |
| **Any analytical error** | **12 (48%)** | **4 (16%)** |

All baseline code ran without error, yet nearly half the trap questions produced an analytically wrong chart. Asking the model to state its choices first removed every silent error and cut analytical errors from 48% to 16%, with all three control questions still answered correctly.

**Step 3: asking the model to check finished code (28 baseline scripts)**

| Baseline label | Count | Model named the real problem | Model said "Problem" for another reason | Model said "Correct" |
|---|---|---|---|---|
| Silent error | 4 | 2 | 2 | 0 |
| Flagged error | 8 | 7 | 0 | 1 |
| Visible error | 1 | 1 | 0 | 0 |
| Valid (incl. 3 controls) | 15 | n/a | 13 false alarms (87%) | 2 |

**Step 4: lay readers (5 people, 8 charts each)**

| Charts | Judged "correct" |
|---|---|
| Silent-error charts | 14 of 20 (70%) |
| Valid charts | 19 of 20 (95%) |

## What the pilot suggests

1. **Silent errors are real and convincing.** One in six trap questions produced a chart that was wrong with no warning, and lay readers judged those charts correct 70% of the time (against 95% for valid charts). The chart readers rejected most often (T13, by 3 of 5) was the one showing an obviously impossible value, a vaccination rate of 127%.
2. **Prevention worked better than detection.** Stating choices before coding corrected most mistakes in how data were selected and combined: taking one country per continent (T05), computing a continental share from whichever countries happened to report that day (T06), plotting a year with no data (T15), dropping most countries through a case threshold (T22) and hiding a reporting gap (T24). It did not stop the model from choosing a measure it had itself argued against: in T01 and T21 it explained why deaths per million would be fairer, then ranked by total deaths anyway. And it never detected that the "daily" figures were weekly totals (T08, T10).
3. **Checking afterwards was unreliable.** The model said "Problem" for 25 of 28 scripts, including 13 of 15 valid charts. It caught half of the silent errors, chiefly those visible in the code's logic or in documented facts about the data, and missed those that require looking at the data itself (weekly totals read as daily values; UK nations listed alongside the UK). Its most common objection, that countries "may be compared at different dates", was raised in most checks but mattered in only a few.
4. **Caution has a cost.** In one case (T13), the model's own validation check found a genuine data problem and stopped with an error, trading a silent wrong chart for no chart at all.

## Limitations

This is a pilot, designed to show that the problem exists and to suggest where to look, not to estimate precise rates.

- **One model, one run per question.** Chatbot output varies between runs; a full study would sample each question repeatedly and compare several models.
- **One labeller.** All labels were assigned by the author; a full study would use independent labellers and report their agreement.
- **Small numbers.** 25 trap questions, four baseline silent errors and five lay readers. The percentages describe this sample only.
- **Convenience sample.** The five readers were people known to the author, not a representative sample of the public, and saw the charts in one fixed order.
- **Model version.** The free ChatGPT interface does not report an exact model version, and some responses drew on web search.
- **One dataset.** All questions use a single public health dataset; the traps are specific to its structure.

## Repository contents

| File or folder | Contents |
|---|---|
| `pilot_analysis.ipynb` | Every chart, regenerated from the model's unedited code, with an assessment under each. |
| `Pilot_Silent_Chart_Errors.xlsx` | The full record: About sheet, Summary, questions with answer key and exact prompts, Run log (56 runs; T13 with the state-choices prompt produced no chart), Detection log (28 checks) and People check. |
| `outputs/` | One folder per question with the model's response, code and chart for each condition, and its check response. |

## Reproducing the analysis

1. Open `pilot_analysis.ipynb` in Jupyter. Its first cell downloads the dataset from the address above and saves it as `owid-covid-data.csv`.
2. Run all cells. T13 (state choices) stops with an error raised by the model's own validation check; that error is its recorded result. To continue, select the cell below it and use *Run → Run Selected Cell and All Below*. Both T20 scripts download World Bank income groups and need an internet connection.
3. To re-run the experiment itself, use the prompts in the Questions sheet of the workbook. Because model output varies, new runs will not reproduce these exact charts.
