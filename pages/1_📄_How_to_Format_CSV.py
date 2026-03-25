"""
ShiftCover – How to Format Your CSV
====================================
Helper page explaining the required demand-file format.
"""

import streamlit as st

st.set_page_config(page_title="How to Format CSV – ShiftCover", layout="wide")
st.title("📄 How to Format Your CSV")

st.markdown("""
The optimiser needs a **demand curve** for each occupation — a single column of
numbers telling it how many workers are required at every 5-minute interval
across the entire week.

---

## Quick summary

| Property | Value |
|---|---|
| **File type** | `.csv` (comma-delimited) or `.xlsx` |
| **Rows** | Exactly **2 016** (= 7 days × 288 intervals/day) |
| **Columns** | At least one numeric column |
| **Preferred column name** | `Required` or `Demand` |
| **Row order** | Monday 00:00 → Sunday 23:55 (chronological) |
| **Interval size** | 5 minutes |

---

## Step-by-step: converting a day/time/headcount table

You may already have your demand data in a table like this:

| Day | Time | Required |
|---|---|---|
| Monday | 00:00 | 2 |
| Monday | 00:05 | 2 |
| Monday | 00:10 | 3 |
| … | … | … |
| Sunday | 23:55 | 1 |

### 1. Make sure every 5-minute slot exists

Each day has **288** intervals (24 h × 12 intervals/h).  
The full week has **7 × 288 = 2 016** rows.

If your source data uses 15- or 30-minute steps you need to expand it first.
The simplest approach: repeat each value for the number of 5-min slots it
spans. For example, a 30-min row becomes 6 identical rows.

### 2. Sort chronologically

Rows must be in order: **Monday 00:00** at the top through
**Sunday 23:55** at the bottom.

### 3. Keep only the demand column

The app looks for a column named `Required` or `Demand` (case-insensitive).
If neither is found it uses the **first numeric column**.

You can include extra columns (Day, Time, etc.) — they will be ignored. But
the numeric demand column must be present.

### 4. Save as CSV

Save the file as a `.csv` using comma delimiters. Example:

```
Day,Time,Required
Monday,00:00,2
Monday,00:05,2
Monday,00:10,3
...
Sunday,23:55,1
```

Or a minimal single-column CSV (no header needed if it is numeric):

```
Required
2
2
3
...
1
```

---

## Minimal CSV example (first 10 rows)

```csv
Required
0
0
0
0
0
0
0
0
0
0
```

These correspond to Monday 00:00 – 00:45 with zero demand.

---

## Tips

- **Multiple occupations**: Upload one file per occupation (e.g. Technicians,
  Labourers). Each file follows the same 2 016-row format.
- **XLSX also works**: You can upload an Excel file instead of CSV. The same
  column-detection rules apply.
- **Zeros are fine**: If you have no demand overnight just fill those rows
  with `0`.
- **Rounding**: Decimal values are rounded to the nearest integer
  automatically.
- **Sample file**: Use the *Generate sample* tab on the main page to create an
  example demand curve, then download it from the XLSX report to see the exact
  format.

---

## Common errors

| Error message | Cause | Fix |
|---|---|---|
| *Need 2016 rows, got N* | File has fewer rows than required | Ensure every 5-min slot for all 7 days is present |
| *No numeric column found* | All columns are text | Add a numeric `Required` column |
| *Missing file for …* | Upload slot left empty | Upload a file for each occupation |
""")
