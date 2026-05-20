# PyCol complexity tool — team slides (non-technical)

**Use this version** when people ask: *What was hard? What did you do? Why does it matter?*

Copy each slide into PowerPoint / Google Slides (about 2 minutes per slide).

---

## Slide 1 — The challenge

### Title
**Making dataset “complexity” scores practical for real data**

### What we are measuring
- Tools like **PyCol** summarise how **hard** a dataset is for machine learning (e.g. overlap between classes, local structure).
- Some scores only need simple summaries of the table.
- Others need **every pair of rows compared** — that means building a large **distance table** (who is similar to whom).

### What was the problem?
1. **Very slow** — the standard PyCol way of building that table uses nested loops over all rows. On thousands of rows it becomes painfully slow.
2. **Heavy memory** — the full table still has to exist in memory for those scores (we did not remove that requirement).
3. **Real datasets are messy** — missing values, text categories, different column types. We clean and encode data first so the maths is consistent.

### What we needed
- Keep **the same definitions** as PyCol (so results stay comparable and trustworthy).
- Make the **slow step much faster** so we can run batches and use the app on more datasets.
- Let users choose: **quick run** (no distance table) vs **full run** (with distance-based scores).

### One sentence for Q&A
> *“PyCol’s distance step didn’t scale; we sped up only that step and left the actual complexity scores unchanged.”*

---

## Slide 2 — What we did and what you get

### Title
**Faster distance table — same PyCol scores**

### How we solved it (plain language)
| Step | What it means |
|------|----------------|
| **1. Clean the data** | Handle missing values; turn text categories into numbers the tool understands. |
| **2. Two modes** | **Quick:** only fast scores (no big distance table). **Full:** build the table, then run scores like N2, N3, C1, C2. |
| **3. Faster table build** | We reimplemented the distance calculation with efficient maths (same rules, less waiting). |
| **4. Checked it** | On wine and other sets, our table **matches PyCol’s** to numerical precision; sample scores (e.g. N3) agree. |

### What did *not* change?
- The **meaning** of PyCol metrics (N3 still means the same thing).
- The need for enough **RAM** on large datasets when you run the full mode.
- The recommendation to **subsample** very large datasets (e.g. a few thousand rows) for exploratory runs.

### What you can do in the app
- **Skip distance table** — fast, good for screening many datasets.
- **Build distance table** — slower, needed for structure-based complexity (N2, N3, C1, C2).
- **Batch script** — run the same settings across wine, breast cancer, adult, etc., and save one CSV for comparison.

### Results in practice
- **Wine / medium sets:** full mode becomes usable in the UI and batch jobs.
- **Very large sets (e.g. CDC):** still need row limits or patience — the table grows with dataset size squared.
- **Trust:** validation script confirms our fast build matches PyCol’s original table.

### One sentence for Q&A
> *“We built a faster engine for the distance table; PyCol still reads that table and computes the same complexity numbers.”*

---

## Optional backup slide (if someone asks about categories)

### Title
**How do categories work?**

- In our pipeline, **text categories are converted to 0/1 columns** before distances are computed (standard ML preprocessing).
- Distances are then computed with the **numeric** rule (scaled differences), not a separate “category mismatch” rule on raw labels.
- **Wine** is all numeric already — no category issue there.
- This matches how we use PyCol in the app today; raw ARFF-style categorical handling is only relevant for special PyCol file workflows.

---

## Cheat sheet — likely questions & short answers

| Question | Short answer |
|----------|----------------|
| What was the challenge? | PyCol’s pairwise distance step was too slow for interactive use and batches. |
| How did you solve it? | Faster distance-table build; same PyCol formulas afterward. |
| Is it still PyCol? | Yes for the **scores**; only the **table build** is our optimised version. |
| Skip vs Build? | Skip = fast, fewer metrics. Build = full metrics that need the distance table. |
| Did you prove it? | Yes — automated check vs PyCol on wine (matrices and N3 match). |
| Euclidean distance? | No — still HEOM-style (feature-wise scaling), not straight Euclidean. |
| Why is large data still hard? | The full n×n table is inherent to those metrics; we made building it faster, not smaller. |
