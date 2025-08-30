# Weighted PageRank vs. Classic PageRank for Query‑Focused Retrieval (WikiSpeedia)

> Course project for **“Topics in Mathematics & Its Applications (Applications of Linear Algebra in Data Science)”** — *Instructor: Dr. Shakeri*.

This repository implements a small **information‑retrieval (IR)** pipeline on the **WikiSpeedia** link graph to study how **Weighted PageRank (WPR)** compares to **classic PageRank (PR)** for surfacing documents relevant to a user query. It combines a **textual ranking** stage (inverted index + **BM25**) with a **link‑analysis** stage (PR/WPR) run on a **query‑focused subgraph**.

---

## ✨ What’s inside

- A clean **preprocessing and inverted index** (tokenization, stop‑word removal, Porter stemming).
- **BM25** retrieval (default `k1=1.5`, `b=0.75`) to build the initial set of candidates.
- A **root/base set** expansion strategy to form a **query‑specific subgraph**.
- **Classic PageRank** and **Weighted PageRank** over the subgraph using **sparse power iteration**.
- A simple **evaluation protocol** that compares PR vs. WPR on prefix quality (κ‑score and relevant@k).
- Compact, **sparse** data structures throughout (`scipy.sparse`) for efficiency.

> **Theoretical reference**: WPR is implemented following **Xing & Ghorbani (2004)**, “An Improvement of PageRank: A Weighted PageRank Algorithm.”  
> **Dataset**: The experiments use **WikiSpeedia** (the “wikispeedia” Wikipedia click/graph dataset, obtained from Stanford).

---

## 📁 Expected repository layout

```
.
├── Project_6.ipynb               # Main notebook (end-to-end pipeline)
├── articles.tsv                  # (id, title) table from WikiSpeedia
├── links.tsv                     # directed edges between articles
└── plaintext_articles/           # one .txt file per article (title.txt)
```

> The notebook asserts the presence of `articles.tsv`, `links.tsv`, and the folder `plaintext_articles/` and will raise a helpful error if any are missing.

---

## 🔎 Scenario: from query → root set → base set → ranking

1. **User query.** Example in the notebook:
   ```
   QUERY = "pictish kings and scottish history"
   ```
2. **Root set (textual stage).**  
   We scan up to 200 documents (configurable) and collect those that **match the query** (OR/AND matching selectable).  
   - Text is **lower‑cased**, tokenized with `r"[A-Za-z]{2,}"`, **English stop‑words** removed, and **Porter‑stemmed**.
   - We build an **inverted index** with term frequencies (`tf`), document frequencies (`df`), and lengths (`doc_len`), then compute **BM25** scores:
     \[
     \mathrm{BM25}(d,q) = \sum_{t \in q}
       \underbrace{\log\!\left(\frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t) + 0.5} + 1\right)}_{\mathrm{idf}(t)}
       \cdot
       \frac{f_{t,d}(k_1+1)}{f_{t,d} + k_1\left((1-b)+b\,\frac{|d|}{\overline{|d|}}\right)}
     \]
     with defaults **`k1=1.5`**, **`b=0.75`**.  
     The **root set** is the list of BM25‑matching documents (limited to 200 by default).
3. **Base set (link‑expansion).**  
   Starting from the root set, we **add all in‑neighbors and out‑neighbors** (via `links.tsv`). The union forms the **base set**, a query‑focused subgraph that captures local link structure around the textual matches.
4. **Graph construction.**  
   We index base‑set nodes, build a **sparse adjacency** `A` (`csc`), **keep only edges within the base set**, and compute **in/out degrees**. Dangling columns are handled explicitly.
5. **Link‑analysis ranking.**
   - **Classic PageRank**: column‑stochastic transition matrix from `A`, damping `d=0.85`, teleportation `1-d`, solved by **sparse power iteration** with early stopping (`L1` change `< tol`).
   - **Weighted PageRank (WPR)**: transition weights are biased by in‑ and out‑degrees, following Xing & Ghorbani. For an edge \(v \to u\) (with \(u \in R(v)\), the references of \(v\)):
     \[
     W_{in}(u,v)=\frac{\mathrm{in}(u)}{\sum_{k\in R(v)} \mathrm{in}(k)},\qquad
     W_{out}(u,v)=\frac{\mathrm{out}(u)}{\sum_{k\in R(v)} \mathrm{out}(k)}
     \]
     and the final edge weight is \( \alpha\,W_{in}(u,v) + (1-\alpha)\,W_{out}(u,v)\) (default **\(\alpha=0.5\)**). The resulting matrix is normalized to be **column‑stochastic** and fed to the **same** power iteration solver.
6. **Re‑ranking within the root set.**  
   After PR/WPR are computed on the base‑set graph, we **reorder only the root‑set** documents by their PR/WPR scores and compare the two rankings.

---

## 📊 Evaluation protocol (κ‑score & relevant@k)

We assign a **soft relevance label** to every document using the BM25 scores as weak supervision:

- Compute the set of **positive** BM25 scores (>0).  
  Let `median = median(positive)` and `vr_threshold = 0.8 * max(positive)`.
- Label each doc **VR** (very relevant) if `score ≥ vr_threshold`, **R** (relevant) if `score ≥ median`, **WR** (weakly relevant) otherwise; **IR** if `score = 0`.

A simple **utility** weight is used during prefix evaluation:
```
VR: 1.0,   R: 0.5,   WR: 0.1,   IR: 0.0
```

We then compare PR vs. WPR on top‑k prefixes (k ∈ {5,10,20,40,50}) using:

- **relevant@k**: how many {VR,R} appear in the top‑k.
- **κ‑score**: a **prefix‑utility** over the top‑k based on the weights above (higher is better).

The notebook prints a JSON summary like:
```json
{
  "PR":  { "count": [..], "kappa": [..] },
  "WPR": { "count": [..], "kappa": [..] }
}
```
and renders a bar chart titled **“Relevance comparison – PR vs WPR.”**

---

## ⚙️ Implementation details & engineering choices

- **Sparse everything.** Graphs and transition matrices use `scipy.sparse` (`csc/coo`), and we normalize **in place** to avoid densification.
- **Efficient eigenvector.** The principal stationary distribution is obtained via a **sparse power iteration** with damping and an explicit **dangling‑mass correction**, which is significantly lighter than dense eigensolvers and fits naturally with sparse matrices.
- **Better sparsity at query time.** By working on the **base set** (root ± neighbors), the transition matrix is **much sparser** than the full graph, accelerating convergence and reducing memory.
- **Text processing.** Tokenization with `r"[A-Za-z]{2,}"`, **English stop‑words** via NLTK, and **Porter stemming** to reduce vocabulary size and improve recall.
- **Configurable search mode.** The root‑set collection supports **OR** (any term) or **AND** (all terms) matching before BM25 scoring.

---

## 🚀 Getting started

### 1) Environment
- Python **3.10+** recommended
- Install dependencies:
  ```bash
  pip install numpy pandas scipy matplotlib nltk
  ```
- First run only (for NLTK stop‑words):
  ```python
  import nltk
  nltk.download('stopwords')
  ```

### 2) Data
Place the following in the repository root:
- `articles.tsv` — mapping from article **id** to **title**  
- `links.tsv` — directed edges between article titles/ids  
- `plaintext_articles/` — folder containing one **.txt** file per article, named after the **title** (e.g., `Alan_Turing.txt`)

### 3) Run
Open **`Project_6.ipynb`** and execute the cells.  
You can customize:
- `QUERY` — the user query string  
- `collect_200(..., limit=200, mode="OR")` — root‑set size and matching mode  
- `bm25(query, bm, k1=1.5, b=0.75)` — BM25 parameters  
- `power(M, d=0.85, tol=1e-10, max_iter=100)` — PageRank solver parameters  
- `alpha` in the WPR weighting (if exposed in the code)

---

## 📈 Interpreting results

- **If WPR > PR** on κ‑score and relevant@k, it indicates that **degree‑aware weighting** is beneficial for the topic; authoritative pages with strong in‑/out‑link profiles are prioritized.
- **If PR ≈ WPR**, the local graph around the query may be fairly uniform, or the textual BM25 already isolates relevant hubs.
- **If PR > WPR**, heavy out‑degree bias can sometimes pull rank toward navigational hubs; tune **α** or restrict the base set further.

---

## 🧭 Notes & tips

- **Dangling pages** (no out‑links) are handled by redistributing their probability mass **uniformly** each iteration.
- The transition matrices are explicitly **column‑stochastic**.  
- Ensure that article titles in `links.tsv` **match** file stems in `plaintext_articles/`; otherwise edges are dropped.
- You can combine **BM25** and **(W)PR** linearly for a **hybrid score** (not enabled by default), e.g., `λ·BM25 + (1-λ)·PR`.
- For larger experiments, consider switching to **ARPACK** (`scipy.sparse.linalg.eigs`) or **power iteration with acceleration** once the base‑set matrix is formed.

---

## 📚 References

- **Brin, S., & Page, L. (1998).** The anatomy of a large-scale hypertextual Web search engine. *Computer Networks*.  
- **Xing, W., & Ghorbani, A. (2004).** An Improvement of PageRank: A Weighted PageRank Algorithm. *IEEE WI 2004*.  
- **WikiSpeedia Dataset.** (Used for this project; originally released by Stanford).

---

## 🙏 Acknowledgements

This project was developed as an assignment for *“Topics in Mathematics & Its Applications (Applications of Linear Algebra in Data Science)”* under **Dr. Shakeri**.

---

## 📄 License

Add your preferred license (e.g., MIT) here.
