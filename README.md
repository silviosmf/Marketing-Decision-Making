# AI in the Marketing Decision-Making: Applied Framework in Organizational Contexts

Code and data of a systematic literature review (PRISMA 2020) on the role of artificial intelligence (AI) in marketing decision-making. Records from Scopus and the Web of Science Core Collection are merged, screened with explicit rules and classified with a large language model (LLM) on the strategic AI framework of Huang and Rust (2021, *Journal of the Academy of Marketing Science*, 49, 30-50): marketing stage (research, strategy, action) × AI intelligence (mechanical, thinking, feeling).

## Pipeline

Run the notebooks in order from the repository root.

| Notebook | Paper | What it does | Main outputs |
|---|---|---|---|
| `01_Merge_and_screening.ipynb` | 3.1, 3.2, Fig. A1 | Merges the two exports, removes duplicates, records without abstract and retracted publications, and applies the screening criteria E1-E3 | `data/articles/screened_articles.csv` (1,600 records), `screening_log.csv`, `figures/Fig_A1_PRISMA_2020.*` |
| `02_Classification.ipynb` | 3.3 | Classifies the screened abstracts with the LLM: pilot of 50 records, then all 1,600 | `data/articles/llm_classification/pilot/`, `.../full/classification_full.csv` |
| `03_Classification_analysis.ipynb` | 3.3, 4.1 | Figures and tables of the classification (Figs. 1-4, Tables 3-4); Table 4 shows the two most cited and two most recent articles per keyword, and `tables/Table_4_candidates.csv` lists all filtered articles (supplementary material) | `tables/`, `figures/` |

Flow (collections of 05/11/2026 and 06/11/2026): 4,174 records identified (Scopus 2,475; WoS 1,699) → 1,070 duplicates → 3,104 unique → 8 without abstract and 1 retracted removed → 3,095 screened → 1,495 excluded (E1 = 1,285; E2 = 196; E3 = 14) → **1,600 records classified**.

Final classification: OpenAI `gpt-5.4-nano-2026-03-17`, prompt version 2.1, reasoning effort low, default temperature, pilot in batches of 10 abstracts and full run in batches of 25. Every row keeps the provider, the model version, the temperature, the prompt version and hash and the date of the request; the raw answers of the model are in `raw/batch_NNNN.json`.

## Repository

```
01_Merge_and_screening.ipynb
02_Classification.ipynb
03_Classification_analysis.ipynb
mdm/                         code used by the notebooks
  screening.py               screening rules E1-E3 (section 3.2)
  prisma.py                  PRISMA 2020 flow diagram (Fig. A1)
  prompt.py                  classification prompt v2.1, batch format, JSON schema and parser
  llm_client.py              REST client of the LLM API (rate limit, retries, daily quota)
  classify.py                batch classification (resumable), pilot sample, human-coding sheet, Cohen's kappa
  analysis.py                tables and figures of sections 3.3 and 4.1
  style.py                   figure style (grayscale, Arial/Helvetica, PNG 300 dpi + PDF + SVG)
data/
  articles/
    scopus/                  raw Scopus export (05/11/2026)
    wos/                     raw Web of Science export (06/11/2026)
    LOG_coleta_2026-11-05.md search strings, filters and access of the collections
    all_articles.csv         unique records after deduplication (3,095)
    all_articles_meta.csv    same records with database IDs, document type, journal and database keywords
    duplicates_log.csv       one row per merge of duplicate records
    collection_flow.csv      counts of each step (PRISMA)
    screened_articles.csv    records retained by the screening (input of the classification)
    screening_log.csv        decision and criterion of every screened record (supplementary material)
    screening_validation_sample.csv  blind sample (100 retained + 100 excluded) for the authors to check the rules
    llm_classification/
      pilot/                 50 records: batch_*.csv, raw/, classification_pilot.csv, pilot_human_coding.csv
      full/                  1,600 records: batch_*.csv, raw/, classification_full.csv
  framework/
    few_shot_prompt.csv      studies of Tables 2 and 3 of Huang and Rust (2021); 13 worked examples
    classification_prompt_v2.1.txt  exact prompt sent to the model
tables/                      tables of the paper (Figs. 1-4, Tables 3-4)
figures/                     figures of the paper (Fig. A1, Figs. 2-4)
tests/                       checks of the screening rules, the prompt, the parser and the API client
```

## Running

1. `pip install -r requirements.txt`
2. Only for `02_Classification.ipynb`: create a file named `.env` in this folder with one line, `LLM_API_KEY=your-key` (the file is ignored by git). The provider and the model are set in the first cell of the notebook. Batches already saved are kept, so an interrupted run continues from the next batch; batches from another provider or model are never mixed in.
3. Human check of the pilot: code the 50 records of `pilot/pilot_human_coding.csv` blind and run section 2 of `02_Classification.ipynb` (percent agreement and Cohen's kappa).
4. `01` and `03` do not call any API and reproduce the data files, tables and figures from the files above.

## Tests

`python tests/test_pipeline.py`, `python tests/test_classification_prompt.py` and `python tests/test_llm_client.py`.
