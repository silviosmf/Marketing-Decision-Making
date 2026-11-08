"""LLM-assisted classification of the screened abstracts, in batches.

Used by 02_Classification.ipynb, where the provider, the model and the run settings are chosen.

Every batch is saved as soon as it is classified (data/articles/llm_classification/<run>/batch_NNNN.csv,
raw answers in raw/batch_NNNN.json), so an interrupted run (network error, daily quota) resumes where
it stopped when it is started again. Studies that the model leaves out of an answer, or that come back
with an invalid answer, are sent again in smaller batches (down to one study).
"""
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

from mdm import prompt as cp
from mdm.llm_client import LLMClient, LLMError, QuotaExceeded, load_api_key

INPUT_FILE = 'data/articles/screened_articles.csv'
EXAMPLES_FILE = 'data/framework/few_shot_prompt.csv'
OUTPUT_DIR = 'data/articles/llm_classification'
AUDIT_FIELDS = ['run', 'batch', 'provider', 'model', 'model_version', 'temperature', 'reasoning_effort',
                'prompt_version', 'prompt_hash', 'batch_size_sent', 'classified_at']


def load_records(input_file=INPUT_FILE):
    """All records retained by the screening (input of the classification)."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'{input_file} not found: run 01_Merge_and_screening.ipynb first.')
    return pd.read_csv(input_file, dtype=str, keep_default_na=False).reset_index(drop=True)


def system_prompt(examples_file=EXAMPLES_FILE):
    sp = cp.build_system_prompt(cp.load_examples(examples_file))
    os.makedirs('data/framework', exist_ok=True)
    with open(f'data/framework/classification_prompt_v{cp.PROMPT_VERSION}.txt', 'w', encoding='utf-8') as f:
        f.write(sp)
    return sp, cp.prompt_hash(sp)


def _call(client, system, rows, log):
    """Classify rows (DataFrame slice) in one request; returns {record ID: parsed dict}."""
    studies = [(f'S{j + 1}', r['Title'], r.get('Keywords'), r['Abstract']) for j, (_, r) in enumerate(rows.iterrows())]
    key_of = {f'S{j + 1}': r['ID'] for j, (_, r) in enumerate(rows.iterrows())}
    resp = client.generate(system, cp.format_batch(studies))
    parsed, problem = cp.parse_batch_response(resp['text'], [(s[0], s[3]) for s in studies])
    log.append({'ids': [r['ID'] for _, r in rows.iterrows()], 'finish_reason': resp['finish_reason'],
                'model_version': resp['model_version'], 'usage': resp['usage'], 'seconds': resp['seconds'],
                'problem': problem, 'text': resp['text']})
    out = {}
    for sid, d in parsed.items():
        d['model_version'] = resp['model_version']
        d['batch_size_sent'] = len(rows)
        out[key_of[sid]] = d
    return out


def classify_rows(client, system, rows, log, depth=0):
    """Classify a batch; studies missing or invalid in the answer are sent again in halves."""
    got = _call(client, system, rows, log)
    bad = [i for i in rows['ID'] if i not in got or (got[i]['parse_errors'] or '').startswith('invalid JSON')]
    if not bad:
        return got
    retry = rows[rows['ID'].isin(bad)]
    if len(retry) == 1:
        if depth < 2:                       # one study: ask again, at most twice
            got.update(classify_rows(client, system, retry, log, depth + 1))
        else:
            d = {k: None for k in cp.OUTPUT_FIELDS}
            d.update(stage=cp.NOT_IDENTIFIED, ai_intelligence=cp.NOT_IDENTIFIED, strategic_element=cp.NOT_IDENTIFIED,
                     classification=cp.NOT_IDENTIFIED, parse_errors='no valid answer after retries', batch_size_sent=1)
            got[retry['ID'].iloc[0]] = d
        return got
    half = (len(retry) + 1) // 2
    print(f'    {len(retry)} studies without a valid answer; sending them again in two halves')
    got.update(classify_rows(client, system, retry.iloc[:half], log, depth))
    got.update(classify_rows(client, system, retry.iloc[half:], log, depth))
    return got


def consolidate(out_dir, run):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith('batch_') and f.endswith('.csv'))
    if not files:
        return None
    df = pd.concat([pd.read_csv(os.path.join(out_dir, f), dtype=str, keep_default_na=False) for f in files],
                   ignore_index=True)
    df.to_csv(os.path.join(out_dir, f'classification_{run}.csv'), index=False)
    return df


def check_same_source(out_dir, provider, model):
    """Batches already saved must come from the same provider and model; otherwise they would be mixed."""
    files = sorted(f for f in os.listdir(out_dir) if f.startswith('batch_') and f.endswith('.csv')) if os.path.isdir(out_dir) else []
    if not files:
        return
    head = pd.read_csv(os.path.join(out_dir, files[0]), dtype=str, keep_default_na=False, nrows=1)
    old = (head['provider'].iloc[0] if 'provider' in head else '?', head['model'].iloc[0] if 'model' in head else '?')
    if old != (provider, model):
        raise ValueError(f'{out_dir} has batches from {old[0]} / {old[1]}, not {provider} / {model}. '
                         'Move the old batch files and raw/ to a backup folder before running again.')


def run(run, provider, model, batch_size, temperature=0.0, min_interval=7.0, max_output_tokens=32768,
        thinking_budget=None, reasoning_effort=None, input_file=INPUT_FILE, output_dir=OUTPUT_DIR):
    """Classify all screened records; `run` names the output folder (the final classification is run='full')."""
    df = load_records(input_file)
    out_dir = os.path.join(output_dir, run)
    check_same_source(out_dir, provider, model)
    os.makedirs(os.path.join(out_dir, 'raw'), exist_ok=True)
    system, phash = system_prompt()
    size = batch_size
    client = LLMClient(provider, model, load_api_key(), temperature=temperature, min_interval=min_interval,
                       max_output_tokens=max_output_tokens, response_schema=cp.RESPONSE_SCHEMA,
                       thinking_budget=thinking_budget, reasoning_effort=reasoning_effort)
    n_batches = (len(df) + size - 1) // size
    print(f'{run}: {len(df)} records, {n_batches} batches of up to {size}; {provider} / {model}; '
          f'prompt v{cp.PROMPT_VERSION} ({phash})')
    t0 = time.time()
    for b in range(n_batches):
        path = os.path.join(out_dir, f'batch_{b + 1:04d}.csv')
        if os.path.exists(path):
            continue
        rows = df.iloc[b * size:(b + 1) * size]
        log = []
        try:
            got = classify_rows(client, system, rows, log)
        except QuotaExceeded as e:
            print(f'\nDaily quota reached ({e}).\n'
                  f'{b} of {n_batches} batches are saved. Run the same cell tomorrow to continue.')
            return consolidate(out_dir, run)
        except LLMError as e:
            print(f'\nAPI error on batch {b + 1}: {e}\nNothing was saved for this batch; run the cell again.')
            return consolidate(out_dir, run)
        coded = pd.DataFrame([got[i] for i in rows['ID']], index=rows.index)
        audit = pd.DataFrame({'run': run, 'batch': b + 1, 'provider': provider, 'model': model,
                              'temperature': 'default' if temperature is None else temperature,
                              'reasoning_effort': reasoning_effort or '',
                              'prompt_version': cp.PROMPT_VERSION, 'prompt_hash': phash,
                              'classified_at': datetime.now(timezone.utc).isoformat(timespec='seconds')},
                             index=rows.index)
        res = pd.concat([rows, coded, audit], axis=1)
        res = res[list(rows.columns) + cp.OUTPUT_FIELDS + AUDIT_FIELDS]
        with open(os.path.join(out_dir, 'raw', f'batch_{b + 1:04d}.json'), 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
        res.to_csv(path, index=False)
        tokens = sum((c['usage'] or {}).get('totalTokenCount', 0) for c in log)
        print(f'  batch {b + 1}/{n_batches}: {len(rows)} records, {len(log)} call(s), {tokens:,} tokens, '
              f'{time.time() - t0:.0f}s elapsed')
    df_all = consolidate(out_dir, run)
    summary(df_all, run)
    return df_all


def summary(df, run):
    if df is None or df.empty:
        print('Nothing classified yet.')
        return
    print(f'\n{run}: {len(df)} records classified')
    print('Answers with problems:', int((df['parse_errors'] != '').sum()))
    print('Evidence not found verbatim in the abstract:', int((df['evidence_in_abstract'] == 'False').sum()))
    print(pd.crosstab(df['stage'], df['ai_intelligence'], margins=True).to_string())
    print(df['confidence'].value_counts().to_string())


def status(output_dir=OUTPUT_DIR):
    """Batches and records saved in each run folder."""
    runs = sorted(r for r in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, r))) if os.path.isdir(output_dir) else []
    for r in runs:
        d = os.path.join(output_dir, r)
        n = len([f for f in os.listdir(d) if f.startswith('batch_') and f.endswith('.csv')]) if os.path.isdir(d) else 0
        df = consolidate(d, r) if n else None
        print(f'{r}: {n} batches saved, {0 if df is None else len(df)} records')
