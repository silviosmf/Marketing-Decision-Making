"""Results of the classification as reported in sections 3.3 and 4.1 of the paper (used by 03_Classification_analysis.ipynb).

The logic follows the text of the paper:
- the articles are counted by AI intelligence (Mechanical, Thinking, Feeling; "Not identified" reported apart);
- the author keywords of the articles of each intelligence are normalized and counted (Fig. 2);
- the most frequent keywords of each intelligence are the criteria to group the articles (Table 3);
- for each keyword, up to the ten most cited and the ten most recent articles are selected (Table 4);
- the articles of each intelligence are counted by publication year (Fig. 3 and Fig. 4).

Figures follow mdm.style (grayscale, Arial/Helvetica), as required by the Journal of Marketing Analytics.
"""
import json
import os
import re
import unicodedata
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from mdm import prompt as cp
from mdm import style

INTEL = cp.INTELLIGENCES                     # ['Mechanical AI', 'Thinking AI', 'Feeling AI']
NI = cp.NOT_IDENTIFIED

# ---------------------------------------------------------------- loading
def load(path='data/articles/llm_classification/full/classification_full.csv'):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['Cited'] = pd.to_numeric(df['Cited'], errors='coerce').fillna(0).astype(int)
    df['keyword_list'] = df['Keywords'].map(keywords_of)
    return df


# ---------------------------------------------------------------- keywords
# Keywords that repeat the concepts of the search (Table 1), name the medium only or name a marketing stage already
# coded by the classifier; they appear across all groups and do not distinguish applications. The first nine come
# from the original analysis of the paper.
GENERIC = ['marketing', 'artificial intelligence', 'machine learning', 'big data', 'data mining', 'social media',
           'internet', 'facebook', 'deep learning', 'decision making', 'decision', 'marketing strategy',
           'marketing research', 'marketing management']

# Spelling variants and acronyms written as the same keyword
SYNONYMS = {
    'ai': 'artificial intelligence', 'ml': 'machine learning', 'nlp': 'natural language processing',
    'crm': 'customer relationship management', 'iot': 'internet of things', 'genai': 'generative ai',
    'gen ai': 'generative ai', 'rfm': 'rfm model', 'k means': 'k means clustering', 'k means algorithm': 'k means clustering',
    'cluster analysis': 'clustering',
    'clustering analysis': 'clustering', 'clustering algorithm': 'clustering', 'opinion mining': 'sentiment analysis',
    'decision support': 'decision support system', 'dss': 'decision support system',
    'e commerce': 'e-commerce', 'ecommerce': 'e-commerce', 'word of mouth': 'word-of-mouth', 'ewom': 'electronic word-of-mouth',
    'electronic word of mouth': 'electronic word-of-mouth',
}
BRITISH = {'behaviour': 'behavior', 'behaviours': 'behaviors', 'personalisation': 'personalization',
           'optimisation': 'optimization', 'customisation': 'customization', 'organisational': 'organizational',
           'analyse': 'analyze', 'modelling': 'modeling', 'utilisation': 'utilization'}
NO_SINGULAR = {'analytics', 'business', 'sales', 'news', 'ethics', 'logistics', 'economics', 'statistics', 'dynamics',
               'tactics', 'services', 'series', 'process', 'access', 'success', 'loss', 'mass', 'bias', 'analysis', 'status',
               'means'}


def _strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def normalize_keyword(k):
    """Lower case, no accents, no acronym in parentheses, American spelling, singular last word, synonyms."""
    k = _strip_acc(str(k)).lower().strip()
    k = re.sub(r'\s*\([^)]*\)\s*', ' ', k)               # "artificial intelligence (ai)" -> "artificial intelligence"
    k = k.replace('–', '-').replace('—', '-')
    k = re.sub(r'(?<=\w)-(?=\w)', ' ', k)                 # "decision-making" -> "decision making"
    k = re.sub(r'[^\w\s]', ' ', k)
    words = [BRITISH.get(w, w) for w in k.split()]
    if words:
        last = words[-1]
        if len(last) > 3 and last.endswith('s') and not last.endswith(('ss', 'us', 'is')) and last not in NO_SINGULAR:
            words[-1] = last[:-3] + 'y' if last.endswith('ies') else last[:-1]
    k = ' '.join(words)
    return SYNONYMS.get(k, k)


def keywords_of(s):
    """Author keywords of a record (';'-separated), normalized, without repetitions."""
    out = []
    for k in re.split(r';', s or ''):
        n = normalize_keyword(k)
        if n and n not in out:
            out.append(n)
    return out


def is_generic(k):
    return k in GENERIC


def keyword_frequencies(df, intelligence):
    """Number of articles of one intelligence that use each normalized keyword (generic keywords removed)."""
    c = Counter(k for ks in df.loc[df['ai_intelligence'] == intelligence, 'keyword_list'] for k in ks if not is_generic(k))
    return pd.DataFrame(c.most_common(), columns=['keyword', 'articles'])


def _contains(keyword, topic):
    return re.search(r'\b' + re.escape(topic) + r'\b', keyword) is not None


def articles_with(df, topic):
    """Boolean mask: the article has a keyword that contains the topic as a whole phrase
    (as in the original analysis, 'segmentation' also counts 'customer segmentation')."""
    return df['keyword_list'].map(lambda ks: any(_contains(k, topic) for k in ks))


def top_keywords(df, intelligence, k=3, pool=15):
    """Keywords that group the most articles of an intelligence (Table 3).

    Candidates are the `pool` most frequent normalized keywords of the intelligence (Fig. 2). Each candidate is then
    counted as in the original analysis, by phrase containment ('segmentation' also counts 'customer segmentation'),
    and the k candidates with the most articles are kept; a candidate that contains, or is contained in, a keyword
    already kept is skipped, so the keywords do not repeat each other."""
    sub = df[df['ai_intelligence'] == intelligence]
    cand = keyword_frequencies(df, intelligence).head(pool)['keyword']
    counts = {t: int(articles_with(sub, t).sum()) for t in cand}
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    chosen = []
    for t, n in ranked:
        if any(_contains(t, c) or _contains(c, t) for c, _ in chosen):
            continue
        chosen.append((t, n))
        if len(chosen) == k:
            break
    return chosen


def label(k):
    """Keyword as printed in the tables (first letter upper case; acronyms upper case)."""
    special = {'rfm': 'RFM', 'crm': 'CRM', 'ai': 'AI', 'iot': 'IoT', 'e-commerce': 'E-commerce', 'b2b': 'B2B'}
    return ' '.join(special.get(w, w) for w in k.split()).replace('e-commerce', 'e-commerce')[:1].upper() + \
        ' '.join(special.get(w, w) for w in k.split())[1:]


# ---------------------------------------------------------------- tables
def table_classification(df):
    """Articles by AI intelligence (Section 4.1). Percentages over all classified records."""
    n = df['ai_intelligence'].value_counts().reindex(INTEL + [NI]).fillna(0).astype(int)
    t = pd.DataFrame({'AI intelligence': n.index, 'Articles': n.values})
    t['% of all records'] = (100 * t['Articles'] / len(df)).round(1)
    t.loc[len(t)] = ['Total', int(n.sum()), 100.0]
    return t


def table_keywords(df, k=3):
    """Table 3: frequency distribution of articles across keywords.

    Frequency = articles of the intelligence whose author keywords contain the keyword; Frequency (%) = that number
    over all articles classified as Mechanical, Thinking or Feeling AI (the denominator of the original table)."""
    denom = int(df['ai_intelligence'].isin(INTEL).sum())
    rows = []
    for intel in INTEL:
        for t, n in top_keywords(df, intel, k=k):
            rows.append({'AI intelligence': intel, 'Keyword': label(t), 'keyword_key': t, 'Frequency': n,
                         'Frequency (%)': round(100 * n / denom, 2)})
    return pd.DataFrame(rows), denom


def _is_initials(tok):
    """True for Scopus initials such as 'J.', 'J.-P.', 'D.-A.' or 'Ș.-A.' (any alphabet)."""
    t = tok.strip()
    return bool(t) and any(ch.isalpha() for ch in t) and all(ch.isupper() or ch in '.-' for ch in t) and len(t) <= 8


def citation(authors, year):
    """'Surname (Year)', 'Surname and Surname (Year)' or 'Surname et al. (Year)' from Scopus or WoS author lists."""
    names = [a.strip() for a in str(authors).split(';') if a.strip()]
    def surname(a):
        if ',' in a:                                     # WoS: "Surname, I"
            return a.split(',')[0].strip()
        toks = a.split()                                 # Scopus: "Surname I.J."
        while len(toks) > 1 and _is_initials(toks[-1]):
            toks.pop()
        return ' '.join(toks)
    s = [surname(a) for a in names]
    y = '' if pd.isna(year) else int(year)
    if not s:
        return f'Anonymous ({y})'
    if len(s) == 1:
        return f'{s[0]} ({y})'
    if len(s) == 2:
        return f'{s[0]} and {s[1]} ({y})'
    return f'{s[0]} et al. ({y})'


def _disambiguate(detail):
    """Harvard style: different works with the same citation get a, b, ... after the year (ordered by title)."""
    first = detail.drop_duplicates('ID')
    for cit, g in first.groupby('Citation'):
        if g['ID'].nunique() > 1:
            for letter, (_, a) in zip('abcdefgh', g.sort_values('Title').iterrows()):
                detail.loc[detail['ID'] == a['ID'], 'Citation'] = cit[:-1] + letter + ')'
    return detail


def table_selected(df, keywords, n=10, n_table=2):
    """Table 4. Candidates: for each intelligence and keyword of Table 3, up to the n most cited and the n most recent
    articles (most recent: latest year, then most cited); they are the supplementary list. Table 4 shows, for each
    keyword, the n_table most cited and the n_table most recent candidates, without repeating an article already
    shown for the same intelligence. Returns (candidates, table)."""
    rows = []
    for _, r in keywords.iterrows():
        sub = df[(df['ai_intelligence'] == r['AI intelligence']) & articles_with(df, r['keyword_key'])]
        cited = sub.sort_values(['Cited', 'Year'], ascending=[False, False]).head(n)
        recent = sub.sort_values(['Year', 'Cited'], ascending=[False, False]).head(n)
        for _, a in pd.concat([cited, recent]).drop_duplicates('ID').iterrows():
            crit = ' and '.join(c for c, d in [('most cited', cited), ('most recent', recent)] if a['ID'] in set(d['ID']))
            rows.append({'AI intelligence': r['AI intelligence'], 'Keyword': r['Keyword'], 'Selected as': crit,
                         'Citation': citation(a['Authors'], a['Year']), 'Title': a['Title'], 'Year': a['Year'],
                         'Cited': a['Cited'], 'DOI': a['DOI'], 'Journal': a.get('Journal', ''),
                         'Stage': a['stage'], 'Strategic element': a['strategic_element'], 'ID': a['ID']})
    detail = _disambiguate(pd.DataFrame(rows))
    detail['In Table 4'] = ''
    shown = {i: set() for i in INTEL}
    table = []
    for _, r in keywords.iterrows():
        intel, kw = r['AI intelligence'], r['Keyword']
        g = detail[(detail['AI intelligence'] == intel) & (detail['Keyword'] == kw)]
        picked = []
        for crit, order in [('most cited', ['Cited', 'Year']), ('most recent', ['Year', 'Cited'])]:
            pool = g[g['Selected as'].str.contains(crit) & ~g['ID'].isin(shown[intel])]
            for _, a in pool.sort_values(order, ascending=[False, False]).head(n_table).iterrows():
                shown[intel].add(a['ID'])
                picked.append(a['Citation'])
                detail.loc[(detail['ID'] == a['ID']) & (detail['Keyword'] == kw), 'In Table 4'] = crit
        table.append({'AI intelligence': intel, 'Keyword': kw, 'Articles': ', '.join(picked)})
    return detail, pd.DataFrame(table)


def table_annual(df):
    """Fig. 4 data: articles per publication year and intelligence."""
    d = df[df['ai_intelligence'].isin(INTEL)]
    t = pd.crosstab(d['Year'], d['ai_intelligence']).reindex(columns=INTEL, fill_value=0)
    years = range(int(d['Year'].min()), int(d['Year'].max()) + 1)
    t = t.reindex(years, fill_value=0)
    t['Total'] = t.sum(axis=1)
    t.index.name = 'Year'
    return t


def table_two_year(df):
    """Articles in two-year groups starting at the first publication year of the corpus (as in the original analysis)."""
    y0 = int(df['Year'].min())
    d = df[df['ai_intelligence'].isin(INTEL)].copy()
    d['group'] = d['Year'].map(lambda y: y0 + 2 * ((int(y) - y0) // 2))
    t = pd.crosstab(d['group'], d['ai_intelligence']).reindex(columns=INTEL, fill_value=0)
    t['Total'] = t.sum(axis=1)
    last = int(df['Year'].max())
    t.index = [f'{g}–{g + 1}' if g + 1 <= last else f'{g}' for g in t.index]
    t.index.name = 'Years'
    return t


# ---------------------------------------------------------------- Fig. 1 (example of a model answer)
def model_answer(record_id, raw_dir='data/articles/llm_classification/full/raw'):
    """The JSON returned by the model for one record, taken from the raw answers of the full run."""
    for f in sorted(os.listdir(raw_dir)):
        for call in json.load(open(os.path.join(raw_dir, f), encoding='utf-8')):
            if record_id in call['ids']:
                sid = f'S{call["ids"].index(record_id) + 1}'
                text = call['text']
                data = json.loads(text[text.find('{'):text.rfind('}') + 1])
                items = data.get('results', [data])
                for it in items:
                    if it.get('id') == sid:
                        return {k: v for k, v in it.items() if k != 'id'}
    return None


def example_record(df):
    """Record shown in Fig. 1: the most cited article coded with high confidence, a verbatim quotation and no warning."""
    ok = df[(df['confidence'] == 'high') & (df['evidence_in_abstract'] == 'True') & (df['parse_errors'] == '')
            & df['ai_intelligence'].isin(INTEL)]
    return ok.sort_values(['Cited', 'Year'], ascending=False).iloc[0]


# ---------------------------------------------------------------- figures
def fig_keyword_tree(t3):
    """Fig. 2: each AI intelligence (strategic decision) linked to its Table 3 keywords, in the layout of the paper."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    style.use()
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis('off')
    box = dict(boxstyle='round,pad=0.4,rounding_size=0.8', facecolor='#eeeeee', edgecolor='#555555', linewidth=0.7)
    for y0, h, title in [(33, 10, 'AI intelligence (strategic decision)'), (1, 30, 'Keywords')]:
        ax.add_patch(Rectangle((1, y0), 98, h, fill=False, edgecolor='#999999', linewidth=0.7))
        ax.text(98, y0 + h - 1.2, title, ha='right', va='top', fontsize=7.5, fontweight='bold', color='#333333')
    for i, intel in enumerate(INTEL):
        x = 4 + 32 * i                                     # left edge of the column
        ax.add_patch(FancyBboxPatch((x, 35.5), 15, 3.6, **box))
        ax.text(x + 7.5, 37.3, intel, ha='center', va='center', fontsize=8)
        kws = t3.loc[t3['AI intelligence'] == intel, 'Keyword'].tolist()
        ys = [24 - 8.5 * j for j in range(len(kws))]
        ax.plot([x + 3, x + 3], [35.1, ys[-1]], color='#555555', linewidth=0.7)
        for k, y in zip(kws, ys):
            ax.plot([x + 3, x + 7], [y, y], color='#555555', linewidth=0.7)
            ax.add_patch(FancyBboxPatch((x + 7.4, y - 1.9), 21, 3.8, **box))
            ax.text(x + 17.9, y, k, ha='center', va='center', fontsize=8)
    fig.tight_layout(pad=0.2)
    return fig


def fig_classification(df):
    """Fig. 3: number of articles by AI intelligence."""
    style.use()
    n = df['ai_intelligence'].value_counts().reindex(INTEL).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ax.bar(n.index, n.values, color=style.GRAYS[:3], edgecolor='black', linewidth=0.6)
    for b, h in zip(bars, style.HATCHES[:3]):
        b.set_hatch(h)
    for b, v in zip(bars, n.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Articles')
    ax.set_ylim(0, n.max() * 1.12)
    fig.tight_layout()
    return fig


def fig_annual(df):
    """Fig. 4: annual distribution of the articles of each intelligence."""
    style.use()
    t = table_annual(df)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    marks = ['o', 's', '^']
    lines = ['-', '--', ':']
    for intel, c, m, ls in zip(INTEL, style.GRAYS[:3], marks, lines):
        ax.plot(t.index, t[intel], color=c, marker=m, markersize=3, linestyle=ls, linewidth=1.1, label=intel)
    ax.set_xlabel('Publication year')
    ax.set_ylabel('Articles')
    ax.set_xlim(t.index.min() - 1, t.index.max() + 1)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- saving
def save_tables(tables, folder='tables'):
    os.makedirs(folder, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(os.path.join(folder, f'{name}.csv'), index=not isinstance(t.index, pd.RangeIndex), encoding='utf-8')
    return folder
