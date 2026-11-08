"""Prompt and response parser for the LLM-assisted classification (version 2.1, 07/11/2026).

Version 2.1 = version 2.0 (definitions, rules R1-R7, reference applications, worked examples)
with the output section written for batches: the model codes several studies per request and
returns {"results": [...]} with one entry per study id. A batch of one study uses the same format.

Each abstract is coded on the two dimensions of the strategic AI framework of
Huang, M.-H. and Rust, R.T. (2021) 'A strategic framework for artificial intelligence in
marketing', Journal of the Academy of Marketing Science, 49(1), pp. 30-50,
doi:10.1007/s11747-020-00749-9 (open access, CC BY 4.0):

- marketing stage: research, strategy (STP) or action (4Ps/4Cs);
- AI intelligence: mechanical, thinking or feeling.

Stage x intelligence gives the nine strategic elements of Table 1 of the article
(data collection, market analysis, customer understanding, segmentation, targeting,
positioning, standardization, personalization, relationalization).

The classifier receives only the records retained by the rule-based screening
(criteria E1-E3, section 3.2 of the paper). It does not judge relevance again.

The prompt has a fixed part (definitions, decision rules, reference applications and
worked examples from data/framework/few_shot_prompt.csv), sent as the system instruction,
and a variable part with a batch of studies (id, title, author keywords, abstract), sent as
the user message. The fixed part comes first so that providers can cache it.

The module has no dependency on any LLM library.
"""
import hashlib
import json
import re

import pandas as pd

PROMPT_VERSION = '2.1'

RESEARCH, STRATEGY, ACTION = 'Marketing research', 'Marketing strategy', 'Marketing action'
MECHANICAL, THINKING, FEELING = 'Mechanical AI', 'Thinking AI', 'Feeling AI'
NOT_IDENTIFIED = 'Not identified'

STAGES = [RESEARCH, STRATEGY, ACTION]
INTELLIGENCES = [MECHANICAL, THINKING, FEELING]
MARKETING_MIX = ['Product/Consumer', 'Price/Cost', 'Place/Convenience', 'Promotion/Communication']
CONFIDENCE = ['high', 'medium', 'low']

# Table 1 of Huang and Rust (2021): strategic element of each stage x intelligence
ELEMENTS = {
    (RESEARCH, MECHANICAL): 'Data collection',
    (RESEARCH, THINKING): 'Market analysis',
    (RESEARCH, FEELING): 'Customer understanding',
    (STRATEGY, MECHANICAL): 'Segmentation',
    (STRATEGY, THINKING): 'Targeting',
    (STRATEGY, FEELING): 'Positioning',
    (ACTION, MECHANICAL): 'Standardization',
    (ACTION, THINKING): 'Personalization',
    (ACTION, FEELING): 'Relationalization',
}

# Columns added to each record by parse_response()
OUTPUT_FIELDS = [
    'stage', 'ai_intelligence', 'strategic_element', 'marketing_mix',
    'secondary_stage', 'secondary_ai_intelligence', 'secondary_strategic_element', 'ai_techniques',
    'evidence', 'evidence_in_abstract', 'rationale', 'confidence', 'classification', 'parse_errors',
]

INTRO = """You code studies for a systematic literature review on artificial intelligence (AI) in marketing decision-making. You apply the strategic AI framework of Huang and Rust (2021, "A strategic framework for artificial intelligence in marketing", Journal of the Academy of Marketing Science, 49, 30-50). For each study you receive its title, author keywords and abstract. All studies have already passed a screening that confirmed that AI, marketing and decision-making are substantive topics, so do not judge their relevance: code every study. Code only what the abstract reports; do not infer content that is not there.

Work in two steps: (1) code the marketing stage that the AI supports; (2) code the AI intelligence by the purpose of the AI. Then apply the decision rules and return the coding in the JSON format of the Output section."""

STEP_STAGE = """## Step 1. Marketing stage (the marketing decision that the AI supports)
Huang and Rust (2021) describe marketing planning as a research-strategy-action cycle.
- Marketing research: understanding the market, the firm, the competitors and the customers. Strategic elements: data collection, market analysis, customer understanding.
- Marketing strategy (STP): segmentation (slicing the market into groups of customers with distinct needs), targeting (choosing the segments or customers to pursue) and positioning (the place of the product in customers' minds).
- Marketing action (4Ps/4Cs): executing the strategy through product/consumer (including customer service), price/cost (payment, price setting, negotiation), place/convenience (retailing, frontline, distribution, logistics, delivery) and promotion/communication (advertising, digital and social media marketing, sales communication). For this stage also code the marketing-mix area."""

STEP_INTELLIGENCE = """## Step 2. AI intelligence (by the purpose of the AI, not by the technique)
- Mechanical AI is designed for automating repetitive and routine tasks. Its benefit is standardization: consistent, reliable outcomes at scale. Typical technologies: sensors, IoT, remote sensing, robots and cobots, self-service technologies, classification and clustering algorithms, dimensionality reduction.
- Thinking AI is designed for processing data, often unstructured, to arrive at new conclusions or decisions. Its benefit is personalization. Typical technologies: machine learning, neural networks and deep learning, text mining, speech and facial recognition, predictive analytics, expert systems, recommender systems.
- Feeling AI is designed for two-way interactions with humans and/or for analyzing human feelings and emotions. Its benefit is relationalization. Typical technologies: sentiment analysis, natural language processing of customer language, chatbots and conversational agents, embodied or embedded virtual agents, social robots, affective analytics."""


def _matrix_text():
    lines = ['## Strategic elements (stage x intelligence; Huang and Rust 2021, Table 1)',
             '| Stage | Mechanical AI | Thinking AI | Feeling AI |', '|---|---|---|---|']
    lines.append('| Marketing research | Data collection: automate continuous market and customer data sensing, tracking, collecting and processing '
                 '| Market analysis: use marketing analytics to identify competitors and competitive advantages '
                 '| Customer understanding: use emotional data and customer analytics to understand existing and potential customer needs and wants |')
    lines.append('| Marketing strategy (STP) | Segmentation: use mechanical AI to identify novel customer preference patterns '
                 '| Targeting: use thinking AI to recommend the best target segments '
                 '| Positioning: use feeling AI to develop positioning that resonates with customers |')
    lines.append('| Marketing action: product/consumer | Standardization: automate the process and output of meeting customer needs and wants '
                 '| Personalization: personalize products based on customer preferences '
                 '| Relationalization: understand and meet customer emotional needs and wants |')
    lines.append('| Marketing action: price/cost | Standardization: automate the process of price setting and payment '
                 '| Personalization: personalize prices based on customer willingness to pay '
                 '| Relationalization: negotiate price and justify the cost interactively |')
    lines.append('| Marketing action: place/convenience | Standardization: automate customer access to product '
                 '| Personalization: personalize frontline interactions '
                 '| Relationalization: personalize experience for customer engagement |')
    lines.append('| Marketing action: promotion/communication | Standardization: automate communication with customers '
                 '| Personalization: customize promotional content for personal communication '
                 '| Relationalization: tailor communication based on customer emotional preferences and reactions |')
    return '\n'.join(lines)


RULES = """## Decision rules
- R1. Purpose over technique. Code the intelligence by what the AI is used for in the study, not by the name of the technique; an application can share features of more than one intelligence. Facial recognition used to identify who a customer is, is thinking AI; facial recognition used to read a customer's emotional state, is feeling AI.
- R2. Feelings and two-way interaction. AI that analyzes customers' feelings, sentiments, perceptions, needs and wants, or that interacts with customers in two-way communication (chatbots, conversational agents, social robots), is feeling AI even when it is built with thinking-AI techniques such as machine learning, deep learning or NLP.
- R3. Routine patterns versus new decisions. Routine and repetitive work is mechanical AI: sensing, tracking and collecting data, executing processes automatically, and pattern discovery that slices customers into groups (clustering or classifying customers into segments). AI that learns from data to reach a new conclusion or decision for a market, segment or customer (prediction, recommendation, optimization of price, target or content) is thinking AI, even when the abstract calls the system "automated" or "automatic".
- R4. The stage is the marketing decision supported. Choosing which segments or customers to pursue is strategy (targeting); executing ad targeting or retargeting in media is action (promotion). Studies that develop or compare AI methods to analyze market or customer data are marketing research. The technique and the industry (for example, healthcare, finance or tourism) do not define the stage.
- R5. Customers' responses to AI. When the study examines customers' general attitudes toward AI (acceptance, resistance, trust, perceived warmth), code Marketing research and Feeling AI (customer understanding). When it evaluates an AI deployed in a specific marketing action (for example, chatbots making sales calls or robots serving customers in the frontline), code the stage, marketing-mix area and intelligence of that deployment.
- R6. One primary cell, at most one secondary. Code the primary stage and intelligence from the main purpose stated in the abstract. Fill the secondary fields only when the abstract states a second, distinct use of AI; otherwise set them to null.
- R7. Do not guess. If the abstract does not allow assigning the stage or the intelligence, use "Not identified" for that field and set confidence to "low". Code conceptual and review articles by the stage and intelligence they focus on; if they cover all of them without a main focus, use "Not identified"."""

OUTPUT_FORMAT = """## Output
You receive one or more studies, each with an id. Code each study independently, as if it were the only one: do not let one study influence the coding of another. Return only one JSON object, {"results": [...]}, with one entry per study, in the order received. Each entry has these keys and allowed values:
{
  "id": the id of the study, exactly as received,
  "evidence": the shortest exact quotation from the abstract that shows what the AI is used for,
  "ai_techniques": list of the AI techniques or applications named in the abstract (empty list if none),
  "rationale": one or two sentences that apply the definitions and rules,
  "stage": "Marketing research", "Marketing strategy", "Marketing action" or "Not identified",
  "ai_intelligence": "Mechanical AI", "Thinking AI", "Feeling AI" or "Not identified",
  "marketing_mix": "Product/Consumer", "Price/Cost", "Place/Convenience" or "Promotion/Communication" when the stage is Marketing action, otherwise null,
  "secondary_stage": null or a stage (rule R6),
  "secondary_ai_intelligence": null or an intelligence (rule R6),
  "confidence": "high", "medium" or "low"
}
The worked examples below show the coding of one study (without the id)."""

# JSON schema of the answer, for APIs that support structured output (OpenAPI subset)
RESPONSE_SCHEMA = {
    'type': 'OBJECT',
    'properties': {'results': {'type': 'ARRAY', 'items': {
        'type': 'OBJECT',
        'properties': {
            'id': {'type': 'STRING'},
            'evidence': {'type': 'STRING'},
            'ai_techniques': {'type': 'ARRAY', 'items': {'type': 'STRING'}},
            'rationale': {'type': 'STRING'},
            'stage': {'type': 'STRING', 'enum': STAGES + [NOT_IDENTIFIED]},
            'ai_intelligence': {'type': 'STRING', 'enum': INTELLIGENCES + [NOT_IDENTIFIED]},
            'marketing_mix': {'type': 'STRING', 'enum': MARKETING_MIX, 'nullable': True},
            'secondary_stage': {'type': 'STRING', 'enum': STAGES, 'nullable': True},
            'secondary_ai_intelligence': {'type': 'STRING', 'enum': INTELLIGENCES, 'nullable': True},
            'confidence': {'type': 'STRING', 'enum': CONFIDENCE},
        },
        'required': ['id', 'evidence', 'ai_techniques', 'rationale', 'stage', 'ai_intelligence', 'confidence'],
        'propertyOrdering': ['id', 'evidence', 'ai_techniques', 'rationale', 'stage', 'ai_intelligence',
                             'marketing_mix', 'secondary_stage', 'secondary_ai_intelligence', 'confidence'],
    }}},
    'required': ['results'],
}


def _clean(x):
    return None if pd.isna(x) or str(x).strip() == '' else str(x).strip()


def _reference_text(examples):
    """Short application descriptions per strategic element (Tables 2 and 3 of the article)."""
    lines = ['## Reference applications (Huang and Rust 2021, Tables 2 and 3)']
    ex = examples[examples['Prompt role'].isin(['reference_application', 'practice_example', 'worked_example'])]
    for (stage, intel), element in ELEMENTS.items():
        sub = ex[(ex['Strategic decision'] == stage) & (ex['AI intelligence'] == intel)]
        if stage == ACTION:
            for mix in MARKETING_MIX:
                s = sub[sub['Marketing mix'] == mix]
                items = list(dict.fromkeys(s['Summary'].map(_clean).dropna()))
                if items:
                    lines.append(f'- {element} ({intel}, {stage}, {mix}): ' + '; '.join(items) + '.')
        else:
            items = list(dict.fromkeys(sub['Summary'].map(_clean).dropna()))
            lines.append(f'- {element} ({intel}, {stage}): ' + '; '.join(items) + '.')
    return '\n'.join(lines)


def example_coding(row):
    """Expected JSON for a worked example row of few_shot_prompt.csv."""
    stage = row['Strategic decision']
    return {
        'stage': stage,
        'ai_intelligence': row['AI intelligence'],
        'marketing_mix': _clean(row['Marketing mix']) if stage == ACTION else None,
        'secondary_stage': _clean(row['Secondary stage']),
        'secondary_ai_intelligence': _clean(row['Secondary AI intelligence']),
        'ai_techniques': json.loads(row['AI techniques']) if _clean(row['AI techniques']) else [],
        'evidence': _clean(row['Evidence']),
        'rationale': _clean(row['Rationale']),
        'confidence': _clean(row['Confidence']),
    }


def format_study(title, keywords, abstract, study_id=None):
    """One study as it appears in the user message (and in the worked examples, without id)."""
    kw = _clean(keywords) or 'not available'
    tag = f'<study id="{study_id}">' if study_id is not None else '<study>'
    return (f'{tag}\nTitle: {_clean(title) or "not available"}\nAuthor keywords: {kw}\n'
            f'Abstract: {_clean(abstract) or ""}\n</study>')


def format_batch(studies):
    """User message for a batch. studies: list of (study_id, title, keywords, abstract)."""
    body = '\n'.join(format_study(t, k, a, study_id=i) for i, t, k, a in studies)
    ids = ', '.join(str(s[0]) for s in studies)
    return (f'<studies>\n{body}\n</studies>\n'
            f'Code each of the {len(studies)} studies (ids: {ids}) independently. '
            'Return the JSON object {"results": [...]} with one entry per study, in the same order, and nothing else.')


def _examples_text(examples):
    ex = examples[examples['Prompt role'] == 'worked_example'].sort_values('Example order')
    lines = ['## Worked examples (studies cited by Huang and Rust 2021; the rule illustrated is in brackets)']
    for n, (_, row) in enumerate(ex.iterrows(), start=1):
        rule = _clean(row['Rule'])
        tag = f' [{rule}]' if rule and rule != 'Definitions' else ''
        lines.append(f'### Example {n}{tag}: {row["Study"]}')
        lines.append(format_study(row['Title'], None, row['Abstract']))
        lines.append('Coding: ' + json.dumps(example_coding(row), ensure_ascii=False))
    return '\n'.join(lines)


def load_examples(path='data/framework/few_shot_prompt.csv'):
    return pd.read_csv(path)


def build_system_prompt(examples):
    """Fixed part of the prompt (system message)."""
    parts = [INTRO, STEP_STAGE, STEP_INTELLIGENCE, _matrix_text(), RULES,
             _reference_text(examples), OUTPUT_FORMAT, _examples_text(examples)]
    return '\n\n'.join(parts)


def prompt_hash(system_prompt):
    return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:12]


# ---------------------------------------------------------------- response parsing
def _match(value, allowed):
    """Map a model value to an allowed label (case and spacing tolerant)."""
    if value is None:
        return None
    v = re.sub(r'\s+', ' ', str(value)).strip().lower()
    if v in ('', 'null', 'none', 'n/a'):
        return None
    for a in allowed:
        if v == a.lower():
            return a
    # "mechanical" -> "Mechanical AI"; "Marketing strategy (STP)" -> "Marketing strategy".
    # Accepted only when exactly one label matches ("marketing" or "AI" alone are rejected).
    hits = [a for a in allowed if (len(v) >= 4 and v in a.lower()) or a.lower() in v]
    return hits[0] if len(hits) == 1 else False


def _extract_json(text):
    text = str(text).strip()
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find('{'), text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError('no JSON object in the response')
        text = text[start:end + 1]
    return json.loads(text)


def _norm(s):
    return ' '.join(str(s).replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').split()).lower()


def parse_response(text, abstract=None):
    """Parse and validate a model response. Returns a dict with OUTPUT_FIELDS; problems go to parse_errors."""
    out = {k: None for k in OUTPUT_FIELDS}
    errors = []
    try:
        d = _extract_json(text)
    except Exception as e:                  # noqa: BLE001
        out['parse_errors'] = f'invalid JSON: {e}'
        out['classification'] = NOT_IDENTIFIED
        return out

    def pick(key, allowed, extra=()):
        val = _match(d.get(key), list(allowed) + list(extra))
        if val is False:
            errors.append(f'{key}: unexpected value {d.get(key)!r}')
            return None
        return val

    out['stage'] = pick('stage', STAGES, [NOT_IDENTIFIED]) or NOT_IDENTIFIED
    out['ai_intelligence'] = pick('ai_intelligence', INTELLIGENCES, [NOT_IDENTIFIED]) or NOT_IDENTIFIED
    mix = pick('marketing_mix', MARKETING_MIX)
    if out['stage'] == ACTION:
        if mix is None:
            errors.append('marketing_mix missing for Marketing action')
        out['marketing_mix'] = mix
    elif mix is not None:
        errors.append('marketing_mix given for a stage other than Marketing action (ignored)')
    out['strategic_element'] = ELEMENTS.get((out['stage'], out['ai_intelligence']), NOT_IDENTIFIED)

    s2, i2 = pick('secondary_stage', STAGES), pick('secondary_ai_intelligence', INTELLIGENCES)
    if (s2 is None) != (i2 is None):
        errors.append('secondary stage and intelligence must be given together (ignored)')
        s2 = i2 = None
    if s2 is not None and (s2, i2) == (out['stage'], out['ai_intelligence']):
        s2 = i2 = None                      # same as primary
    out['secondary_stage'], out['secondary_ai_intelligence'] = s2, i2
    out['secondary_strategic_element'] = ELEMENTS.get((s2, i2)) if s2 else None

    tech = d.get('ai_techniques') or []
    out['ai_techniques'] = json.dumps(tech if isinstance(tech, list) else [str(tech)], ensure_ascii=False)
    out['evidence'] = d.get('evidence')
    if abstract is not None and out['evidence']:
        out['evidence_in_abstract'] = _norm(out['evidence']) in _norm(abstract)
    out['rationale'] = d.get('rationale')
    conf = pick('confidence', CONFIDENCE)
    out['confidence'] = conf

    # Column used by 03_Classification_analysis.ipynb (previous versions of the prompt)
    out['classification'] = out['ai_intelligence']
    out['parse_errors'] = '; '.join(errors) or None
    return out


def parse_batch_response(text, studies):
    """Parse a batch response. studies: list of (study_id, abstract).

    Returns (parsed, problem): parsed maps study_id -> dict with OUTPUT_FIELDS for the studies found in
    the response; studies missing from it are left out so the caller can send them again. problem is
    None or a short description of what went wrong with the whole response.
    """
    try:
        d = _extract_json_any(text)
    except Exception as e:                  # noqa: BLE001
        return {}, f'invalid JSON: {e}'
    items = d.get('results') if isinstance(d, dict) else d
    if isinstance(d, dict) and items is None and 'stage' in d:
        items = [d]                         # single object for a batch of one
    if not isinstance(items, list):
        return {}, 'no "results" list in the response'
    abstracts = dict(studies)
    parsed = {}
    for k, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        sid = str(item.get('id', '')).strip()
        if sid not in abstracts and len(items) == len(studies):
            sid = str(studies[k][0])        # id missing or altered: fall back to the position
        if sid in abstracts and sid not in parsed:
            parsed[sid] = parse_response(json.dumps(item), abstract=abstracts[sid])
    return parsed, None


def _extract_json_any(text):
    text = str(text).strip()
    fence = re.search(r'```(?:json)?\s*([\[{].*?[\]}])\s*```', text, re.S)
    if fence:
        text = fence.group(1)
    starts = [i for i in (text.find('{'), text.find('[')) if i != -1]
    if not starts:
        raise ValueError('no JSON in the response')
    start = min(starts)
    end = max(text.rfind('}'), text.rfind(']'))
    return json.loads(text[start:end + 1])
