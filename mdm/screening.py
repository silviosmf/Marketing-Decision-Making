"""Rule-based screening of titles, abstracts and author keywords (section 3.2 of the paper).

Each record gets three flags: AI context (C1), marketing context (C2) and decision-making context (C3).
A record is excluded for the first criterion it fails:

- E1 AI absent or mentioned only in passing: no AI term used in a technical sense (E1a), or an AI term
  that appears once, as background, outside the title and author keywords and without an applied
  technique (E1b). Generic uses of search terms ("recommendations" to managers, "predict" in hypotheses,
  "autonomous motivation") and fsQCA do not count as AI.
- E2 Marketing absent or mentioned only in passing: "marketing" only in a non-managerial sense (E2a),
  absent without enough marketing vocabulary (E2b), mentioned once in a list of sectors or functions (E2c),
  or mentioned once without any other marketing term (E2d).
- E3 No decision-making context: "decision" only as a technical (decision tree, DMU) or legal term, and no
  other reference to decisions, choice, managerial action or strategy, including the keywords assigned by
  the databases (Keywords Plus, Scopus Index Keywords).

Run on 07/11/2026 over 3,095 records: 1,600 retained; E1 = 1,285, E2 = 196, E3 = 14.
"""
import re


# ---------- C1: AI context ----------
AI_STRONG = [
    r'\bai\b', r'\ba\.i\.', r'artificial[- ]intelligen', r'machine[- ]learning', r'deep[- ]learning', r'neural[- ]net',
    r'natural[- ]language[- ]processing', r'\bnlp\b', r'text[- ]mining', r'text[- ]analytic', r'data[- ]mining',
    r'big[- ]data', r'data[- ]science', r'chat-?bots?\b', r'conversational agent', r'virtual assistant', r'voice assistant',
    r'computer[- ]vision', r'image recognition', r'facial recognition', r'\bfuzzy\b', r'genetic algorithm',
    r'evolutionary algorithm', r'particle swarm', r'internet of things', r'\biot\b', r'k-means', r'\bclustering algorithm',
    r'soft[- ]computing', r'smart technolog', r'large language model', r'\bllms?\b', r'generative (ai|artificial|model|pre-?trained)',
    r'chatgpt', r'robo-?advis', r'\bgpt\b', r'\bgpt-\d', r'robot', r'cobot', r'recommender', r'recommendation (system|engine|algorithm|agent|model|technolog)',
    r'collaborative filtering', r'content-based filtering', r'predictive (analytic|model|algorithm|system)', r'prediction (model|algorithm|system|technique)',
    r'random forest', r'support vector', r'decision tree', r'gradient boost', r'xgboost', r'bayesian network', r'reinforcement learning',
    r'supervised learning', r'unsupervised', r'sentiment analysis', r'opinion mining', r'topic model', r'latent dirichlet',
    r'expert system', r'decision support system', r'intelligent (agent|system|decision|technolog|automation|assistant)',
    r'algorithmic', r'algorithm (aversion|appreciation)', r'\b(learning|clustering|classification|recommendation|recommender|genetic|evolutionary|pricing|platform|search|matching|ranking|bandit|knn|nearest[- ]neighbou?r|ai|ml|machine|intelligent|heuristic|swarm|colony|optimi[sz]ation|prediction|predictive|forecasting|segmentation|mining)[- ]algorithms?', r'automation', r'\bautomated\b', r'automating', r'autonomous (vehicle|car|agent|system|robot|driving|deliver|store|shop|technolog|machine|decision)',
    r'self-driving', r'smart (speaker|device|product|home|retail|store|system|technolog)',
    r'(marketing|data|business|digital|social media|social|customer|web|text|predictive|prescriptive|advanced|big data|consumer|people|sales|retail|learning|visual|video|location) analytics', r'web mining',
    r'social media mining', r'mining (techniques|algorithms|methods)',
    r'cognitive computing', r'na[iï]ve bayes', r'classifiers?\b', r'\bclustering\b', r'association rules?', r'pattern mining', r'process mining',
    r'sequential recommendation', r'self-attentive', r'attention (mechanism|network)', r'\btransformers?\b (model|architecture|network)',
    r'representation learning', r'cost-sensitive learning', r'statistical learning', r'ensemble (method|model|learning|classifier)',
    r'contextual bandit', r'multi-armed bandit', r'business intelligence', r'semi-?automat', r'software agents?',
    r'personali[sz]ed (product )?recommendations?', r'k-nearest', r'\bknn\b', r'\bsvm\b', r'boosting', r'\blstm\b', r'\bbert\b',
    r'word embedding', r'word2vec', r'graph neural', r'(market|auction)[- ]algorithms?',
    r'\bdss\b', r'programmatic (advertising|buying|media|trading)', r'real-time bidding', r'\brtb\b', r'uplift model', r'link prediction', r'trading agents?',
]
# AI / machine-learning techniques applied in the study: a single mention is enough
AI_METHOD = [r'neural[- ]net', r'k-means', r'decision tree', r'random forest', r'support vector', r'gradient boost', r'xgboost',
    r'bayesian network', r'topic model', r'latent dirichlet', r'sentiment analysis', r'opinion mining', r'text[- ]mining', r'data[- ]mining',
    r'\bclustering\b', r'classifiers?\b', r'na[iï]ve bayes', r'genetic algorithm', r'\bfuzzy\b', r'particle swarm', r'evolutionary algorithm',
    r'deep[- ]learning', r'machine[- ]learning', r'natural[- ]language[- ]processing', r'\bnlp\b', r'association rules?', r'pattern mining',
    r'k-nearest', r'\bknn\b', r'\bsvm\b', r'boosting', r'\blstm\b', r'\bbert\b', r'word embedding', r'word2vec', r'ensemble (method|model|learning|classifier)',
    r'reinforcement learning', r'supervised learning', r'unsupervised', r'collaborative filtering', r'content-based filtering', r'cost-sensitive learning',
    r'representation learning', r'statistical learning', r'soft[- ]computing', r'contextual bandit', r'multi-armed bandit', r'computer[- ]vision',
    r'image recognition', r'process mining', r'graph neural', r'self-attentive', r'sequential recommendation', r'uplift model', r'link prediction']

# Strings that look like AI terms but are not (fsQCA, 'appreciative inquiry (AI)', proper names)
AI_FUZZY_EXCLUDE = [r'appreciative inquiry \(ai\)', r'\bai fini\b', r'\btable ai\b', r'dimensions\.ai', r'vihaan\.ai', r'fuzzy[- ]sets? qualitative comparative analysis', r'fuzzy[- ]set qca', r'\bfs/?qca\b', r'fuzzy[- ]set analysis']
# Search terms that are ambiguous on their own; they count only in a technical sense (covered above)
AI_WEAK_GENERIC = [r'recommend', r'predict', r'autonom']

# ---------- C2: marketing context ----------
MKT_NON_BUSINESS = [  # uses of 'marketing' outside marketing management
    r'marketing authori[sz]ation', r'post-?marketing', r'pre-?marketing', r'marketing (year|board|loan|order)s?\b',
    r'marketing margins?', r'(grain|crop|cattle|livestock|hog|milk|dairy|farm|farmers.?|agricultural|commodity) marketing',
    r'marketing (of|for) (grain|crops?|cattle|livestock|hogs|milk|commodit)',
]
MKT_DOMAIN = [
    r'customer', r'consumer', r'\bbrand', r'advertis', r'\bsales\b', r'\bselling\b', r'pric(e|es|ing)\b', r'promotion',
    r'retail', r'e-?commerce', r'segmentation', r'targeting', r'positioning', r'\bcrm\b', r'loyalty', r'churn',
    r'purchas', r'\bbuyers?\b', r'shopp', r'market research', r'market segment', r'new product', r'product development',
    r'campaign', r'personali[sz]', r'influencer', r'word[- ]of[- ]mouth', r'market share', r'clients?\b', r'touris',
    r'destination', r'hospitality', r'\bb2b\b', r'b2c', r'omnichannel', r'marketplace',
]

# single mention of marketing inside a list of sectors or functions (e.g. 'finance, marketing, and healthcare')
ENUM_PAT = r'(\b[a-z&-]+(,| and| or) marketing(,| and| or)\s|,\s*(and|or)\s+marketing\b(?!\s+(manager|decision|strateg))|\bmarketing,\s+[a-z&-]+( [a-z&-]+)?,\s)'
TITLE_CORE = r'(\bbrand|advertis|consumer|consumption behavio|\bb2b\b|business-to-business|market entry|marketplace|\bhotels?\b|online reviews?|customer|retail|pric(e|es|ing)\b|\bsales\b|e-?commerce|shopp|purchas|segmentation|\bcrm\b|loyalty|churn|recommender|influencer|touris|hospitality)'
MKT_MANAGERIAL = r'marketing (manager|decision|strateg|campaign|activit|practi|research|analytic|automation|communication|mix|performance|budget|plan|intelligence|team|department|function|professional|executive|science|model)'

# phrases that contain marketing vocabulary but are not marketing
DOM_EXCLUDE = [r'consumer price index', r'price index', r'(stock|share|oil|electricity|energy|housing|house|commodity|crude|gold|bitcoin|carbon|futures|asset|spot) prices?',
               r'demand forecasting for (electricity|energy|water)', r'(electricity|energy|water|power) demand']

# ---------- C3: decision-making context ----------
DEC_TOKEN = r'decision|decid'
DEC_TECH = [  # 'decision' only as a technical or legal term
    r'decision[- ]trees?', r'decision[- ]forests?', r'decision[- ]boundar(y|ies)', r'decision[- ]making[- ]units?', r'\bdmus?\b',
    r'decision[- ]variables?', r'(court|judicial|supreme court|tribunal|commission|ecj|cjeu|legal)\'?s? decisions?',
    r'decisions? of the (court|commission|tribunal)',
]

# choice / managerial vocabulary that signals a decision context without the word 'decision'
DEC_VOCAB = [r'\bchoices?\b', r'\bchoos', r'\bchose', r'manageri', r'\bmanagers?\b', r'strateg', r'\bplanning\b', r'allocat',
             r'optimi[sz]', r'forecast', r'segment', r'target', r'prioriti', r'\bselect(ion|ing)?\b', r'judg(e)?ment', r'\bmarketers?\b', r'practitioners?']



def norm(t):
    return re.sub(r'\s+', ' ', (t or '').lower())


def _sub_all(text, patterns):
    for p in patterns:
        text = re.sub(p, ' ', text)
    return text


def flags(title, abstract, author_keywords, index_keywords=''):
    """Screening flags for one record. index_keywords = Keywords Plus (WoS) and Index Keywords (Scopus)."""
    title, abstract = norm(title), norm(abstract)
    akw, okw = norm(author_keywords), norm(index_keywords)
    t = f'{title} . {abstract} . {akw}'
    # C1 AI
    t_ai = _sub_all(t, AI_FUZZY_EXCLUDE)
    ai_hits = sorted({p for p in AI_STRONG if re.search(p, t_ai)})
    n_ai = sum(len(re.findall(p, t_ai)) for p in AI_STRONG)
    tk_ai = _sub_all(f'{title} . {akw}', AI_FUZZY_EXCLUDE)
    ai_title_kw = any(re.search(p, tk_ai) for p in AI_STRONG)
    ai_method = any(re.search(p, t_ai) for p in AI_METHOD)
    ai = bool(ai_hits) and (ai_title_kw or n_ai >= 2 or ai_method)
    # C2 marketing
    t_nb = _sub_all(t, MKT_NON_BUSINESS)
    n_mkt = len(re.findall(r'\bmarketing\b', t_nb))
    n_mkt_raw = len(re.findall(r'\bmarketing\b', t))
    mkt_title_kw = bool(re.search(r'\bmarketing\b', _sub_all(f'{title} . {akw}', MKT_NON_BUSINESS)))
    dom = sorted({p for p in MKT_DOMAIN if re.search(p, _sub_all(t, DOM_EXCLUDE))})
    enum = bool(re.search(ENUM_PAT, t_nb))
    mgr = bool(re.search(MKT_MANAGERIAL, t_nb))
    title_core = bool(re.search(TITLE_CORE, _sub_all(title, DOM_EXCLUDE)))
    if mkt_title_kw or n_mkt >= 2 or (n_mkt >= 1 and title_core):
        mkt = True
    elif n_mkt == 1:
        mkt = (not enum) and (len(dom) >= 1 or mgr)
    else:
        mkt = (n_mkt_raw == 0) and (len(dom) >= 2 or title_core)
    # C3 decision-making
    t_dec = _sub_all(t, DEC_TECH)
    n_dec = len(re.findall(DEC_TOKEN, t_dec))
    dec_vocab = sorted({p for p in DEC_VOCAB if re.search(p, t_dec)})
    dec_idx = bool(re.search(DEC_TOKEN, _sub_all(okw, DEC_TECH)))
    dec = n_dec >= 1 or len(dec_vocab) >= 1 or dec_idx

    if not ai:
        detail = 'E1a no AI term used in a technical sense' if not ai_hits else 'E1b AI mentioned only once, as background'
    elif not mkt:
        if n_mkt == 0 and n_mkt_raw > 0:
            detail = 'E2a marketing only in a non-managerial sense'
        elif n_mkt == 0:
            detail = 'E2b no "marketing" and not enough marketing vocabulary'
        elif enum:
            detail = 'E2c marketing mentioned once in a list of sectors or functions'
        else:
            detail = 'E2d marketing mentioned once without other marketing terms'
    elif not dec:
        detail = 'E3 "decision" only as a technical or legal term'
    else:
        detail = ''
    return dict(ai=ai, mkt=mkt, dec=dec, criterion=detail[:2], detail=detail,
                ai_terms='; '.join(ai_hits[:6]), n_ai=n_ai, n_marketing=n_mkt, n_marketing_terms=len(dom), n_decision=n_dec)


def screen(df):
    """Apply the rules to a DataFrame with Title, Abstract, Keywords and (optional) IndexKeywords.

    Returns a copy with the columns screening_decision (included/excluded), screening_criterion (E1-E3),
    screening_detail and the diagnostic counts.
    """
    import pandas as pd
    idx = df['IndexKeywords'] if 'IndexKeywords' in df.columns else pd.Series('', index=df.index)
    rows = [flags(t, a, k, o) for t, a, k, o in zip(df['Title'].fillna(''), df['Abstract'].fillna(''),
                                                     df['Keywords'].fillna(''), idx.fillna(''))]
    f = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    out['screening_decision'] = f['criterion'].map(lambda c: 'excluded' if c else 'included')
    out['screening_criterion'] = f['criterion']
    out['screening_detail'] = f['detail']
    for c in ['ai_terms', 'n_ai', 'n_marketing', 'n_marketing_terms', 'n_decision']:
        out[f'screening_{c}'] = f[c]
    return out
