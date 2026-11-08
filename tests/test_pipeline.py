"""Checks for the screening rules and the batch format of the classification (python tests/test_pipeline.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdm import prompt as cp  # noqa: E402
from mdm import screening  # noqa: E402


def test_screening_rules():
    f = screening.flags
    keep = f('Machine learning for customer churn prediction', 'We use random forests to predict churn and support '
             'retention decisions of marketing managers in telecom.', 'churn; machine learning')
    assert keep['criterion'] == '', keep
    no_ai = f('Word-of-mouth and destination image', 'We offer recommendations for destination marketing managers '
              'and predict that word of mouth matters for decision makers.', 'tourism')
    assert no_ai['criterion'] == 'E1', no_ai
    fsqca = f('Configurations of trust', 'Using fuzzy-set qualitative comparative analysis (fsQCA), we explain purchase '
              'decisions in e-commerce marketing.', 'fsQCA')
    assert fsqca['criterion'] == 'E1', fsqca
    pharma = f('Deep learning for drug safety', 'A neural network model supports regulatory decisions after the '
               'marketing authorization of medicines.', 'deep learning')
    assert pharma['detail'].startswith('E2a'), pharma
    listed = f('AI in finance', 'Machine learning supports decisions in finance, marketing, and healthcare.', 'AI')
    assert listed['detail'].startswith('E2c'), listed
    tree = f('Decision trees for bank marketing data', 'We compare decision tree classifiers on a bank marketing '
             'dataset and report accuracy for the marketing campaign data.', 'decision tree; classification')
    assert tree['criterion'] == 'E3', tree


def test_batch_format_and_parser():
    msg = cp.format_batch([('S1', 'T1', 'k', 'We set dynamic prices online.'), ('S2', 'T2', None, 'A chatbot talks to customers.')])
    assert '<study id="S1">' in msg and '<study id="S2">' in msg and 'ids: S1, S2' in msg
    resp = {'results': [
        {'id': 'S2', 'evidence': 'A chatbot talks to customers', 'ai_techniques': ['chatbot'], 'rationale': 'R2',
         'stage': 'Marketing action', 'ai_intelligence': 'Feeling AI', 'marketing_mix': 'Product/Consumer',
         'secondary_stage': None, 'secondary_ai_intelligence': None, 'confidence': 'high'},
        {'id': 'S1', 'evidence': 'dynamic prices', 'ai_techniques': [], 'rationale': 'R3',
         'stage': 'Marketing action', 'ai_intelligence': 'Thinking AI', 'marketing_mix': 'Price/Cost',
         'secondary_stage': None, 'secondary_ai_intelligence': None, 'confidence': 'medium'}]}
    parsed, problem = cp.parse_batch_response(json.dumps(resp), [('S1', 'We set dynamic prices online.'),
                                                                 ('S2', 'A chatbot talks to customers.')])
    assert problem is None and set(parsed) == {'S1', 'S2'}
    assert parsed['S1']['strategic_element'] == 'Personalization' and parsed['S2']['strategic_element'] == 'Relationalization'
    assert parsed['S1']['evidence_in_abstract'] is True
    # a study left out of the answer is reported missing, so it can be sent again
    resp['results'] = resp['results'][:1]
    parsed, _ = cp.parse_batch_response(json.dumps(resp), [('S1', 'x'), ('S2', 'y')])
    assert set(parsed) == {'S2'}
    # truncated answer
    parsed, problem = cp.parse_batch_response('{"results": [{"id": "S1", "stage"', [('S1', 'x')])
    assert parsed == {} and problem.startswith('invalid JSON')


def test_schema_labels():
    item = cp.RESPONSE_SCHEMA['properties']['results']['items']
    assert set(item['properties']['stage']['enum']) == set(cp.STAGES + [cp.NOT_IDENTIFIED])
    assert set(item['required']) <= set(item['properties'])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
