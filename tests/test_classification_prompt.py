"""Checks for the classification prompt, mdm/prompt.py (run from the repository root: python tests/test_classification_prompt.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdm import prompt as cp  # noqa: E402

ex = cp.load_examples()
prompt = cp.build_system_prompt(ex)


def test_examples_file():
    worked = ex[ex['Prompt role'] == 'worked_example']
    assert len(worked) == 13
    # every strategic element has a worked example
    cells = set(zip(worked['Strategic decision'], worked['AI intelligence']))
    assert cells == set(cp.ELEMENTS), cells ^ set(cp.ELEMENTS)
    # labels are consistent with Table 1 of Huang and Rust (2021)
    for _, r in ex.iterrows():
        assert cp.ELEMENTS[(r['Strategic decision'], r['AI intelligence'])] == r['Operation'], r['ID']
        assert (r['Strategic decision'] == cp.ACTION) == isinstance(r['Marketing mix'], str), r['ID']
    # evidence of each worked example is an exact quotation of its abstract
    for _, r in worked.iterrows():
        assert cp._norm(r['Evidence']) in cp._norm(r['Abstract']), r['ID']
        coding = cp.example_coding(r)
        back = cp.parse_response(json.dumps(coding), abstract=r['Abstract'])
        assert back['parse_errors'] is None, (r['ID'], back['parse_errors'])
        assert back['strategic_element'] == r['Operation']


def test_prompt():
    for text in ['Step 1. Marketing stage', 'Step 2. AI intelligence', 'R1.', 'R7.', 'Worked examples', '"stage"']:
        assert text in prompt, text
    # relevance was decided by the screening; the classifier must not judge scope again
    for text in ['in_scope', 'exclusion_criterion', 'E1.', 'E2.', 'E3.', 'out of scope', 'Out of Scope']:
        assert text not in prompt, text
    assert 'do not judge their relevance' in prompt
    assert prompt.count('### Example ') == 13
    assert 'Amazon Go' not in prompt          # row excluded from the prompt (conflicts with rule R1)


def test_parser():
    ok = cp.parse_response('```json\n{"stage": "Marketing action (4Ps/4Cs)", "ai_intelligence": "thinking",'
                           ' "marketing_mix": "Price/Cost", "secondary_stage": null, "secondary_ai_intelligence": null,'
                           ' "ai_techniques": ["bandit"], "evidence": "dynamic prices", "rationale": "x", "confidence": "High"}\n```',
                           abstract='We set Dynamic  prices online.')
    assert ok['parse_errors'] is None, ok['parse_errors']
    assert (ok['stage'], ok['ai_intelligence'], ok['strategic_element']) == (cp.ACTION, cp.THINKING, 'Personalization')
    assert ok['evidence_in_abstract'] is True and ok['confidence'] == 'high' and ok['classification'] == cp.THINKING
    assert 'in_scope' not in ok and 'exclusion_criterion' not in ok

    unk = cp.parse_response('{"stage": "Not identified", "ai_intelligence": "Feeling AI", "confidence": "low"}')
    assert unk['stage'] == cp.NOT_IDENTIFIED and unk['strategic_element'] == cp.NOT_IDENTIFIED
    assert unk['classification'] == cp.FEELING and unk['parse_errors'] is None

    amb = cp.parse_response('{"stage": "marketing", "ai_intelligence": "AI", "confidence": "low"}')
    assert amb['stage'] == cp.NOT_IDENTIFIED and amb['ai_intelligence'] == cp.NOT_IDENTIFIED
    assert 'unexpected value' in amb['parse_errors']

    miss = cp.parse_response('{"stage": "Marketing action", "ai_intelligence": "Mechanical AI",'
                             ' "secondary_stage": "Marketing research", "confidence": "medium"}')
    assert 'marketing_mix missing' in miss['parse_errors'] and 'together' in miss['parse_errors']
    assert miss['secondary_stage'] is None

    bad = cp.parse_response('Sorry, I cannot help.')
    assert bad['parse_errors'].startswith('invalid JSON')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
