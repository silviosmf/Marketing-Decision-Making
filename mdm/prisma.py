"""PRISMA 2020 flow diagram (identification via databases only), grayscale and Arial, from the counts of the pipeline."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from mdm import style


def counts_from_files(flow_csv='data/articles/collection_flow.csv', log_csv='data/articles/screening_log.csv'):
    import pandas as pd
    flow = pd.read_csv(flow_csv).set_index('step')['records']
    log = pd.read_csv(log_csv, dtype=str, keep_default_na=False)
    exc = log[log['screening_decision'] == 'excluded']['screening_criterion'].value_counts()
    return {
        'scopus': int(flow['Scopus records exported']), 'wos': int(flow['WoS records exported']),
        'identified': int(flow['Records identified (Scopus + WoS)']),
        'duplicates': int(flow['Records identified (Scopus + WoS)'] - flow['Unique records']),
        'no_abstract': int(flow['Removed: without abstract']), 'retracted': int(flow['Removed: retracted publication']),
        'screened': len(log), 'excluded': int(exc.sum()),
        'E1': int(exc.get('E1', 0)), 'E2': int(exc.get('E2', 0)), 'E3': int(exc.get('E3', 0)),
        'included': int((log['screening_decision'] == 'included').sum()),
    }


def figure(c):
    style.use()
    f = lambda n: f'{n:,}'
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(x, y, w, h, text):
        ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor='black', linewidth=0.9))
        ax.text(x + 1.6, y + h - 1.8, text, ha='left', va='top', fontsize=8.6, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color='black', lw=0.9, mutation_scale=10))

    def band(y, h, label):
        ax.add_patch(FancyBboxPatch((0.5, y), 6.5, h, boxstyle='round,pad=0,rounding_size=1.2', facecolor='#d9d9d9',
                                    edgecolor='black', linewidth=0.8))
        ax.text(3.75, y + h / 2, label, rotation=90, ha='center', va='center', fontsize=9.5, fontweight='bold')

    ax.add_patch(Rectangle((9, 94), 89, 5, facecolor='#d9d9d9', edgecolor='black', linewidth=0.8))
    ax.text(53.5, 96.5, 'Identification of studies via databases', ha='center', va='center', fontsize=9.5, fontweight='bold')
    band(71, 21, 'Identification')
    box(9, 74, 40, 17, f'Records identified from databases\n(n = {f(c["identified"])}):\n   Scopus (n = {f(c["scopus"])})\n'
                       f'   Web of Science Core Collection\n   (n = {f(c["wos"])})')
    box(57, 74, 41, 17, f'Records removed before screening:\n   Duplicate records (n = {f(c["duplicates"])})\n'
                        f'   Records without abstract (n = {f(c["no_abstract"])})\n   Retracted records (n = {f(c["retracted"])})')
    arrow(49, 82.5, 57, 82.5)
    band(37, 31, 'Screening')
    box(9, 53, 40, 12, f'Records screened\n(title, abstract and author keywords)\n(n = {f(c["screened"])})')
    box(57, 41, 41, 24, f'Records excluded (n = {f(c["excluded"])}):\n   E1 AI absent or only mentioned\n   in passing (n = {f(c["E1"])})\n'
                        f'   E2 Marketing absent or only\n   mentioned in passing (n = {f(c["E2"])})\n'
                        f'   E3 No decision-making context\n   (n = {f(c["E3"])})')
    arrow(29, 74, 29, 65); arrow(49, 59, 57, 59)
    band(17, 17, 'Included')
    box(9, 21, 40, 9, f'Studies included in the review\n(n = {f(c["included"])})')
    arrow(29, 53, 29, 30)
    ax.text(9, 11, 'Eligibility was assessed on titles, abstracts and author keywords with rule-based screening;\n'
                   'full texts were not retrieved. Screening rules are described in Section 3.2.',
            fontsize=7.4, va='center', color='#333333')
    return fig
