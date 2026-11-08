# Log de coleta — SLR "AI in the Marketing Decision-Making"

**Data da coleta:** 05/11/2026 (quinta-feira), fuso America/Sao_Paulo (UTC-03:00)
**Acesso:** Portal de Periódicos CAPES (proxy ez49.periodicos.capes.gov.br), login CAFe
**Coletor:** Claude in Chrome, sob solicitação de Silvio (silviosmf@gmail.com)

> **Atualização (06/11/2026):** a seção 1 (Web of Science, 6.253 registros) foi substituída por uma nova exportação do WoS de 06/11/2026 com 1.699 registros (arquivo `data/articles/wos/WoS_1699_registros_completos.tsv`; "marketing" entre aspas para desligar a lematização). A seção 2 (Scopus, 2.475 registros) continua válida. A string da nova consulta ao WoS ainda deve ser registrada neste log.

## 1. Web of Science Core Collection

- Edições: All (Core Collection, todos os índices disponíveis na assinatura CAPES) — **confirmar a lista de índices na interface antes de reportar no artigo**.
- Primeira exportação: 2026-11-05 15:55:33 (-03:00); lote 2: 15:56:54; lote 3: 15:58:55. Lotes 4–7: entre 16:00 e 16:04 (horário aproximado, registrado a partir do carregamento dos arquivos).
- Formato: Tab-delimited (Win, UTF-8), Record Content = **Full Record** (inclui TI, AU, AF, PY, AB, DE, ID, DI, TC, Z9, UT etc.; 71 colunas), máx. 1.000 registros por exportação.

### String final (V2)
```
TS=( marketing AND decision AND ( "AI" OR "artificial intelligence" OR automation OR autonomous OR "big data" OR chatbot OR "computer vision" OR "data mining" OR "data science" OR "fuzzy logic" OR "genetic algorithm" OR "Internet of Things" OR iot OR "k-means" OR "machine learning" OR "natural language processing" OR "neural network" OR prediction OR recommendation OR robot OR "smart technologies" OR "soft computing" OR "text mining" ) ) AND DT=( Article ) AND SU=( "Business Economics" OR "Social Sciences Other Topics" ) 
```
**Resultado: 6.253 registros.** A string não tem restrição de período (a variante anterior com `PY=( 1973-2026 )` retornou o mesmo total, conforme conferido pelo autor). Anos dos registros exportados: 1990–2027 (13 com ano 2027).

### Variantes testadas (05/11/2026)
| Variante | Scopus | WoS |
|---|---|---|
| V0 — núcleo + período | 11.095 | 23.372 |
| V1 — + tipo "article" | 5.247 | 17.359 |
| V2 — + áreas (string da Tabela A2, sem filtro de período) | 2.475 (2.468 com PUBYEAR 1973–2026) | 6.253 |
| V3 — WoS apenas TI/AB/AK (sem Keywords Plus) | — | 4.960 |

### Arquivos
| Arquivo | Registros (ordem de relevância) |
|---|---|
| WoS_raw/WoS_V2_lote_0001-1000.txt | 1–1.000 |
| WoS_raw/WoS_V2_lote_1001-2000.txt | 1.001–2.000 |
| WoS_raw/WoS_V2_lote_2001-3000.txt | 2.001–3.000 |
| WoS_raw/WoS_V2_lote_3001-4000.txt | 3.001–4.000 |
| WoS_raw/WoS_V2_lote_4001-5000.txt | 4.001–5.000 |
| WoS_raw/WoS_V2_lote_5001-6000.txt | 5.001–6.000 |
| WoS_raw/WoS_V2_lote_6001-6253.txt | 6.001–6.253 |
| WoS_V2_6253_registros_completos.tsv | união dos 7 lotes |

### Validação
- 7 lotes: 1.000 × 6 + 253 = 6.253 linhas; UT únicos = 6.253; sem sobreposição entre lotes.
- Um lote (3.001–4.000) foi exportado duas vezes por engano; a cópia duplicada foi descartada (idêntica, mesmo hash).

## 2. Scopus

- Acesso: Portal CAPES (CAFe), Advanced search, 05/11/2026. Exportação final feita pelo autor em 05–06/11/2026 (arquivo com carimbo 2026-11-06T00:45:56 UTC, ou 05/11/2026 21:45 -03:00).
- String final (V2, **sem restrição de período**):
```
TITLE-ABS-KEY ( marketing AND decision AND ( "AI" OR "artificial intelligence" OR automation OR autonomous OR "big data" OR chatbot OR "computer vision" OR "data mining" OR "data science" OR "fuzzy logic" OR "genetic algorithm" OR "Internet of Things" OR iot OR "k-means" OR "machine learning" OR "natural language processing" OR "neural network" OR prediction OR recommendation OR robot OR "smart technologies" OR "soft computing" OR "text mining" ) ) AND DOCTYPE ( ar ) AND SUBJAREA ( busi OR soci )
```
- **Resultado: 2.475 documentos** (a variante com `PUBYEAR > 1972 AND PUBYEAR < 2027` retornou 2.468; os 7 a mais são de 2027).
- Formato: CSV, 1 único lote (limite da Scopus: 20.000), 33 colunas: autores, título, ano, fonte, citações (Cited by), DOI, afiliações, resumo, palavras-chave do autor e indexadas, tipo, EID etc. (sem referências).
- Arquivo: `Scopus_V2_2475_registros_completos.csv` (original: `export_bf156bd3-8695-4727-9e49-6e7bc588f684_2026-11-06T004556.909886749.csv`).
- Validação: 2.475 linhas; EID únicos = 2.475; todos "Article"; anos 1973–2027 (2027: 7; 2026: 411); resumo em 2.475; DOI em 2.340; palavras-chave do autor em 2.297.

## 3. Observações
- O corpus (WoS 6.253; Scopus 2.475) é muito maior que os 2.174 registros do protocolo/simulação; a Tabela A4 (PRISMA) e o Resumo precisam ser refeitos com os números reais após deduplicação e triagem.
- Strings sem filtro de período (decisão do autor em 05/11/2026): o intervalo de anos real deve ser informado nos Resultados (Scopus 1973–2027; WoS 1990–2027).
- No WoS, DT=Article inclui também registros marcados como Proceedings Paper/Book Chapter junto de Article; checar o campo DT/PT na triagem.
- Há registros com PY 2027 (early access/publicação antecipada); o filtro final pode excluí-los.
- Keywords Plus responde por ~1.293 registros (V2 6.253 vs. V3 4.960).
