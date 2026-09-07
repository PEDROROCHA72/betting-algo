# betting-algo — Análise de Value Bets (Futebol)

Ferramenta **MVP educativa** para analisar *value bets* no futebol, no contexto de casas como a **Betclic Portugal**.

> **Apenas análise.** Não faz login na Betclic, não faz scraping, não coloca apostas automaticamente. As odds entram por **CSV / manual**.

## Aviso legal / risco

- Apostas envolvem **risco de perda** do capital.
- Este software **não garante** lucros nem tipa apostas.
- O *Kelly* fracionário é uma **sugestão educativa**, não conselho financeiro.
- Use por sua conta e risco. Em Portugal, jogue com responsabilidade (linha de apoio do SICAD / Jogos Santa Casa quando aplicável).

## O que faz

1. Estima forças de ataque/defesa a partir de **resultados reais** (ou ratings SAMPLE).
2. Constrói uma matriz de resultados com **Poisson independente** + ajuste leve **Dixon-Coles**.
3. Deriva probabilidades de vários mercados a partir da **mesma matriz de scores**.
4. Compara com odds decimais → **edge** (`P_modelo − 1/odds`).
5. Se `edge ≥ limiar` (default **4%**), sugere stake via **Kelly fracionário** (default **0.25**).

## Mercados

Todos derivados da matriz P(casa=i, fora=j):

| Mercado | Descrição |
|---------|-----------|
| **1X2** | Casa / Empate / Fora |
| **Double chance** | 1X, 12, X2 |
| **BTTS** | Ambas marcam Yes / No |
| **Over/Under** | Linhas **1.5**, **2.5**, **3.5** |
| **Asian handicap (casa)** | −1.5, −1.0, −0.5, +0.5, +1.0, +1.5 |

### Asian handicap e push

- Linhas **meias** (−0.5, +0.5, −1.5, +1.5): sem push; cobertura = vitória no handicap.
  - AH −0.5 ≈ vitória da casa (1X2 “1”).
  - AH +0.5 ≈ casa não perde (double chance 1X).
- Linhas **inteiras** (−1.0, +1.0): se o resultado com handicap for exactamente 0, a casa **devolve a stake** (push).
  - Para value/edge/Kelly tratamos o push como **no-bet**:  
    `P_modelo = P(cover) / (1 − P(push))`  
    (excluímos outcomes de push; a stake não está em risco nesses casos).

## Ligas

Primeira Liga (`P1`), Premier League (`E0`), La Liga (`SP1`), Serie A (`I1`), Bundesliga (`D1`) — ratings em `data/team_ratings.json`.

## Instalação

```bash
cd /workspace/betting-algo
python3 -m venv .venv && source .venv/bin/activate   # recomendado (PEP 668)
pip install -e ".[dev]"
# ou: pip install -r requirements.txt && pip install -e .
```

## Calibrar ratings (dados reais)

Fonte: **[football-data.co.uk](https://www.football-data.co.uk/)** (resultados históricos em CSV). Atribuição: dados © football-data.co.uk — uso conforme os termos do site.

```bash
# Descarregar CSVs (épocas 2526 e 2425 por omissão) e regenerar data/team_ratings.json
python -m betting_algo calibrate --download

# Offline (já tem ficheiros em data/raw/{season}_{code}.csv)
python -m betting_algo calibrate --seasons 2526,2425

# Só uma época
python -m betting_algo calibrate --seasons 2526 --download
```

- CSVs brutos → `data/raw/` (gitignored; manter `.gitkeep`).
- O JSON gerado inclui `metadata.seasons`, `metadata.calibrated_at` e `metadata.source`.
- Nomes CSV mapeados para os nomes do sample: ver `data/TEAM_RENAMES.md`.

Método: média de golos da liga + forças relativas iterativas (estilo Maher), alinhadas com
`λ = attack × defense × (avg/2) × [home_adv]`.

## Comandos

```bash
# Demo com sample_odds.csv — oportunidades ranqueadas
python -m betting_algo demo

# Prever um jogo (imprime todos os mercados)
python -m betting_algo predict Benfica Porto --league primeira_liga
python -m betting_algo predict Benfica Porto --league primeira_liga \
  --odds-1 2.10 --odds-x 3.40 --odds-2 3.50 \
  --odds-over 1.80 --odds-under 2.00 \
  --odds-btts-yes 1.70 --odds-ah-m05 2.10 --odds-dc-1x 1.35

# Varrer um CSV de odds
python -m betting_algo scan data/sample_odds.csv --edge 0.04 --kelly 0.25

# Ver ratings
python -m betting_algo rate --league primeira_liga

# Calibrar (ver secção acima)
python -m betting_algo calibrate --download
```

Também: `betting-algo demo` (entry point).

Flags `predict` úteis (todas opcionais além de HOME AWAY):

`--odds-1/--odds-x/--odds-2`, `--odds-over/--odds-under` (2.5),
`--odds-over-15/--odds-under-15`, `--odds-over-35/--odds-under-35`,
`--odds-btts-yes/--odds-btts-no`,
`--odds-ah-m15/--odds-ah-m10/--odds-ah-m05/--odds-ah-p05/--odds-ah-p10/--odds-ah-p15`,
`--odds-dc-1x/--odds-dc-12/--odds-dc-x2`.

## Como colar odds da Betclic no CSV

1. Abra o jogo na Betclic (app/site) e anote as odds **decimais** dos mercados que quiser.
2. Copie o ficheiro `data/sample_odds.csv` ou crie o seu.
3. Formato — cabeçalho (colunas obrigatórias marcadas com *):

```csv
home,away,league,odds_1,odds_x,odds_2,odds_over_15,odds_under_15,odds_over_25,odds_under_25,odds_over_35,odds_under_35,odds_btts_yes,odds_btts_no,odds_ah_m15,odds_ah_m10,odds_ah_m05,odds_ah_p05,odds_ah_p10,odds_ah_p15,odds_dc_1x,odds_dc_12,odds_dc_x2
Benfica,Porto,primeira_liga,2.10,3.40,3.50,1.30,3.40,1.80,2.00,2.80,1.42,1.70,2.15,4.80,3.60,2.10,1.40,1.18,1.12,1.35,1.28,1.55
```

| Coluna | Obrigatória | Notas |
|--------|-------------|--------|
| `home` / `away` | sim | Nome igual ao de `team_ratings.json` |
| `league` | sim | `primeira_liga`, `premier_league`, `la_liga`, `serie_a`, `bundesliga` |
| `odds_1` / `odds_x` / `odds_2` | sim | Odds decimais 1X2 |
| `odds_over_15` / `odds_under_15` | não | Over/Under 1.5 |
| `odds_over_25` / `odds_under_25` | não | Over/Under 2.5 |
| `odds_over_35` / `odds_under_35` | não | Over/Under 3.5 |
| `odds_btts_yes` / `odds_btts_no` | não | Ambas marcam |
| `odds_ah_m15` / `odds_ah_m10` / `odds_ah_m05` | não | AH casa −1.5 / −1.0 / −0.5 |
| `odds_ah_p05` / `odds_ah_p10` / `odds_ah_p15` | não | AH casa +0.5 / +1.0 / +1.5 |
| `odds_dc_1x` / `odds_dc_12` / `odds_dc_x2` | não | Double chance |

Células vazias = mercado ignorado nesse jogo. Colunas em falta no ficheiro também são OK (só 1X2 é obrigatório).

4. Corra: `python -m betting_algo scan o_seu_ficheiro.csv`

**Não** é necessário login nem export automático — copie as odds à mão.

## Modelo (resumo)

- λ casa = `attack_casa × defense_fora × (média_golos_liga / 2) × vantagem_casa`
- λ fora = `attack_fora × defense_casa × (média_golos_liga / 2)`
- Matriz Poisson até 8 golos; τ Dixon-Coles (`ρ ≈ −0.05`) nos resultados baixos.
- **Value** se `P_modelo − 1/odds ≥ edge` (default 0.04).
- Kelly: `f* = (b·p − q) / b`, stake sugerido = `0.25 · f*` (educacional).
- AH inteiro: `P_modelo` = cobertura condicionada a não-push (ver acima).

## Testes

```bash
cd /workspace/betting-algo
pytest -q
```

## Estrutura

```
betting-algo/
  betting_algo/     # pacote CLI + modelo + calibrate + markets
  data/             # ratings JSON + sample_odds.csv
    raw/            # CSVs football-data (gitignore)
  tests/            # pytest
  pyproject.toml
  requirements.txt
  README.md
```

## Dados / atribuição

Resultados históricos descarregados de [football-data.co.uk](https://www.football-data.co.uk/).
Este projeto não é afiliado; os CSVs são redistribuídos apenas localmente em `data/raw/` para calibração offline.

## Licença

MIT — uso educativo.
