# SME0300 — Projeto #9: Mecanismos Articulados (Movimento do Ciclista)

Modelagem cinemática plana de quatro barras aplicada ao movimento da perna de um ciclista, resolvida via método de Newton-Raphson (SME0300 Cálculo Numérico — 2º semestre de 2026, ICMC-USP).

## Parâmetros do Modelo
* **Comprimentos dos elos:** $a = 0{,}17\text{ m}$ (pedivela), $b = 0{,}45\text{ m}$ (perna), $c = 0{,}45\text{ m}$ (coxa), $d = 0{,}55\text{ m}$ (quadril-eixo, fixo)
* **Unidade angular:** radianos
* **Critério de parada (Newton-Raphson):** $\max\{\vert{}\Delta\theta_B^{(k)}\vert{}, \vert{}\Delta\theta_D^{(k)}\vert{}\} < 10^{-8}$
* **Limite de iterações:** 50
* **Discretização do ciclo completo:** $\ge 360$ pontos em $[0, 2\pi]$

## Estrutura do Repositório
* `nucleo/`: equações de fechamento, matriz Jacobiana analítica, solver de Newton-Raphson e reconstrução geométrica.
* `experimentos/`: scripts de execução para cada item do roteiro (análise pontual, múltiplas soluções, varredura de ciclo, singularidade e critério biomecânico do joelho).
* `saidas/`: diretório local para gráficos e tabelas temporárias.
* `relatorio/`: documento técnico final e figuras consolidadas.

## Configuração do Ambiente (Windows / Linux)
1. Clone o repositório e entre na pasta do projeto.
2. Crie e ative um ambiente virtual Python:
~~~bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
~~~
3. Instale as dependências:
~~~bash
pip install -r requirements.txt
~~~
