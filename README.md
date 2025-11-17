# Estimação de Densidade por Kernel e Detecção de Outliers em Preços do Airbnb em Lisboa

Este repositório foi reduzido para um estudo focado em **estimação de densidade por kernel (KDE)** com escolha de largura de banda por **máxima verossimilhança leave-one-out (MLKDE)**. O objetivo é estimar a distribuição dos preços de acomodações do Airbnb em Lisboa e usar a densidade estimada para identificar preços improváveis (outliers).

## Estrutura do projeto
- `analise_kde_airbnb_lisboa.ipynb`: notebook principal com toda a análise, pronto para ser executado e exportado para PDF.
- `data/`: diretório para armazenar o arquivo de dados `listings_lisboa.csv` obtido no Inside Airbnb.
- `requirements.txt`: dependências mínimas para executar o notebook.

## Fonte dos dados
Use o dataset **Inside Airbnb** correspondente a Lisboa. Baixe o arquivo `listings.csv` mais recente em <http://insideairbnb.com/get-the-data/> e salve-o como `data/listings_lisboa.csv`.

## Como executar
1. Crie (e ative) um ambiente virtual se desejar.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Garanta que o arquivo `data/listings_lisboa.csv` esteja no diretório `data/`.
4. Abra o notebook:
   ```bash
   jupyter notebook analise_kde_airbnb_lisboa.ipynb
   ```
5. Execute todas as células de cima para baixo.
6. Para exportar para PDF, use a opção de exportação do Jupyter (requer `nbconvert`/LaTeX configurados no seu ambiente).

## Observação
O notebook está totalmente comentado em português e organizado para ser reproduzível, assumindo apenas a presença do arquivo de dados na pasta `data/`.
