# Manual do Projeto EVO-STACK-KDE

## Visão geral
O EVO-STACK-KDE é um pipeline de pesquisa para otimização evolutiva de modelos de densidade baseada em Kernel Density Estimation (KDE). O projeto combina uma etapa de preparação de dados, um algoritmo evolutivo multiobjetivo (NSGA-II) e uma fase de ajuste fino/avaliação para selecionar um ensemble de especialistas KDE que equilibra três objetivos: log-verossimilhança negativa, estabilidade estatística e complexidade do modelo.

## Preparação do ambiente
1. **Criar ambiente virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   ```
2. **Instalar dependências**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Dados de entrada
O dataset principal esperado pelo pipeline é o arquivo `data/raw/winequality-red.csv`, correspondente ao conjunto "Wine Quality" (vinhos tintos) do repositório UCI. Garanta que o arquivo esteja presente nesse caminho antes de iniciar a execução.

## Fluxo de execução
O entrypoint oficial é o módulo `src.runner`. Após ativar o ambiente virtual e confirmar a presença do dataset, execute:
```bash
python -m src.runner --csv data/raw/winequality-red.csv --outdir outputs --seed 42
```
Outros parâmetros úteis:
- `--pop` (tamanho da população NSGA-II, padrão 20)
- `--gens` (número de gerações, padrão 10)
- `--kfold` (dobras para validação cruzada, padrão 2)
- `--bootstraps` (amostragens bootstrap para estabilidade, padrão 3)

O comando realiza as etapas de carregamento e escalonamento dos dados, otimização evolutiva, seleção da solução de "joelho", ajuste final do ensemble, geração de métricas, artefatos gráficos e salvamento do modelo.

## Estrutura dos módulos em `src/`
- `data.py`: funções para carregar o CSV de vinhos, dividir os dados em treino/validação/teste com normalização e persistir splits processados.
- `kde_expert.py`: define o especialista KDE individual, incluindo a regra de banda de Scott e aplicação opcional de máscaras de features.
- `ensemble.py`: implementa o ensemble de especialistas KDE, combinando probabilidades via softmax e log-sum-exp.
- `eval.py`: métricas de avaliação (NLL via k-fold, estabilidade bootstrap) e cálculo da penalidade de complexidade das configurações.
- `genome.py`: representação genética do modelo, utilitários para gerar configurações aleatórias, serializar/ desserializar e decodificar para objetos treináveis.
- `nsga.py`: orquestra o algoritmo evolutivo NSGA-II usando DEAP, incluindo inicialização, operadores de cruzamento/mutação, avaliação e armazenamento da frente de Pareto.
- `plots.py`: geração de figuras com a frente de Pareto e histogramas da log-verossimilhança no teste.
- `runner.py`: script principal que conecta todas as etapas: parsing de argumentos, preparação de dados, execução NSGA-II, seleção do joelho, treinamento final, salvamento de métricas, modelos e gráficos.

## Uso de modelo treinado
Após a primeira execução completa do pipeline, um ensemble final treinado é armazenado em `outputs/final_model.pkl`, contendo o modelo, o scaler utilizado no pré-processamento e metadados de treinamento.

1. **Treinar e salvar o modelo**
   ```bash
   python -m src.runner --csv data/raw/winequality-red.csv --outdir outputs --seed 42
   ```
   O comando acima salva o modelo final em `outputs/final_model.pkl` além de demais artefatos.

2. **Reutilizar modelo salvo para novos dados**
   ```bash
   python -m src.runner --load_model outputs/final_model.pkl --csv data/raw/novos_dados.csv
   ```
   Esse modo carrega o ensemble previamente treinado, aplica o mesmo pré-processamento aos novos dados e gera predições/relatórios sem executar novamente o processo evolutivo.
