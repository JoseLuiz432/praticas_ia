Repositório de Práticas em Inteligência Artificial

Este repositório contém notebooks com exemplos práticos de técnicas de Inteligência Artificial (IA), abordando desde conceitos básicos até aplicações mais avançadas. O objetivo é oferecer uma coleção de exercícios e demonstrações de algoritmos de IA, aprendizado de máquina (ML), deep learning (DL), e processamento de linguagem natural (NLP).
Conteúdo

1. Fundamentos de IA

    Introdução à IA: Conceitos básicos, tipos de IA, e exemplos de aplicações.
    Algoritmos Clássicos: Implementação de algoritmos de busca e otimização, como busca em profundidade, busca em largura e algoritmos genéticos.

2. Aprendizado de Máquina

    Regressão Linear e Logística: Exemplos de modelagem preditiva com sklearn.
    K-Nearest Neighbors (KNN): Implementação de algoritmos de classificação e análise de desempenho.
    Árvores de Decisão e Random Forest: Classificação com dados estruturados.

3. Aprendizado Profundo

    Redes Neurais Artificiais: Introdução a redes neurais densas utilizando TensorFlow/Keras.
    Redes Convolucionais (CNNs): Aplicações em reconhecimento de imagens.
    Redes Recorrentes (RNNs): Implementação para análise de séries temporais e processamento de sequência de dados.
    Autoencoders e Variational Autoencoders (VAEs): Redução de dimensionalidade e geração de novos dados a partir de representações latentes.

4. Processamento de Linguagem Natural (NLP)

    Modelos de Linguagem com NLTK: Pré-processamento de texto e análise de sentimento.
    BERT e Modelos Transformer: Fine-tuning para classificação de texto.

5. Aprendizado por Reforço

    Q-Learning: Implementação de um agente de aprendizado por reforço básico.
    Deep Q-Learning: Uso de redes neurais para melhorar o desempenho de agentes.

6. IA Generativa

    Generative Adversarial Networks (GANs): Implementação de GANs para gerar imagens, incluindo GANs clássicos e variantes como DCGAN.
    Variational Autoencoders (VAEs): Aplicação para geração de dados a partir de representações latentes.
    GANs Condicionais: Controle do output gerado utilizando informações adicionais.

7. Utilizando Modelos Pré-treinados no Hugging Face

    Stable Diffusion: Geração de imagens de alta qualidade usando o modelo Stable Diffusion.
    LLaMA (Large Language Model Meta AI): Demonstração e utilização para tarefas de NLP.
    Outros Modelos: Utilizando a API do Hugging Face para carregar e aplicar modelos de visão, texto e multimodais pré-treinados.

8. Práticas Utilizando a Plataforma da OpenAI

    Básico: Introdução ao uso da API da OpenAI para processamento de linguagem natural.
    Fine-Tuning: Treinamento personalizado de modelos OpenAI como GPT-3 e GPT-4 para tarefas específicas.
    Integração com Aplicações: Utilização de modelos OpenAI em pipelines de produção e automação de tarefas.

9. Vae

10. Sagemaker exemplos
    
    Exemplos de como treinar e subir um modelo utilizando o sagemaker


## Estrutura do repositório
```
├── notebooks/
│   ├── 01_fundamentos_ia.ipynb
│   ├── 02_regressao_logistica.ipynb
│   ├── 03_redes_neurais_artificiais.ipynb
│   ├── 06_gans.ipynb
│   ├── 07_stable_diffusion.ipynb
│   ├── 08_openai_finetuning.ipynb
│   └── ... 
├── data/
│   ├── dataset1.csv
│   └── dataset2.csv
├── README.md
└── requirements.txt
```

## Requisitos
Python 3.12
Bibliotecas: Pytorch, Scikit-learn, Pandas, NumPy, Matplotlib, NLTK

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou submeter pull requests com melhorias.