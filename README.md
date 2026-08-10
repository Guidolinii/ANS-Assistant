# ANS-Assistant

Assistente de Inteligência Artificial baseado em RAG (Retrieval-Augmented Generation) para consultas a documentos oficiais e regulamentações da **Agência Nacional de Saúde Suplementar (ANS)**.

> ⚠️ **Status do Projeto**: O projeto encontra-se em fase inicial de estruturação. A lógica de RAG, integração com LLMs, banco vetorial e chamadas a APIs serão implementadas nas próximas etapas.

---

## 🎯 Objetivo

O **ANS-Assistant** tem como objetivo responder a perguntas sobre normas, resoluções normativas e regulamentações do setor de saúde suplementar no Brasil, fornecendo respostas precisas com **citação explícita das fontes oficiais** e informando com clareza quando não houver informação suficiente disponível.

---

## 📚 Contexto

Projeto desenvolvido no âmbito do programa **Oracle ONE** em parceria com a **Alura**, focado na aplicação prática de soluções de Inteligência Artificial Generativa, Engenharia de Prompt e arquitetura RAG para tratamento de dados complexos do setor regulado.

---

## ⚙️ Arquitetura Planejada

### Fluxo de Consulta (RAG)

```text
Usuário
  ↓
Streamlit (UI)
  ↓
Query Processing
  ↓
Retriever (Busca Semântica)
  ↓
Vector Store (Banco Vetorial)
  ↓
Reranker (Reordenação de Contextos)
  ↓
Context Builder
  ↓
LLM (Geração de Resposta)
  ↓
Resposta Final + Citação de Fontes
```

### Pipeline de Processamento de Documentos

```text
Documentos Oficiais ANS
  ↓
Loaders (Múltiplos formatos: PDF, DOCX, XLSX, PPTX, MD, CSV, JSON, HTML)
  ↓
Text Cleaning (Higienização e Remoção de Ruídos)
  ↓
Chunking (Divisão em fragmentos)
  ↓
Metadata Tagging (Norma, Tema, Data, Seção, etc.)
  ↓
Embedding Generation (Geração de Vetores Densos)
  ↓
Vector Store Indexing (Armazenamento e Indexação Vetorial)
```

---

## 📋 Fontes e Temas Planejados

### Base Normativa Inicial
- **RN nº 465/2021**: Rol de Procedimentos e Eventos em Saúde e atualizações;
- **RN nº 566/2022**: Garantia de Atendimento e prazos máximos;
- **RN nº 438/2018**: Portabilidade de Carências;
- **RN nº 565/2022**: Regras de Reajuste de Planos Individuais e Coletivos;
- **RN nº 557/2022**: Regulamentação das Formas de Contratação de Planos;
- **Lei nº 9.656/1998**: Dispõe sobre os planos e seguros privados de assistência à saúde;
- Documentos oficiais sobre **NIP (Notificação de Intermediação Preliminar)** e temas regulatórios correlatos.

### Temas Iniciais
- Cobertura de procedimentos e eventos em saúde
- Garantia de atendimento e prazos
- Portabilidade de carências
- Reajustes de mensalidades
- Contratação de planos coletivos e individuais
- Notificação de Intermediação Preliminar (NIP)

### Metadados por Chunk
- `document_id`: Identificador único do documento
- `document_name`: Nome oficial do documento
- `document_type`: Tipo de documento (Lei, RN, Resolução, Manual)
- `norm_number`: Número da norma
- `issuing_body`: Órgão emissor (ANS, Ministério da Saúde, etc.)
- `thematic_domain`: Domínio temático principal
- `effective_date`: Data de vigência
- `is_active`: Status de vigência (ativo/revogado)
- `source_url`: URL oficial da fonte
- `page`: Número da página (quando aplicável)
- `section`: Seção/Capítulo/Artigo

---

## 🛠️ Tecnologias Planejadas

- **Linguagem**: Python 3.12
- **Orquestração RAG**: LangChain
- **Interface Web**: Streamlit
- **Banco Vetorial**: Banco de dados vetorial a definir (ex.: FAISS, Chroma, Qdrant)
- **Modelos de Embeddings & LLM**: A definir nas etapas de integração
- **Containerização**: Docker
- **Infraestrutura / Cloud**: Oracle Cloud Infrastructure (OCI)

---

## 📁 Estrutura do Projeto

```text
ANS-Assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── cleaner.py
│   │   └── chunker.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── rag.py
│   │
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py
│
├── data/
│   ├── documents/
│   │   └── .gitkeep
│   └── metadata/
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   └── .gitkeep
│
├── scripts/
│   ├── ingest.py
│   └── rebuild_index.py
│
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🗺️ Roadmap de Desenvolvimento

- [x] **Fase 1: Estruturação Inicial do Projeto**
  - Criação da árvore de diretórios e arquivos base em Python 3.12.
  - Documentação da arquitetura e especificações iniciais.
- [ ] **Fase 2: Pipeline de Ingestão de Documentos**
  - Implementação de loaders para PDFs e outros formatos de documentos normativos da ANS.
  - Desenvolvimento dos módulos de limpeza de texto, chunking e enriquecimento de metadados.
- [ ] **Fase 3: Armazenamento Vetorial e Recuperação Semântica**
  - Integração de modelos de embeddings e configuração do banco vetorial.
  - Implementação do retriever semântico e reranking.
- [ ] **Fase 4: Pipeline RAG e Integração com LLM**
  - Construção dos templates de prompt e integração do LLM com citação rigorosa de fontes.
  - Tratamento de cenários de falta de informação.
- [ ] **Fase 5: Interface de Usuário e Testes**
  - Desenvolvimento da interface web interativa em Streamlit.
  - Criação de testes unitários e de integração.
- [ ] **Fase 6: Containerização e Implantação na OCI**
  - Configuração do `Dockerfile` e publicação na Oracle Cloud Infrastructure (OCI).
