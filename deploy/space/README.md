---
title: Vietnamese Hybrid Search for RAG
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: BM25 vs dense vs hybrid retrieval on Vietnamese law
---

# Vietnamese Hybrid Search for RAG

Search 61,425 articles of Vietnamese legal documents (Zalo AI 2021 Legal Text Retrieval) with four retrieval configurations side by side:

- **R0:** BM25 on syllables
- **R1:** BM25 on pyvi-segmented words
- **R2:** dense retrieval with bge-m3
- **R3:** hybrid Reciprocal Rank Fusion of R1 and R2

You can then generate an answer from R3's top-5 with Claude Haiku, with citations to the exact articles.

Code, experiments and results: https://github.com/Phvster/vn-hybrid-search

*The first search after the Space wakes up takes about a minute while the models load. Answers are for reference only, based on 2021 legal texts, and are not legal advice.*
