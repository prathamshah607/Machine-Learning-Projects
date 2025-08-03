# Contextual Semantics Across Languages: Understanding Word Meaning in Multilingual Embedding Spaces  
**Cryptonite NLP Research Taskphase**  
**Pratham Shah**  
**ID: 240905614**

<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/156f6c87-ab5f-4a4e-bb97-8b590ff38248" />


---

## Motivation

<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/156f6c87-ab5f-4a4e-bb97-8b590ff38248" />


Word meaning is not fixed. It varies with context, speaker intent and cultural background. Contextual embedding models like BERT and its multilingual variants have improved the way language models find meaning within a given sentence. However, how these meanings shift across languages and cultures is still not fully understood.

This project focuses on semantic variation across languages, especially in multilingual models. It aims to study how certain words, particularly polysemous or culturally significant ones, change their meaning depending on language and context. Understanding these shifts can help reveal how well multilingual models capture real-world language use, and where they fall short. It can also help humankind better understand our history and surroundings through language.

---

## Related Work

Contextual word embeddings model how meaning is influenced by context. Models like BERT, RoBERTa, and their multilingual counterparts (eg XLM-R, mBERT) can adjust word representations based on surrounding words. Multilingual models aim to map semantically similar words across languages to similar regions in embedding space.

Some foundational papers for this include:

1. **Devlin et al. (2019)** – *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*  
   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)  
   Introduced BERT, showing how deep contextual embeddings can be learned through unsupervised pretraining.

2. **Conneau et al. (2020)** – *Unsupervised Cross-lingual Representation Learning at Scale*  
   [https://arxiv.org/abs/1911.02116](https://arxiv.org/abs/1911.02116)  
   Proposed XLM-R, a robust multilingual transformer model trained on 100+ languages.

3. **Chakravarti & Gonen (2023)** – *Understanding Cross-Lingual Semantic Shift with Contextualized Word Representations*  
   [https://arxiv.org/abs/2302.04127](https://arxiv.org/abs/2302.04127)  
   Examines how multilingual models capture meaning shifts and alignment challenges in cross-lingual settings.

Despite such advances, there remains limited analysis of cultural and contextual shifts in meaning, especially for abstract or socially complex terms.

---

## Research Gap

<img width="585" height="421" alt="image" src="https://github.com/user-attachments/assets/62c321d5-d1c1-4cd6-baf1-ea3d89d4de00" />

Current multilingual models rely on the assumption that words with similar translations have equivalent meanings across languages. This breaks down in practice due to:

- **Polysemy**: Words with multiple senses depending on context (eg. court in sports vs. law).
- **Cultural nuance**: Words that carry different associations in different societies (eg. *freedom*, *karma*, *family*).
- **Contextual variation**: Words whose meaning changes depending on syntax, register or domain.

There is a lack of research on how such words behave in multilingual embedding spaces. Most evaluations don’t test how these embeddings respond to cultural or contextual shifts.

---

## Research Direction

### Goal

To analyze how the meanings of words vary across different contexts and languages using multilingual contextual embeddings.

### Objectives

1. Measure how specific words shift in meaning across contexts and languages.
2. Compare embeddings of culturally sensitive terms across multiple languages.
3. Visualize these semantic shifts using projection and clustering methods.
4. Identify patterns of divergence for emotion terms, social roles, and abstract values.

---

## Methodology

### Languages and Data

- **Languages**: English, Hindi, French, Spanish.
- **Sources**:
  - Wikipedia (formal / encyclopedic)
  - OpenSubtitles (informal / dialog-based)
  - Domain-specific corpora (cultural, social or religious contexts)

### Embedding Models

- **XLM-R**: A multilingual transformer model trained on 100+ languages.
- **LaBSE**: A sentence embedding model designed for language-agnostic semantic similarity.

**Implementation**: HuggingFace Transformers for tokenization and embedding extraction.

### Analysis Techniques

- **Nearest Neighbor Analysis**: Track semantically closest words in embedding space.
- **Dimensionality Reduction**: Use PCA + UMAP/t-SNE for 2D or 3D visualization of embeddings.
- **Semantic Axes Projection**: Create abstract dimensions (eg. sentiment, formality) and project embeddings accordingly.
- **Cross-Lingual Anchoring**: Align embeddings of translated terms across languages to detect divergence.

### Evaluation Metrics

- **Cosine Similarity**: Measure embedding closeness.
- **Jaccard Similarity**: Assess overlap in top-k nearest neighbors.
- **Clustering Entropy**: Evaluate how distinctly senses are separated in context.
- **Qualitative Inspection**: Manually analyze selected examples for interpretability and cultural fidelity.

---

## Timeline

| Week | Task Description                             |
|------|-----------------------------------------------|
| 1–2  | Literature review, finalize datasets          |
| 3–4  | Embedding extraction and preprocessing        |
| 5    | Initial visualizations and semantic tracking  |
| 6    | Cross-lingual comparison and analysis         |
| 7    | Documentation and result synthesis            |
| 8    | Final report writing and project submission   |

---

## Potential Impact

<img width="550" height="293" alt="image" src="https://github.com/user-attachments/assets/e0b4e2f3-74b4-49c1-800c-04dbafc344a5" />

This research aims to improve understanding of how multilingual language models capture meaning variation across contexts and cultures. It could contribute to:

- **Better machine translation**: Especially for ambiguous or culturally nuanced terms.
- **More accurate multilingual chatbots and assistants**: By grounding responses in context.
- **Cross-lingual search engines**: With deeper semantic awareness.
- **Language learning tools**: That reflect real-world usage and meaning shifts.

By identifying limitations and patterns in current models, the project will help inform the design of future language technologies that are more context-aware, culturally adaptive, and robust. The analysis of these embeddings, derived from history and culture, will also draw appreciation for how each nation understands words and their emotions differently, and how our context affects our language.

