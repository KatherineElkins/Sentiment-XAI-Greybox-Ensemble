# SentimentArcs-Greybox: XAI Ensemble for Diachronic Sentiment Analysis

## Novel Framework Combining GPT-4 with Explainable AI

**Authors**: Jon Chun & Katherine L. Elkins, Ph.D.  
**Institution**: Kenyon College  
**Published**: *International Journal of Digital Humanities* (2023)

This repository contains the implementation of a novel greybox ensemble framework that combines state-of-the-art LLMs (GPT-4) with explainable whitebox models for diachronic sentiment analysis of narrative texts.

---

## Overview

**The Problem**: Traditional sentiment analysis faces a trade-off:
- **Blackbox models** (GPT-4, BERT) = High performance but lack explainability
- **Whitebox models** (VADER, lexical) = Transparent but lower performance

**Our Solution**: A greybox ensemble that:
- Uses GPT-4 as the performance benchmark
- Identifies which whitebox models best align with GPT-4 for specific texts
- Provides both high performance AND explainability
- Enables human-in-the-loop supervision at local and global levels

---

## Key Innovations

### 1. **Novel XAI Metrics**

**Ensemble Curve Coherence (ECC)** - Global metric
- Measures alignment between whitebox and blackbox models across entire narrative
- Enables identification of best explainable model for a given text
- Uses Savitzky-Golay smoothing + Euclidean distance between curves

**Ensemble Point Coherence (EPC)** - Local metric  
- Identifies specific sentences where models disagree
- Flags ambiguous or complex emotional moments for human review
- Measures Euclidean distance between max/min sentiment at each point

### 2. **GPT-4 Function Calling Integration**

First published use of GPT-4's function calling API (version 0613) for sentiment analysis:
- Structured JSON interface for consistent sentiment classification
- Returns both polarity (positive/negative/neutral) and emotion type
- Dramatically reduces malformed responses (<0.03% error rate)

### 3. **Human-in-the-Loop Workflow**

- Supervisory oversight rather than full automation
- Comparative model analysis surfaces points requiring human judgment
- Balances efficiency with accuracy and interpretability

---

## Use Cases

This framework is designed for:

✅ **Literary narrative analysis** - novels, short stories, scripts  
✅ **Social media discourse** - tracking opinion evolution over time  
✅ **Financial sentiment** - news articles, market commentary  
✅ **Medical narratives** - patient stories, case studies  
✅ **Policy documents** - analyzing tone shifts in regulations

**Not recommended for**:
- Single short texts (tweets, reviews) - use standard sentiment analysis
- Heavily ironic or sarcastic texts - requires additional preprocessing

---

## Repository Contents

```
sentimentarcs-greybox/
├── notebooks/
│   └── greybox_ensemble_analysis.ipynb    # Main Jupyter notebook
├── data/
│   ├── woolf_to_the_lighthouse.txt        # Example text 1 (coherent)
│   └── morrison_beloved.txt               # Example text 2 (incoherent)
├── models/
│   └── model_configs.json                 # Ensemble model specifications
├── visualizations/
│   ├── ecc_heatmap.png                    # Model alignment visualization
│   └── sentiment_curves.png               # Ensemble sentiment plots
├── docs/
│   ├── methodology.md                     # Detailed methodology
│   └── paper.pdf                          # Published paper
└── README.md
```

---

## Methodology

### Ensemble Models

The framework includes 6 models across 3 families:

**Whitebox (Explainable):**
- VADER - lexical + heuristic rules
- TextBlob - lexical sentiment

**Greybox (BERT Transformers):**
- DistilBERT - 66M parameters
- NLPTown - 110M-340M parameters  
- RoBERTa Large - 355M parameters

**Blackbox (LLMs):**
- GPT-3.5-turbo-0613
- GPT-4-0613 (1.76T parameters)

### Workflow

1. **Text Segmentation** → Split narrative into sentences (optimal semantic unit)
2. **Sentiment Classification** → Each model scores every sentence
3. **Normalization** → Convert all scores to [-1.0, +1.0] range
4. **Smoothing** → Apply Simple Moving Average (10% window)
5. **XAI Analysis**:
   - Calculate ECC (global alignment)
   - Calculate EPC (local disagreement)
   - Generate visualizations
6. **Human Review** → Expert examines points of model disagreement

---

## Key Findings

### Case Study: Virginia Woolf's *To the Lighthouse*

**Results:**
- RoBERTa Large showed highest alignment with GPT-4 (ECC score)
- Only 47/3,700 sentences (1.3%) showed significant model disagreement
- Points of disagreement often indicated genuine emotional ambivalence in text
- Greybox method achieved ~GPT-4 performance with whitebox explainability

**Literary Discovery:**
The emotional arc follows a "distributed heroine" pattern across multiple characters rather than a single protagonist - a finding that emerged from this computational analysis.

---

## Installation & Usage

### Requirements

```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn
pip install transformers torch
pip install vaderSentiment textblob
pip install openai  # For GPT-4 API access
pip install scipy  # For Savitzky-Golay smoothing
```

### Quick Start

```python
from sentimentarcs_greybox import GreyboxEnsemble

# Initialize ensemble
ensemble = GreyboxEnsemble(
    models=['vader', 'textblob', 'roberta', 'gpt4'],
    smoothing_window=0.10
)

# Analyze text
results = ensemble.analyze_text('path/to/novel.txt')

# Calculate XAI metrics
ecc_scores = ensemble.calculate_ecc()  # Global alignment
epc_values = ensemble.calculate_epc()  # Local disagreement

# Visualize
ensemble.plot_sentiment_curves()
ensemble.plot_ecc_heatmap()
ensemble.plot_disagreement_points()
```

### GPT-4 API Setup

Requires OpenAI API key with GPT-4 access:

```python
import openai
openai.api_key = 'your-api-key'

# Uses function calling for structured output
function_schema = {
    "name": "sentiment_analysis",
    "description": "Finds sentiment polarity and emotion",
    "parameters": {
        "polarity": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "emotion": {"type": "string", "enum": ["happiness","sadness","anger","fear","disgust","surprise"]}
    }
}
```

**Note**: Processing 3,700 sentences with GPT-4 takes ~3.5 hours and costs approximately $10-15.

---

## Advantages Over Existing Methods

### vs. SentimentArcs (2021)
- ✅ Adds state-of-the-art GPT-4 models
- ✅ Introduces novel XAI metrics (ECC, EPC)
- ✅ Simplified ensemble (6 vs 30+ models)
- ✅ Human-in-the-loop workflow

### vs. Standard Sentiment Analysis
- ✅ Analyzes evolution over time (diachronic)
- ✅ Multi-model ensemble reduces error
- ✅ Explainability at local AND global levels
- ✅ Works with complex literary language

### vs. Pure GPT-4 Analysis  
- ✅ More explainable (whitebox alignment)
- ✅ More cost-effective (identifies best whitebox proxy)
- ✅ Privacy-preserving (can run whitebox locally)
- ✅ Faster for repeated analysis

---

## Limitations

**Current limitations:**
- Requires GPT-4 API access (restricted beta as of 2023)
- Processing time scales linearly with text length
- Best for texts >1000 sentences
- Struggles with heavily ironic or sarcastic language

**Future work:**
- Integration with more recent LLMs (GPT-4 Turbo, Claude)
- Automated hyperparameter optimization
- Multi-language support
- Real-time streaming analysis

---

## Citation

**Paper:**
```bibtex
@article{chun2023explainable,
  title={eXplainable AI with GPT4 for story analysis and generation: A novel framework for diachronic sentiment analysis},
  author={Chun, Jon and Elkins, Katherine},
  journal={International Journal of Digital Humanities},
  volume={5},
  pages={507--532},
  year={2023},
  publisher={Springer}
}
```

**Code:**
```bibtex
@software{chun2023greybox,
  author={Chun, Jon and Elkins, Katherine},
  title={SentimentArcs-Greybox: XAI Ensemble for Diachronic Sentiment Analysis},
  year={2023},
  publisher={GitHub},
  url={https://github.com/jon-chun/sentimentarcs-greybox}
}
```

---

## Related Projects

**By the Authors:**
- [SentimentArcs (2021)](https://github.com/jon-chun/sentimentarcs) - Original ensemble framework
- [The Shapes of Stories](https://github.com/KatherineElkins/humanities-the-shapes-of-cinderella) - Application to cross-cultural narratives
- [Cinderella Sentiment Analysis](https://github.com/KatherineElkins/humanities-the-shapes-of-cinderella) - Comparative analysis of 9 variants

**Related Work:**
- Syuzhet.R - Matthew Jockers' original R package
- VADER Sentiment - Hutto & Gilbert (2014)
- Hugging Face Transformers - BERT model implementations

---

## Support & Contributing

**Issues**: Report bugs or request features via GitHub Issues

**Contributing**: 
- Fork the repository
- Create feature branch (`git checkout -b feature/improvement`)
- Commit changes (`git commit -am 'Add new feature'`)
- Push to branch (`git push origin feature/improvement`)
- Create Pull Request

**Questions**: Contact the authors via GitHub or email

---

## License

MIT License - See LICENSE file for details

This work was supported by Kenyon College and the National Endowment for the Humanities.

---

## Acknowledgments

- OpenAI for GPT-4 API access during restricted beta
- Kenyon College Digital Humanities program
- Reviewers at *International Journal of Digital Humanities*
- Open-source community for foundational tools

---

## Learn More

- 📄 [Read the full paper](link-to-paper)
- 📚 [View methodology documentation](docs/methodology.md)
- 🎓 [The Shapes of Stories book](https://www.cambridge.org/core/books/shapes-of-stories/)
- 🔬 [Kenyon Digital Humanities](https://digital.kenyon.edu)

**Keywords**: sentiment analysis, explainable AI, XAI, GPT-4, BERT, LLM, digital humanities, computational narratology, greybox ensemble, diachronic analysis, literary analysis
