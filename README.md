# DS_DTEFCC

Code for Dialogue summarization with topic enhancement and factual consistency contrast.


## Setup

```bash
cd src
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The default configs use `facebook/bart-large` from Hugging Face.


## Running

From the `src` directory:

```bash
python run_summarization.py config/samsum_config.json
```


## Acknowledgements

This codebase is mainly based on [Hannibal046/SDDS](https://github.com/Hannibal046/SDDS) and [adjidieng/ETM](https://github.com/adjidieng/ETM). We thank the authors for releasing their code.

## Citation

```bibtex
@article{liu2026dialogue,
  author  = {Zhanghui Liu and Zhang Wentao and Yuzhong Chen and Lin Yixin},
  title   = {Dialogue summarization with topic enhancement and factual consistency contrast},
  journal = {Computer Speech \& Language},
  volume  = {100},
  pages   = {101958},
  year    = {2026},
  doi     = {10.1016/j.csl.2026.101958}
}
```
