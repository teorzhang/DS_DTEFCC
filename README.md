# DS_DTEFCC

Code for dialogue summarization with topic enhancement and factual consistency contrast.


## Setup

```bash
cd src
pip install -r requirements.txt
```

The default configs use `facebook/bart-large` from Hugging Face.


## Running

From the `src` directory:

```bash
python run_summarization.py config/samsum_config.json
```


## Acknowledgements

This codebase is mainly based on [Hannibal046/SDDS](https://github.com/Hannibal046/SDDS), [adjidieng/ETM](https://github.com/adjidieng/ETM), and [shon-otmazgin/fastcoref](https://github.com/shon-otmazgin/fastcoref). We thank the authors for releasing their code.

## Citation

If you find our work useful, please consider citing it as:
```bibtex
@article{2026,
  author  = {Zhanghui Liu and Zhang Wentao and Yuzhong Chen and Lin Yixin},
  title   = {Dialogue summarization with topic enhancement and factual consistency contrast},
  journal = {Computer Speech \& Language},
  volume  = {100},
  pages   = {101958},
  year    = {2026},
  doi     = {10.1016/j.csl.2026.101958}
}
```
