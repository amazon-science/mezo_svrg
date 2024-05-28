## MeZO-SVRG: Variance-Reduced Zero-Order Methods for fine-tuning LLMs

This repository implements the Memory-Efficient Zeroth-Order Stochastic Variance-Reduced Gradient (MeZO-SVRG) algorithm for fine-tuning pre-trained hugging face LMs. As baselines we also implement Memory-efficient ZO Optimizer (MeZO) and first-order SGD (FO-SGD). The repository is written in PyTorch and leverages the Pytorch Lightning framework. 

## Installation

To install the relevant python environment use the command


```bash
  conda create --name zo_opt python=3.9
  conda activate zo_opt
  python -m pip install -r requirements.txt
```
    
## File Overview

This repository implements the MeZO-SVRG algorithm and enables fine-tuning on a range on language models using the GLUE benchmark dataset. To run experiments, execute the 'finetune_llm.sh' bash script. 

The script supports the following models:
1. 'distilbert-base-cased'
2. 'roberta-large'
3. 'gpt2-xl'
4. 'facebook/opt-2.7b'
5. 'facebook/opt-6.7b'

The script supports the following GLUE tasks:
1. MNLI
2. QNLI
3. SST-2
4. CoLA

Indicate the fine-tuning algorithm by passing one of the following {'FO', 'ZO', 'ZOSVRG'}. The exact hyperparameter settings used to generate the tables/figures in the paper are provided in the Appendix.

## Citation

Please consider citing our paper if you use our code:
```text
@misc{gautam2024variancereduced,
      title={Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models}, 
      author={Tanmay Gautam and Youngsuk Park and Hao Zhou and Parameswaran Raman and Wooseok Ha},
      year={2024},
      eprint={2404.08080},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

