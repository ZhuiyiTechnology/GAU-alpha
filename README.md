# GAU-α
基于Gated Attention Unit的Transformer模型（尝鲜版）

## 介绍

- GAU-α：https://kexue.fm/archives/9052
- GAU：https://kexue.fm/archives/8934
- 原始论文：https://arxiv.org/abs/2202.10447

## 评测

### CLUE榜单分类任务结果

|         | iflytek | tnews | afqmc | cmnli | ocnli | wsc | csl | 
| :-----: | :-----: | :---: | :---: | :---: | :---: | :---: | :---: | 
| BERT | 60.06 | 56.80 | 72.41 | 79.56 | 73.93 | 78.62 | 83.93 | 
| RoBERTa | 60.64 | **58.06** | 74.05 | 81.24 | 76.00 | **87.50** | 84.50 | 
| RoFormer | 60.91 | 57.54 | 73.52 | 80.92 | **76.07** | 86.84 | 84.63 | 
| RoFormerV2<sup>*</sup> | 60.87 | 56.54 | 72.75 | 80.34 | 75.36 | 80.92 | 84.67 | 
| GAU-α | **61.41** | 57.76 | **74.17** | **81.82** | 75.86 | 79.93 | **85.67** | 

### CLUE榜单阅读理解和NER结果

|         | cmrc2018 | c3 | chid | cluener |
| :-----: | :-----: | :---: | :---: | :---: | 
| BERT | 56.17 | 60.54 | 85.69 | 79.45 |
| RoBERTa | 56.54 | 67.66 | 86.71 | 79.47 |
| RoFormer | 56.26 | 67.24 | 86.57 | 79.72 |
| RoFormerV2<sup>*</sup> | 57.91 | 64.62 | 85.09 | **81.08** |
| GAU-α | **58.09** | **68.24** | **87.91** | 80.01 |

## 使用

需要bert4keras>=0.11.2。参考代码：
```python
from bert4keras.models import build_transformer_model
from models import GAU_alpha

gau_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=GAU_alpha,
)
```

## 下载

- **Base版**：[chinese_GAU-alpha-char_L-24_H-768.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_GAU-alpha-char_L-24_H-768.zip)、[百度云](https://pan.baidu.com/s/1NnWvJCin3v7MAZfAy2y0Gg)(提取码：0d4s)

## 引用

Bibtex：

```tex
@techreport{gau-alpha,
  title={GAU-α: GAU-based Transformers for NLP - ZhuiyiAI},
  author={Jianlin Su, Shengfeng Pan, Bo Wen, Yunfeng Liu},
  year={2022},
  url="https://github.com/ZhuiyiTechnology/GAU-alpha",
}
```

## 联系

- 邮箱：ai@wezhuiyi.com
- 追一科技：https://zhuiyi.ai
