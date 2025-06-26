# CMB
所有模型都在Python 3.10和PyTorch 2.1.1中实现。分析使用了tqdm 4.66.1, rdkit 2024.03.5, typing-extensions 4.14.0，可视化使用了matplotlib 3.5.3, pandas 2.2.3, numpy 1.22.4。
    CMB.py包含了打开数据库、规范化SMILES分子式、模型训练、预测生成SMILES分子式并写入表格、校验模型top-1、top5-、top-10 accuracy的功能,function group.py代码定义了官能团，并通过CMB.py生成的SMILES分子式的表格，进行官能团准确率的评测。同理，MCS.py和Levenshtein distance.py同理提供了MCS和Levenshtein distance准确率的评测方法
