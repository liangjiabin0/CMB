import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS

# 读取smiles_output.csv文件
file_path = r'D:\360极速浏览器X下载\smiles_output.csv'
smiles_data = pd.read_csv(file_path)


# 函数：计算最大公共子结构（MCS）的准确率
def calculate_mcs_accuracy(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0  # 无效的SMILES，返回0

    # 计算最大公共子结构
    mcs = rdFMCS.FindMCS([mol1, mol2])
    if mcs.numAtoms == 0:
        return 0  # 没有找到公共子结构

    # 获取最大公共子结构的SMILES
    mcs_smiles = mcs.smartsString

    # 计算最大公共子结构的准确率
    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    mcs_size = mcs_mol.GetNumAtoms()
    mol1_size = mol1.GetNumAtoms()
    mol2_size = mol2.GetNumAtoms()

    # 计算准确率（MCS大小与原子总数的比值）
    accuracy1 = mcs_size / mol1_size
    accuracy2 = mcs_size / mol2_size

    # 返回平均准确率
    return (accuracy1 + accuracy2) / 2


# 生成新的列，用于存储每个SMILES的最大公共子结构准确率
mcs_accuracies = []
for smile in smiles_data['SMILES']:
    max_accuracy = 0
    for comparison_smile in smiles_data['SMILES']:
        if smile != comparison_smile:
            accuracy = calculate_mcs_accuracy(smile, comparison_smile)
            max_accuracy = max(max_accuracy, accuracy)
    mcs_accuracies.append(max_accuracy)

# 将计算结果写入新的CSV文件
smiles_data['MCS_Accuracy'] = mcs_accuracies
output_file_path = r'D:\360极速浏览器X下载\smiles_output_with_mcs_accuracy.csv'
smiles_data.to_csv(output_file_path, index=False)

output_file_path
