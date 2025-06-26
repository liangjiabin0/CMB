from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments, rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

functional_groups = {
    "Alcohol": Chem.MolFromSmarts("[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]"),
    "Carboxylic Acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "Ether": Fragments.fr_ether,
    "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
    "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
    "Alkyne": Chem.MolFromSmarts("[$([CX2]#C)]"),
    "Benzene": Fragments.fr_benzene,
    "Primary Amine": Chem.MolFromSmarts("[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]"),
    "Secondary Amine": Fragments.fr_NH1,
    "Tertiary Amine": Fragments.fr_NH0,
    "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "Cyano": Chem.MolFromSmarts("[NX1]#[CX2]"),
    "Fluorine": Chem.MolFromSmarts("[#6][F]"),
    "Chlorine": Chem.MolFromSmarts("[#6][Cl]"),
    "Iodine": Chem.MolFromSmarts("[#6][I]"),
    "Bromine": Chem.MolFromSmarts("[#6][Br]"),
    "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
    "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
    "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
    "Phosphoric Acid": Chem.MolFromSmarts(
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]"
    ),
    "Phosphoester": Chem.MolFromSmarts(
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]"
    ),
}


def strip(data: List[str]) -> List[str]:
    return [smiles.replace(" ", "") for smiles in data]


def load_data() -> Tuple[List[str], List[str]]:
    # 直接指定文件路径
    tgt_path = "resultsALL.csv"
    inference_path = "smiles_outputALL.csv"

    # 加载目标数据（SMILES）
    tgt = pd.read_csv(tgt_path)['rank1'].tolist()
    tgt = strip(tgt)

    # 加载预测数据（SMILES）
    preds = pd.read_csv(inference_path)['rank1'].tolist()
    preds_stripped = strip(preds)

    return preds_stripped, tgt


def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) == Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1


def get_functional_groups(mol):
    func_groups = dict()
    for func_group_name, smarts in functional_groups.items():
        func_groups[func_group_name] = match_group(mol, smarts)
    return func_groups


def score(preds: List[str], tgt: List[str]) -> pd.DataFrame:
    results = dict()

    for pred_smiles, tgt_smiles in tqdm.tqdm(zip(preds, tgt), total=len(tgt)):
        mol = Chem.MolFromSmiles(tgt_smiles)
        if mol is None:  # 如果 mol 为空，跳过该项
            print(f"Warning: Invalid SMILES string detected: {tgt_smiles}")
            continue

        tgt_smiles = Chem.MolToSmiles(mol)
        functional_groups = get_functional_groups(mol)

        results[tgt_smiles] = functional_groups

    # 将结果存入DataFrame
    return pd.DataFrame.from_dict(results, orient="index")

def main() -> None:
    RDLogger.DisableLog("rdApp.*")

    preds, tgt = load_data()
    results_score = score(preds, tgt)

    # 将结果写入CSV文件
    results_score.to_csv("functional_groups_match.csv")

    print("Results: Functional Group Match")
    print(results_score)


if __name__ == "__main__":
    main()
