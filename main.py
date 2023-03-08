import pandas as pd
import numpy as np
from io import StringIO


# 1.1
# 1
X = pd.read_csv("TP01_Manipulation_de_donnees-enonce/data/sy02-p2016.csv")

# 2
# X.shape()
# X.info()

# 3
X2 = pd.read_csv("TP01_Manipulation_de_donnees-enonce/data/sy02-p2016-2.csv", sep='&')
X3 = pd.read_csv("TP01_Manipulation_de_donnees-enonce/data/sy02-p2016-3.csv", sep='\t')
X4 = pd.read_csv("TP01_Manipulation_de_donnees-enonce/data/sy02-p2016-4.csv", sep=';')
X5 = pd.read_csv("TP01_Manipulation_de_donnees-enonce/data/sy02-p2016-5.csv", sep=' ',index_col=0)
# print(X.shape,X2.shape,X3.shape,X4.shape,X5.shape)

# 1.2
df = pd.read_csv(StringIO("0\n1.4"), header=None)

# print(pd.read_csv(StringIO("0\n1"), header=None).dtypes)
# print(pd.read_csv(StringIO("0\n1.3"), header=None).dtypes)
# print(pd.read_csv(StringIO("T\nF"), header=None).dtypes)
# print(pd.read_csv(StringIO("True\nFalse"), header=None).dtypes)

# 4
X.specialite = pd.Categorical(X.specialite)
X.statut = pd.Categorical(X.statut)
X["dernier diplome obtenu"] = pd.Categorical(X["dernier diplome obtenu"])

# correcteur_type =
# X.info()


# 1.3
# 5
X= pd.read_csv("TP01_Manipulation_de_donnees-enonce/data/effectifs.csv")

# X.Semetre = X.Semestre.str[8:] #错误做法
X = X.assign(Semestre=X.Semestre.str[8:])
# print(X.head())

# 6
X = X.assign(Saison=X.Semestre.str[0])
X = X.assign(Annee=X.Semestre.str[1:])
X.drop(columns="Semestre", inplace=True)  # 注意不要先drop了
# print(X.head())

# 7
X = X.melt(id_vars=["Saison", "Annee"], value_name="effectif", var_name="UV")
X = X.loc[~pd.isna(X.effectif)]  # 用于选择非空数据行
# print(X)

# 8
import seaborn as sns
iris = sns.load_dataset("iris")
iris = iris.melt()



