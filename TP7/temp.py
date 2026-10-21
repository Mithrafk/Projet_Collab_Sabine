import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


d_descri = pd.read_csv("tables/description-indiv.csv", sep=";", low_memory=False) #, low_memory=False uniquement pour supprimer le warning sur les colonnes de type object (colonnes qu'on supprime après de tte manière)
d_gpe = pd.read_csv("tables/conso-gpe-inca3.csv", sep=";")
d_app = pd.read_csv("tables/apports-nut-alim.csv", sep=";")
d_hab = pd.read_csv("tables/habitudes-indiv.csv", sep=";", low_memory=False)

noind_communs = set(d_gpe['NOIND']) & set(d_app['NOIND']) & set(d_hab['NOIND']) & set(d_descri['NOIND'])

print(f"Nombre d'individus dans description-indiv.csv : {len(d_descri)}")
print(f"Nombre d'individus dans conso-gpe-inca3.csv : {len(d_gpe)}")
print(f"Nombre d'individus dans apports-nut-alim.csv : {len(d_app)}")
print(f"Nombre d'individus dans habitudes-indiv.csv : {len(d_hab)}")
print(f"Nombre total d'individus communs : {len(noind_communs)}")

df = (
    d_descri[d_descri["NOIND"].isin(noind_communs)]
    # .merge(
    #     d_gpe[d_gpe["NOIND"].isin(noind_communs)],
    #     on="NOIND",
    #     how="inner",
    #     suffixes=("_descri", "_gpe")
    # )
    # .merge(
    #     d_app[d_app["NOIND"].isin(noind_communs)],
    #     on="NOIND",
    #     how="inner",
    #     suffixes=("", "_app")
    # )
    # .merge(
    #     d_hab[d_hab["NOIND"].isin(noind_communs)],
    #     on="NOIND",
    #     how="inner",
    #     suffixes=("", "_hab")
    # )
)
nb_str = len(df.select_dtypes(include=['object', 'string']).columns)
nb_num = len(df.select_dtypes(include=['number']).columns)

print("--- Décompte des types de colonnes ---")
print(f"Nombre de colonnes au format texte (str) : {nb_str}")
print(f"Nombre de colonnes au format numérique (int/float) : {nb_num}")
print('\n')

d_str = df.select_dtypes(include=['object', 'string'])
for k in d_str.columns:
    print(f"Colonne '{k}' : {d_str[k].nunique()} valeurs uniques, {d_str[k].isna().sum()} valeurs manquantes")
    # print(d_str[k].unique())

df["zae"] = df["zae"].str[1:].astype("Int64") # Z78267 -> 78267
colonnes_libelle = df.columns[df.columns.str.endswith("_libelle")]
df.drop(columns=colonnes_libelle, inplace=True) # supprimer les colonnes *_libelles, les raisons du oui à la colonne d'avant, annecdotique pour nous


# Pourcentage de valeurs manquantes dans les deux tables
nan_descri = d_descri.isna().mean() * 100
nan_df = df.isna().mean() * 100

bins = [-0.01, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100.01]
labels = [
    '0', ']0-10]', ']10-20]', ']20-30]', ']30-40]', ']40-50]',
    ']50-60]', ']60-70]', ']70-80]', ']80-90]', ']90-99]', ']99-100]'
]

counts_descri = pd.cut(
    nan_descri, bins=bins, labels=labels
).value_counts(sort=False)

counts_df = pd.cut(
    nan_df, bins=bins, labels=labels
).value_counts(sort=False)

x = np.arange(len(labels))

plt.figure(figsize=(10, 6))

# Table d_descri en arrière-plan
plt.bar(
    x, counts_descri.values,
    color='red', alpha=0.55,
    edgecolor='black', label='d_descri initiale'
)

# Table df au premier plan
plt.bar(
    x, counts_df.values,
    color='blue', alpha=0.75,
    edgecolor='black', label='df après filtrage str + lignes'
)

plt.title("Répartition des colonnes selon le pourcentage de valeurs manquantes")
plt.xlabel("Pourcentage de valeurs manquantes (NaN)")
plt.ylabel("Nombre de colonnes")
plt.xticks(x, labels, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# plt.savefig("graphs/nan_distribution_comparaison.png")
plt.show()


df_adulte = df[df["tage_PS"] >= 7]
df_nourisson = df[df["tage_PS"] < 3]
df_enfant = df[(df["tage_PS"] >= 3) & (df["tage_PS"] < 7)]

bins = [0,1,2,3,4,5,6,7,8,9]
labels = ['0-11 mois', '1-3 ans', '4-6 ans', '7-10 ans', '11-14 ans', '15-17 ans ', '18-44 ans ', '45-64 ans', '65-79 ans ']

counts_adulte = pd.cut(df_adulte["tage_PS"], bins=bins, labels=labels)
counts_nourisson = pd.cut(df_nourisson["tage_PS"], bins=bins, labels=labels)
counts_enfant = pd.cut(df_enfant["tage_PS"], bins=bins, labels=labels)

x = np.arange(len(labels))

plt.figure(figsize=(10, 6))
plt.bar(x, counts_adulte.value_counts(sort=False), color='blue', alpha=0.75, edgecolor='black', label='Adultes (18 ans et plus)')
plt.bar(x, counts_nourisson.value_counts(sort=False), color='green', alpha=0.75, edgecolor='black', label='Nourrissons (moins de 3 ans)')
plt.bar(x, counts_enfant.value_counts(sort=False), color='orange', alpha=0.75, edgecolor='black', label='Enfants (3 à 17 ans)')
plt.xlabel('Tranche d\'âge')
plt.ylabel('Nombre de personnes')
plt.title('Répartition des personnes par tranche d\'âge')
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.show()



dfs = {
    "Adultes": df_adulte,
    "Enfants": df_enfant,
    "Nourrissons": df_nourisson
}

for nom, data in dfs.items():
    # 1. Supprimer les colonnes vides
    colonnes_vides = data.columns[data.isna().all()]
    data.drop(columns=colonnes_vides, inplace=True)

    # 2. Afficher les informations sur les colonnes restantes
    print(f"--- {nom} ---")
    print(f"{nom} : {len(colonnes_vides)} colonne(s) vide supprimée(s)")
    for k in data.columns:
        if data[k].isna().sum() > 0:
            print(
                f"Colonne '{k}' : "
                f"{data[k].nunique()} valeurs uniques, "
                f"{data[k].isna().sum()/len(data)*100:.2f}% de valeurs manquantes"
            )
    print('\n')



# Pourcentage de valeurs manquantes par colonne
bins = [-0.01, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100.01]
labels = [
    "0", "]0-10]", "]10-20]", "]20-30]", "]30-40]", "]40-50]",
    "]50-60]", "]60-70]", "]70-80]", "]80-90]", "]90-99]", "]99-100]"
]

comptages = {}

for nom, data in dfs.items():
    pourcentage_nan = data.isna().mean() * 100
    classes_nan = pd.cut(
        pourcentage_nan,
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    comptages[nom] = classes_nan.value_counts(sort=False)

x = np.arange(len(labels))
largeur = 0.25

plt.figure(figsize=(12, 6))

plt.bar(
    x - largeur, comptages["Adultes"],
    width=largeur, color="blue", alpha=0.75, label="Adultes"
)
plt.bar(
    x, comptages["Enfants"],
    width=largeur, color="orange", alpha=0.75, label="Enfants"
)
plt.bar(
    x + largeur, comptages["Nourrissons"],
    width=largeur, color="green", alpha=0.75, label="Nourrissons"
)

plt.title("Nombre de colonnes selon leur pourcentage de valeurs manquantes")
plt.xlabel("Pourcentage de valeurs manquantes")
plt.ylabel("Nombre de colonnes")
plt.xticks(x, labels, rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

