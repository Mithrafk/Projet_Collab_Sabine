import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def importer_donnees(dossier_tables="tables", info=False):
    '''
    Importe les données des différents fichiers CSV et les fusionner.
    Ne retenir que les lignes communes aux 4 fichiers.
    '''
    d_descri = pd.read_csv(f"{dossier_tables}/description-indiv.csv", sep=";", low_memory=False)
    d_app = pd.read_csv(f"{dossier_tables}/apports-nut-alim.csv", sep=";")

    noind_communs = set(d_app['NOIND']) & set(d_descri['NOIND'])

    dataframe = (d_descri[d_descri["NOIND"].isin(noind_communs)])

    if info:
        print(f"Nombre d'individus dans description-indiv.csv : {len(d_descri)}")
        print(f"Nombre d'individus dans apports-nut-alim.csv : {len(d_app)}")
        print(f"Nombre total d'individus communs : {len(noind_communs)}")

    return dataframe

def nettoyer_str(dataframe):
    dataframe["zae"] = dataframe["zae"].str[1:].astype("Int64") # Z78267 -> 78267
    colonnes_libelle = dataframe.columns[dataframe.columns.str.endswith("_libelle")]
    dataframe.drop(columns=colonnes_libelle, inplace=True) # supprimer les colonnes *_libelles, les raisons du oui à la colonne d'avant, annecdotique pour nous

    return dataframe

def sep_age(dataframe, forme_return="liste"):
    '''
    Sépare le dataframe en trois sous-dataframes selon l'âge : adultes, nourrissons et enfants.
    Peut rentre 3 sous-dataframes ou un dictionnaire de dataframes selon le paramètre forme_return.
    '''
    df_adulte = dataframe[dataframe["tage_PS"] >= 7]
    df_nourisson = dataframe[dataframe["tage_PS_mois"].notna()]
    df_enfant = dataframe[(dataframe["tage_PS_mois"].isna()) & (dataframe["tage_PS"] < 7)]

    if forme_return == "liste":
        return df_adulte, df_nourisson, df_enfant

    elif forme_return == "dict":
        df_age = {
        "Adultes": df_adulte,
        "Enfants": df_enfant,
        "Nourrissons": df_nourisson
    }
        return df_age

    else:
        raise ValueError("Le paramètre 'forme_return' doit être soit 'liste' soit 'dict'.")

def supp_col_vides(dataframe):
    '''
    supprime les colonnes vides (toutes les valeurs sont NaN ou NA_APPL) du dataframe.
    '''
    colonnes_vides = dataframe.columns[(dataframe.isna() | dataframe.eq(NA_APPL)).all()]
    return dataframe.drop(columns=colonnes_vides)

def remplir(data, masque, colonnes, valeur):
    '''Écrit `valeur` dans `colonnes` pour les lignes de `masque`, UNIQUEMENT là où la case est vide (NaN).
    On n'écrase donc jamais une vraie réponse.'''
    colonnes = [colonnes] if isinstance(colonnes, str) else list(colonnes)
    for col in colonnes:
        data.loc[masque & data[col].isna(), col] = valeur

def connue(serie):
    '''Vrai là où on a une vraie valeur (ni NaN, ni non applicable)'''
    return serie.notna() & serie.ne(NA_APPL)

def sans_non_applicable(data):
    '''Copie avec NaN à la place de NA_APPL : à utiliser pour moyennes, régressions, imputations'''
    return data.mask(data.eq(NA_APPL))

def bilan_na(avant, apres):
    '''Tableau NaN avant / NaN après / non applicable, colonne par colonne'''
    t = pd.DataFrame({
        "NaN_avant": avant.isna().sum(),
        "NaN_apres": apres.isna().sum(),
        "non_applicable": apres.eq(NA_APPL).sum(),
    })
    t["valeur retrouvée"] = t["NaN_avant"] - t["NaN_apres"] - t["non_applicable"]
    t["NaN_diff"] = t["NaN_avant"] - t["NaN_apres"]
    return t

def nettoyer_fume(dataframe, bilan=False):
    '''
    Nettoie les données des colonnes liées aux fumeurs.
    '''
    colonnes_fume = ["NOIND", "fume", "nb_cigarettes_jour", "nb_cigarettes_sem", "nb_cigarettes_nsp",
                    "nb_cigares_jour", "nb_cigares_sem", "nb_cigares_nsp",
                    "nb_pipes_jour", "nb_pipes_sem", "nb_pipes_nsp",
                    "fume_age_debut", "fume_age_debut_nsp", "fume_age_arret", "fume_age_arret_nsp"]

    # (jour, semaine, ne sait pas) pour chaque type de tabac
    groupes = {
        "cigarettes": ["nb_cigarettes_jour", "nb_cigarettes_sem", "nb_cigarettes_nsp"],
        "cigares": ["nb_cigares_jour", "nb_cigares_sem", "nb_cigares_nsp"],
        "pipes": ["nb_pipes_jour", "nb_pipes_sem", "nb_pipes_nsp"],
    }

    d_fume = dataframe.loc[:, colonnes_fume].copy()
    autres_colonnes = colonnes_fume[1:]  # tout sauf NOIND
    colonnes_nb = [c for c in autres_colonnes if c.startswith("nb_")]
    colonnes_age = [c for c in autres_colonnes if c.startswith("fume_age_")]
    colonnes_jour = [j for j, _, _ in groupes.values()]
    colonnes_sem = [s for _, s, _ in groupes.values()]

    # 1) Question posée à partir de 11 ans (tage_PS >= 5) -> non applicable en dessous
    remplir(d_fume, dataframe["tage_PS"] < 5, autres_colonnes, NA_APPL)

    # 2) `fume` vide mais consommation renseignée et pas d'âge d'arrêt (2 individus) = fumeur actuel
    #    quotidien si une quantité/jour est renseignée, sinon occasionnel
    conso_jour = d_fume[colonnes_jour].fillna(0).gt(0).any(axis=1)
    conso_sem = d_fume[colonnes_sem].fillna(0).gt(0).any(axis=1)
    pas_arret = d_fume[["fume_age_arret", "fume_age_arret_nsp"]].isna().all(axis=1)
    remplir(d_fume, conso_jour & pas_arret, "fume", 1)
    remplir(d_fume, conso_sem & pas_arret, "fume", 2)

    # 3) Les profils (format_fume : 1 quotidien, 2 occasionnel, 3 ex-fumeur, 4 jamais fumé)
    actuel = d_fume["fume"].isin([1, 2])
    ex_fumeur = d_fume["fume"].eq(3)
    jamais = d_fume["fume"].eq(4)

    # Ex-fumeur ou jamais fumé : consommation actuelle = 0
    # (pour les ex-fumeurs, ça écrase les quantités déclarées pour la période où ils fumaient)
    d_fume.loc[ex_fumeur | jamais, colonnes_nb] = 0

    # Jamais fumé : pas d'âge de début ni d'arrêt -> non applicable
    d_fume.loc[jamais, colonnes_age] = NA_APPL

    # Fumeur actuel : n'a pas arrêté -> âge d'arrêt non applicable
    remplir(d_fume, actuel, ["fume_age_arret", "fume_age_arret_nsp"], NA_APPL)

    # 4) Fumeur actuel : quantités jour <-> semaine, type par type
    for jour, sem, nsp in groupes.values():
        # "ne sait pas" : on ne peut déterminer ni le jour ni la semaine
        d_fume.loc[actuel & d_fume[nsp].eq(1), [jour, sem]] = np.nan

        sait = d_fume[nsp].ne(1)
        mask_jour = actuel & sait & d_fume[jour].isna() & d_fume[sem].notna()
        d_fume.loc[mask_jour, jour] = d_fume.loc[mask_jour, sem] / 7

        mask_sem = actuel & sait & d_fume[sem].isna() & d_fume[jour].notna()
        d_fume.loc[mask_sem, sem] = d_fume.loc[mask_sem, jour] * 7

    # 5) Si un type de tabac est consommé, les autres types passent à 0 (seulement s'ils sont tous vides)
    for source, (jour_s, sem_s, _) in groupes.items():
        consomme = actuel & d_fume[[jour_s, sem_s]].fillna(0).gt(0).any(axis=1)
        for cible, colonnes_cible in groupes.items():
            if cible != source:
                remplir(d_fume, consomme & d_fume[colonnes_cible].isna().all(axis=1), colonnes_cible, 0)

    # 6) "Ne sait pas" = 0 quand la valeur est connue
    for jour, sem, nsp in groupes.values():
        remplir(d_fume, d_fume["fume"].isin([1, 2, 3]) & (connue(d_fume[jour]) | connue(d_fume[sem])), nsp, 0)
    remplir(d_fume, connue(d_fume["fume_age_debut"]), "fume_age_debut_nsp", 0)
    remplir(d_fume, connue(d_fume["fume_age_arret"]), "fume_age_arret_nsp", 0)

    if bilan:
        return d_fume, bilan_na(dataframe[colonnes_fume], d_fume)

    return d_fume

def nettoyer_regime(dataframe, bilan=False):
    ''' 
    Nettoie les données des colonnes liées aux regimes.
    bilan = True donne en 2nd retrun le pd bilan_na
    '''
    
    colonnes_regime = ["regime_passe", "regime_nb_2dernann", "regime_nb_anter2dernann", "regime_type",
                    "regime_duree_sem", "regime_duree_mois", "regime_duree_nsp"]
    nb_reg = ["regime_nb_2dernann", "regime_nb_anter2dernann"]
    duree = ["regime_duree_sem", "regime_duree_mois", "regime_duree_nsp"]

    d_reg = dataframe.loc[:, colonnes_regime].copy()

    # le code 7 n'existe pas dans la notice (format_regime_nb : 1 à 6) et va avec un regime_type vide -> manquant
    d_reg[nb_reg] = d_reg[nb_reg].replace(7, np.nan)

    # regime_passe n'est posée qu'à partir de 15 ans (tage_PS >= 6) -> non applicable en dessous
    remplir(d_reg, dataframe["tage_PS"] < 6, colonnes_regime, NA_APPL)

    # Jamais fait de régime (regime_passe = 2) : 0 régime ; type et durée n'ont pas de sens
    non = d_reg["regime_passe"].eq(2)
    remplir(d_reg, non, nb_reg, 0)
    remplir(d_reg, non, ["regime_type"] + duree, NA_APPL)

    # Durée : la personne donne soit des semaines, soit des mois -> on remplit l'autre colonne
    SEM_PAR_MOIS = 52 / 12 
    oui = d_reg["regime_passe"].eq(1)

    mask_sem = oui & d_reg["regime_duree_sem"].isna() & d_reg["regime_duree_mois"].notna()
    d_reg.loc[mask_sem, "regime_duree_sem"] = d_reg.loc[mask_sem, "regime_duree_mois"] * SEM_PAR_MOIS

    mask_mois = oui & d_reg["regime_duree_mois"].isna() & d_reg["regime_duree_sem"].notna()
    d_reg.loc[mask_mois, "regime_duree_mois"] = d_reg.loc[mask_mois, "regime_duree_sem"] / SEM_PAR_MOIS

    # "ne sait pas la durée" = 0 quand la durée est connue
    remplir(d_reg, oui & connue(d_reg["regime_duree_sem"]), "regime_duree_nsp", 0)

    if bilan:
        return d_reg, bilan_na(dataframe[colonnes_regime], d_reg)

    return d_reg

def nettoyer_poids(dataframe, bilan=False):
    '''
    Nettoie les données des colonnes liées aux poids.
    bilan = True donne en 2nd retrun le pd bilan_na
    '''
    moyens = ["poids_modifalim", "poids_plusAP", "poids_medicaments", "poids_substituts", "poids_chirurgie"]
    qtes = [c for c in dataframe.columns if c.startswith("poids_modifalim_") and not c.endswith("_libelle")]
    adulte_seul = ["poidsmax", "poidsmax_nsp", "age_poidsmax", "age_poidsmax_nsp",
                "poidsmin", "poidsmin_nsp", "age_poidsmin", "age_poidsmin_nsp", "nb_prise_10kg"]
    colonnes_poids = (["poids_anndern", "poids_anndern_nsp", "poids_modif", "poids_perception"]
                    + moyens + qtes + adulte_seul)

    d_poids = dataframe.loc[:, colonnes_poids].copy()
    age = dataframe["tage_PS"]

    # Questions non posées selon l'âge
    remplir(d_poids, age < 5, ["poids_modif", "poids_perception"] + moyens + qtes, NA_APPL)
    remplir(d_poids, age < 6, ["poids_anndern", "poids_anndern_nsp"], NA_APPL)
    remplir(d_poids, age < 7, adulte_seul, NA_APPL)

    # N'a pas essayé de perdre / ne pas prendre de poids -> "Non" pour chaque moyen
    remplir(d_poids, d_poids["poids_modif"].isin([3, 4]), moyens, 0)

    # Quantités modifiées : seulement si a modifié son alimentation
    remplir(d_poids, d_poids["poids_modifalim"].eq(0), qtes, NA_APPL)

    # "ne sait pas" = 0 quand la valeur est connue
    for val in ["poids_anndern", "poidsmax", "age_poidsmax", "poidsmin", "age_poidsmin"]:
        remplir(d_poids, connue(d_poids[val]), val + "_nsp", 0)

    if bilan:
        return d_poids, bilan_na(dataframe[colonnes_poids], d_poids)
    return d_poids

def nettoyer_femmes(dataframe, bilan=False):
    '''
    Nettoie les données des colonnes liées aux femmes.
    bilan = True donne en 2nd retrun le pd bilan_na
    '''
    colonnes_femmes = ["menopause", "enceinte", "enceinte_nbmois", "allaite", "allaite_nbsem", "enceinte_12dermois"]
    d_fem = dataframe.loc[:, colonnes_femmes].copy()

    remplir(d_fem, dataframe["sex_PS"].eq(1), colonnes_femmes, NA_APPL)          # hommes
    remplir(d_fem, dataframe["tage_PS"] < 7, "menopause", NA_APPL)               # ménopause : 18 ans et +
    remplir(d_fem, dataframe["tage_PS"] < 6, colonnes_femmes[1:], NA_APPL)       # grossesse / allaitement : 15 ans et +

    # Femme ménopausée : certain que non
    menopausee = d_fem["menopause"].eq(1)
    remplir(d_fem, menopausee, "enceinte", 2)                              # 2 = Non (format_ouinonnsp)
    remplir(d_fem, menopausee, ["allaite", "enceinte_12dermois"], 0)      # 0 = Non (format_ouinon)

    # Quantités : 0 si pas enceinte / n'allaite pas
    remplir(d_fem, d_fem["enceinte"].eq(2), "enceinte_nbmois", 0)
    remplir(d_fem, d_fem["allaite"].eq(0), "allaite_nbsem", 0)

    if bilan:
        return d_fem, bilan_na(dataframe[colonnes_femmes], d_fem)
    return d_fem

def nettoyer_profession(dataframe, bilan=False):
    '''
    Nettoie les données des colonnes liées aux professions.
    bilan = True donne en 2nd retrun le pd bilan_na
    '''
    colonnes_pro = ["atrav_interv", "trav_nuit_interv", "trav_nuit_2cl_interv", "tps_travail_interv",
                    "atrav_PR", "tps_travail_PR", "lien_interv_PR", "lien_PS_PR"]
    d_pro = dataframe.loc[:, colonnes_pro].copy()

    # Interviewé
    a_un_emploi = dataframe["situ_prof_5cl_interv"].eq(1)
    remplir(d_pro, a_un_emploi, "atrav_interv", 1)                                      # 1 = Oui
    remplir(d_pro, ~a_un_emploi, ["trav_nuit_interv", "trav_nuit_2cl_interv"], NA_APPL)
    remplir(d_pro, d_pro["atrav_interv"].eq(2), "tps_travail_interv", NA_APPL)

    # Personne de référence
    remplir(d_pro, dataframe["PCS_8cl_PR"].between(1, 6), "atrav_PR", 1)
    remplir(d_pro, d_pro["atrav_PR"].eq(2), "tps_travail_PR", NA_APPL)

    # Liens
    interv_est_PR = dataframe["interv_PR"].eq(1)
    adulte = dataframe["tage_PS"] >= 7
    remplir(d_pro, interv_est_PR, "lien_interv_PR", NA_APPL)
    remplir(d_pro, interv_est_PR & adulte, "lien_PS_PR", NA_APPL)
    remplir(d_pro, interv_est_PR & ~adulte & dataframe["lien_rep_enf"].isin([1, 2]), "lien_PS_PR", 2)  # 2 = Enfant

    if bilan:
        return d_pro, bilan_na(dataframe[colonnes_pro], d_pro)
    return d_pro

def nettoyer_divers(dataframe, bilan=False):
    '''
    Nettoie les données des colonnes diverses.
    bilan = True donne en 2nd retrun le pd bilan_na
    '''
    colonnes_div = ["intoall_confirm_med", "regime_forme", "IA_score"]
    d_div = dataframe.loc[:, colonnes_div].copy()

    remplir(d_div, dataframe["maladie_allergie_alim"].eq(0), "intoall_confirm_med", NA_APPL)
    remplir(d_div, dataframe["tage_PS_mois"].notna(), "regime_forme", NA_APPL)
    remplir(d_div, dataframe["IA_statut"].eq(0), "IA_score", NA_APPL)

    if bilan:
        return d_div, bilan_na(dataframe[colonnes_div], d_div)
    return d_div

def importer_et_nettoyer_donnees():
    '''
    fonction qui importe les données et les nettoie en appliquant toutes les fonctions de nettoyage.
    '''
    df = importer_donnees(info=False)  # on reprend le df initial pour ne pas avoir les colonnes nettoyées
    df = supp_col_vides(df)  # on supprime les colonnes vides avant de nettoyer les autres
    df = nettoyer_str(df)

    for func in [nettoyer_fume, nettoyer_regime, nettoyer_poids, nettoyer_femmes, nettoyer_profession, nettoyer_divers]:
        d_nett = func(df, bilan=False)
        colonnes = [c for c in d_nett.columns if c != "NOIND"]
        df[colonnes] = d_nett[colonnes]

    # Mesuré / déclaré : un indicateur à la place de 4 colonnes presque vides
    df["taille"] = np.select([df["taille"].notna(), df["taille_m"].notna(), df["taille_d"].notna()], 
                            [df["taille"], df["taille_m"], df["taille_d"]], 
                            default=np.nan)
    df["poids"] = np.select([df["poids"].notna(), df["poids_m"].notna(), df["poids_d"].notna()], 
                            [df["poids"], df["poids_m"], df["poids_d"]], 
                            default=np.nan)
    df = df.drop(columns=["taille_m", "taille_d", "poids_m", "poids_d"]).copy()
    df = supp_col_vides(df)
    return df

def nsp_refus_en_nan(dataframe):
    """Remplace les codes "Ne sait pas" et "Refus" par NaN (une absence de réponse reste une absence de réponse)."""
    data = dataframe.copy()
    for col, codes in CODES_NSP_REFUS.items():
        if col in data.columns:
            data[col] = data[col].mask(data[col].isin(codes))
    return data

def pourct_na(data):
    return data.isna().mean() * 100

def supprimer_lignes_rares(data):
    """
    Supprime les lignes ayant un NaN dans une colonne presque complète.
    On prend le plus grand seuil de SEUILS_LIGNES qui supprime au plus PLAFOND_LIGNES % des lignes.
    Renvoie (data réduit, infos).
    """
    pct = pourct_na(data)
    choix = {"seuil": 0, "colonnes": [], "pct_lignes": 0.0}
    for seuil in SEUILS_LIGNES:
        colonnes = list(pct[(pct > 0) & (pct <= seuil)].index)
        pct_lignes = data[colonnes].isna().any(axis=1).mean() * 100
        if pct_lignes <= PLAFOND_LIGNES:
            choix = {"seuil": seuil, "colonnes": colonnes, "pct_lignes": pct_lignes}
    masque = data[choix["colonnes"]].isna().any(axis=1)
    return data[~masque].copy(), {**choix, "nb_lignes": int(masque.sum())}

def imputer_colonne(data, col):
    """Impute les NaN de `col` dans `data` (modifié sur place). Renvoie une ligne de journal, ou None."""
    type_col = type_colonne(col)
    pop = population_applicable(data, col)
    cibles = pop & data[col].isna()
    connus = pop & data[col].notna()
    if cibles.sum() == 0 or connus.sum() < MIN_CONNUS:
        return None

    y = data.loc[connus, col]
    valeur = pd.Series(np.nan, index=data.index[cibles])
    methode, r2 = None, np.nan

    # 1) Régression linéaire sur les colonnes complètes (si elle explique assez la colonne)
    if type_col in ("quantitative", "ordinale") and connus.sum() >= MIN_REGRESSION:
        X = data.loc[connus, PREDICTEURS]
        ok = X.notna().all(axis=1)
        if ok.sum() >= MIN_REGRESSION:
            r2 = cross_val_score(LinearRegression(), X[ok], y[ok], cv=5, scoring="r2").mean()
            if r2 >= R2_MIN:
                modele = LinearRegression().fit(X[ok], y[ok])
                Xc = data.loc[cibles, PREDICTEURS]
                okc = Xc.notna().all(axis=1)
                valeur[okc[okc].index] = modele.predict(Xc[okc])
                methode = "régression"

    # 2) Sinon (ou si les prédicteurs manquent) : médiane pour une quantité, valeur la plus fréquente pour un code
    if type_col == "quantitative":
        valeur = valeur.fillna(y.median())
        valeur = valeur.clip(y.min(), y.max())
        if (y % 1 == 0).all():
            valeur = valeur.round()
        methode = methode or "médiane"
    else:
        valeur = valeur.fillna(y.mode().iloc[0])
        if type_col == "ordinale":
            modalites = np.sort(y.unique())
            plus_proche = np.abs(valeur.to_numpy()[:, None] - modalites).argmin(axis=1)
            valeur = pd.Series(modalites[plus_proche], index=valeur.index)
        methode = methode or "plus fréquent"

    data.loc[cibles, col] = valeur
    return {
        "colonne": col, "type": type_col, "methode": methode, "R2_cv": r2,
        "nb_imputes": int(cibles.sum()),
        "moyenne_avant": y.mean() if type_col == "quantitative" else np.nan,
        "moyenne_apres": data.loc[pop, col].mean() if type_col == "quantitative" else np.nan,
    }

def propager_nettoyage(data):
    """
    Rejoue les règles des nettoyer_* : une colonne porte qu'on vient d'imputer fixe ses colonnes dépendantes
    (ex. regime_passe imputé à 2 -> regime_nb_* = 0, durées non applicables). Ne remplit que des NaN.
    """
    data = data.copy()
    for fonction in FONCTIONS_NETTOYAGE:
        bloc = fonction(data)
        colonnes = [c for c in bloc.columns if c != "NOIND"]
        data[colonnes] = bloc[colonnes]
    return data

def imputer_groupe(data):
    """
    Impute un groupe d'âge (toutes les colonnes présentes, même vides).
    Phase 1 : colonnes portes, puis on rejoue les règles de déduction.
    Phase 2 : toutes les autres colonnes qui ont entre 0 et SEUIL_TROP_NA % de NaN.
    """
    data = data.copy()
    journal = []

    # Phase 1 : portes
    pct = pourct_na(data)
    portes = [c for c in COLONNES_PORTES if c in data.columns and 0 < pct[c] <= SEUIL_TROP_NA]
    for col in portes:
        ligne = imputer_colonne(data, col)
        if ligne:
            journal.append({"phase": 1, **ligne})
    data = propager_nettoyage(data)

    # Phase 2 : le reste
    pct = pourct_na(data)
    a_imputer = [c for c in pct[(pct > 0) & (pct <= SEUIL_TROP_NA)].index
                 if not est_exclue(c) and c not in PAIRES_DERIVEES]
    for col in a_imputer:
        ligne = imputer_colonne(data, col)
        if ligne:
            journal.append({"phase": 2, **ligne})

    # colonnes jumelles : dérivées de leur colonne source
    for col, (source, facteur) in PAIRES_DERIVEES.items():
        if col in data.columns and source in data.columns:
            masque = data[col].isna() & connue(data[source])
            data.loc[masque, col] = data.loc[masque, source] * facteur
            if masque.any():
                journal.append({"phase": 2, "colonne": col, "type": "quantitative",
                                "methode": f"= {source} x {facteur:.3g}", "R2_cv": np.nan,
                                "nb_imputes": int(masque.sum()), "moyenne_avant": np.nan, "moyenne_apres": np.nan})
    return data, pd.DataFrame(journal)

def construire_tables(dataframe):
    """
    Renvoie (dfs_na, dfs_imp, resume, journal, non_imputees)
    dfs_na  : par groupe d'âge, NaN conservés (sauf ceux qu'on a pu déduire), toutes les lignes
    dfs_imp : par groupe d'âge, lignes rares supprimées puis imputation
    """
    groupes = sep_age(nsp_refus_en_nan(dataframe), forme_return="dict")
    dfs_na, dfs_imp, resume, journaux, non_imputees = {}, {}, [], [], []

    for nom, data in groupes.items():
        dfs_na[nom] = supp_col_vides(data)

        reduite, info = supprimer_lignes_rares(data)

        imp, journal = imputer_groupe(reduite)
        imp = supp_col_vides(imp)
        dfs_imp[nom] = imp
        journal.insert(0, "groupe", nom)
        journaux.append(journal)

        data = dfs_na[nom]
        pct_avant = pourct_na(data)
        for col in imp.columns[imp.isna().any()]:
            if est_exclue(col):
                raison = "exclue de l'imputation"
            elif pct_avant[col] > SEUIL_TROP_NA:
                raison = f"plus de {SEUIL_TROP_NA} % de NaN"
            elif len(journal) and col in set(journal["colonne"]):
                raison = "NaN restants : pas de valeur sûre (porte inconnue ou population)"
            else:
                raison = "trop peu de valeurs connues ou hors population concernée"
            non_imputees.append({"groupe": nom, "colonne": col, "nb_na": int(imp[col].isna().sum()),
                                 "pct_na": round(imp[col].isna().mean() * 100, 1), "raison": raison})

        resume.append({
            "groupe": nom, "lignes": len(data), "seuil_colonne_%": info["seuil"],
            "colonnes_pour_suppression": len(info["colonnes"]),
            "lignes_supprimees": info["nb_lignes"], "lignes_supprimees_%": round(info["pct_lignes"], 1),
            "colonnes_imputees": len(journal), "NaN_dfs_na": int(data.isna().sum().sum()),
            "NaN_dfs_imp": int(imp.isna().sum().sum()),
        })

    return (dfs_na, dfs_imp, pd.DataFrame(resume).set_index("groupe"),
            pd.concat(journaux, ignore_index=True), pd.DataFrame(non_imputees))

def population_applicable(data, col):
    """Masque des lignes pour lesquelles la question `col` a un sens (hors non applicable)."""
    if col.startswith("poids_modifalim_"):
        porte, valeurs = "poids_modifalim", [1]
    elif col in POPULATION_APPLICABLE:
        porte, valeurs = POPULATION_APPLICABLE[col]
    else:
        return data[col].ne(NA_APPL)
    if porte not in data.columns:
        return pd.Series(False, index=data.index)
    return data[porte].isin(valeurs) & data[col].ne(NA_APPL)

#####################################################
# Fonctions de visualisation
# Toutes acceptent un dict de DataFrames (dfs_na, dfs_imp) ou un DataFrame seul.
#####################################################
BINS_NA = [-0.01, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100.01]
LABELS_NA = ["0", "]0-10]", "]10-20]", "]20-30]", "]30-40]", "]40-50]",
             "]50-60]", "]60-70]", "]70-80]", "]80-90]", "]90-99]", "]99-100]"]

def _vers_dict(x, sep=False, nom="Données"):
    """dict -> tel quel ; DataFrame -> séparé par âge si sep=True, sinon {"Données": x}."""
    if isinstance(x, dict):
        return x
    return sep_age(x, forme_return="dict") if sep else {nom: x}

def _comptage_na(data):
    """Nombre de colonnes par tranche de % de NaN."""
    pct = data.isna().mean() * 100
    return pd.cut(pct, bins=BINS_NA, labels=LABELS_NA, include_lowest=True).value_counts(sort=False)

def afficher_info_colonnes(dfs, sep=False):
    '''
    Affiche les colonnes contenant des NaN, triées par % de NaN décroissant.
    dfs : dict (dfs_na, dfs_imp) ou DataFrame (séparé par âge si sep=True).
    '''
    for nom, data in _vers_dict(dfs, sep).items():
        data = data.copy()

        colonnes_vides = data.columns[(data.isna() | data.eq(NA_APPL)).all()]
        data = data.drop(columns=colonnes_vides)

        infos = []
        for colonne in data.columns:
            nb_na = data[colonne].isna().sum()
            if nb_na > 0:
                infos.append({
                    "colonne": colonne,
                    "valeurs_uniques": data[colonne].nunique(),
                    "nb_na": nb_na,
                    "pct_na": nb_na / len(data) * 100
                })

        infos = sorted(infos, key=lambda x: x["pct_na"], reverse=True)

        print(f"--- {nom} ---")
        print(f"{len(colonnes_vides)} colonne(s) vide(s) à supprimer")

        for info in infos:
            print(
                f"Colonne '{info['colonne']}' : "
                f"{info['valeurs_uniques']} valeurs uniques, "
                f"{info['pct_na']:.2f}% de valeurs manquantes"
            )

def plot_diff_na(dfs1, libelle1, dfs2, libelle2):
    """
    Compare la répartition des colonnes selon leur % de NaN entre deux jeux de données.
    Deux dicts (ex. dfs_na, dfs_imp) -> un graphique par groupe d'âge. Deux DataFrames -> un graphique.
    Barres côte à côte, avec le nombre de colonnes affiché au-dessus.
    """
    d1, d2 = _vers_dict(dfs1), _vers_dict(dfs2)
    groupes = [g for g in d1 if g in d2]
    if not groupes:
        raise ValueError("Aucun groupe en commun entre les deux jeux de données.")

    x = np.arange(len(LABELS_NA))
    largeur = 0.4

    _fig, axes = plt.subplots(len(groupes), 1, figsize=(12, 5 * len(groupes)), squeeze=False)

    for ax, g in zip(axes[:, 0], groupes):
        c1 = _comptage_na(d1[g]).values
        c2 = _comptage_na(d2[g]).values

        barres1 = ax.bar(x - largeur / 2, c1, width=largeur, color="red", alpha=0.75,
                         edgecolor="black", label=libelle1)
        barres2 = ax.bar(x + largeur / 2, c2, width=largeur, color="blue", alpha=0.75,
                         edgecolor="black", label=libelle2)

        # valeurs au-dessus des barres (on n'affiche pas les 0 pour ne pas surcharger)
        for barres in (barres1, barres2):
            ax.bar_label(barres, labels=[int(b.get_height()) if b.get_height() > 0 else "" for b in barres],
                         padding=2, fontsize=8)

        ax.set_title(f"{g} : répartition des colonnes selon le % de valeurs manquantes")
        ax.set_xlabel("Pourcentage de valeurs manquantes (NaN)")
        ax.set_ylabel("Nombre de colonnes")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS_NA, rotation=45, ha="right")
        ax.margins(y=0.12)             # laisse de la place pour les valeurs au-dessus des plus hautes barres
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_age_distribution(dfs):
    """
    Répartition des personnes par tranche d'âge, groupes empilés.
    dfs : dict {"Adultes", "Enfants", "Nourrissons"} ou DataFrame complet (séparé avec sep_age).
    """
    dfs = _vers_dict(dfs, sep=True)

    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = ['0-11 mois', '1-3 ans', '4-6 ans', '7-10 ans', '11-14 ans',
              '15-17 ans ', '18-44 ans ', '45-64 ans', '65-79 ans ']
    style = {
        "Adultes": ("blue", "Adultes (18 ans et plus)"),
        "Nourrissons": ("green", "Nourrissons (moins de 3 ans)"),
        "Enfants": ("orange", "Enfants (3 à 17 ans)"),
    }

    x = np.arange(len(labels))
    cumul = np.zeros(len(labels))          # hauteur déjà occupée par les groupes précédents

    plt.figure(figsize=(10, 6))
    for nom, data in dfs.items():
        couleur, legende = style.get(nom, (None, nom))
        comptage = pd.cut(data["tage_PS"], bins=bins, labels=labels).value_counts(sort=False).values
        plt.bar(x, comptage, bottom=cumul, color=couleur, alpha=0.75,
                edgecolor='black', label=legende)
        cumul += comptage

    plt.xlabel("Tranche d'âge")
    plt.ylabel("Nombre de personnes")
    plt.title("Répartition des personnes par tranche d'âge")
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.show()

def plot_missing_values_distribution(dfs, sep=False):
    """
    Nombre de colonnes selon leur % de NaN, groupes côte à côte.
    dfs : dict (dfs_na, dfs_imp) ou DataFrame (séparé par âge si sep=True).
    Les colonnes vides de chaque table sont supprimées avant le calcul.
    """
    groupes = {nom: supp_col_vides(data) for nom, data in _vers_dict(dfs, sep).items()}
    comptages = {nom: _comptage_na(data) for nom, data in groupes.items()}

    n = len(comptages)
    x = np.arange(len(LABELS_NA))
    largeur = 0.8 / n

    plt.figure(figsize=(12, 6))
    for k, (nom, comptage) in enumerate(comptages.items()):
        plt.bar(x + (k - (n - 1) / 2) * largeur, comptage.values,
                width=largeur, alpha=0.75, label=nom)

    plt.title("Nombre de colonnes selon leur pourcentage de valeurs manquantes")
    plt.xlabel("Pourcentage de valeurs manquantes")
    plt.ylabel("Nombre de colonnes")
    plt.xticks(x, LABELS_NA, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
#####################################################
#Paramètres et constantes pour l'imputation
#####################################################

# ---------------------------------------------------------------- paramètres
NA_APPL = -1                        # code "non applicable"
SEUIL_TROP_NA = 20                  # % de NaN au-delà duquel on ne touche pas à la colonne
SEUILS_LIGNES = [0.5, 1, 1.5, 2]    # seuils de NaN par colonne testés pour supprimer des lignes
PLAFOND_LIGNES = 5                  # % max de lignes supprimées dans un groupe d'âge (donc aussi au total)
R2_MIN = 0.3                        # R² (validation croisée) minimal pour garder la régression
MIN_REGRESSION = 50                 # nombre minimal de valeurs connues pour essayer une régression
MIN_CONNUS = 10                     # en dessous, trop peu de valeurs connues pour imputer quoi que ce soit
PREDICTEURS = ["sex_PS", "tage_PS", "taille", "poids"]   # colonnes complètes utilisées par la régression


# ---------------------------------------------------------------- Ensembles des fonctions de nettoyage
FONCTIONS_NETTOYAGE = [nettoyer_fume, nettoyer_regime, nettoyer_poids, nettoyer_femmes, nettoyer_profession, nettoyer_divers]

# ---------------------------------------------------------------- codes "Ne sait pas" / "Refus" (formats de la notice)
CODES_NSP_REFUS = {
    "diplome_interv": [13, 14],
    "atrav_interv": [3],
    "trav_nuit_interv": [5],
    "PCS_8cl_interv": [9, 10],
    "PCS_4cl_interv": [5],
    "tps_travail_interv": [3],
    "vacances_interv": [3, 4],
    "lien_interv_PR": [12],
    "lien_PS_PR": [12],
    "diplome_PR": [13, 14],
    "atrav_PR": [3],
    "PCS_8cl_PR": [9, 10],
    "PCS_4cl_PR": [5],
    "tps_travail_PR": [3],
    "stat_log_2cl": [3],
    "soins": [3],
    "situ_fin_3cl": [4],
    "revenu": [14, 15],
    "intoall_confirm_med": [3],
    "regime_passe": [3],
    "regime_nb_2dernann": [6],
    "regime_nb_anter2dernann": [6],
    "poids_modif": [5],
    "poids_perception": [4],
    "nb_prise_10kg": [5],
    "menopause": [3],
    "enceinte": [3],
}

# ---------------------------------------------------------------- types de colonnes
# Quantitatives = colonnes sans format dans la notice. Toute colonne non listée est "codée sans ordre".
COLONNES_QUANTITATIVES = [
    "taille", "poids", "IA_score", "regime_duree_sem", "regime_duree_mois", "poids_anndern",
    "poidsmax", "poidsmin", "age_poidsmax", "age_poidsmin", "enceinte_nbmois", "allaite_nbsem",
    "nb_cigarettes_jour", "nb_cigarettes_sem", "nb_cigares_jour", "nb_cigares_sem",
    "nb_pipes_jour", "nb_pipes_sem", "fume_age_debut", "fume_age_arret",
]
COLONNES_ORDINALES = ["revenu", "RUC_4cl", "nb_prise_10kg", "regime_nb_2dernann", "regime_nb_anter2dernann",
                      "diplome_interv", "diplome_PR"]

# Colonnes "portes" : elles décident si les questions suivantes ont un sens (ex. regime_passe -> regime_nb_*)
COLONNES_PORTES = ["fume", "regime_passe", "poids_modif", "poids_modifalim", "menopause", "enceinte", "allaite"]
COLONNES_EXCLUES = ["NOMEN", "NOIND", "zae", "fpc1", "fpc2", "fpc3", "imc", "statnut", "regime_type"]

def est_exclue(col):
    return col in COLONNES_EXCLUES or col.startswith("pond_")

def type_colonne(col):
    if col in COLONNES_QUANTITATIVES:
        return "quantitative"
    if col in COLONNES_ORDINALES:
        return "ordinale"
    return "codée"

# ---------------------------------------------------------------- population concernée par chaque question
# colonne : (colonne "porte", valeurs de la porte pour lesquelles la question est posée)
POPULATION_APPLICABLE = {
    "fume_age_debut": ("fume", [1, 2, 3]), "fume_age_debut_nsp": ("fume", [1, 2, 3]),
    "fume_age_arret": ("fume", [3]), "fume_age_arret_nsp": ("fume", [3]),
    "allaite_nbsem": ("allaite", [1]), "enceinte_nbmois": ("enceinte", [1]),
}
for tabac in ["cigarettes", "cigares", "pipes"]:
    for suffixe in ["jour", "sem", "nsp"]:
        POPULATION_APPLICABLE[f"nb_{tabac}_{suffixe}"] = ("fume", [1, 2])
for col in ["regime_nb_2dernann", "regime_nb_anter2dernann", "regime_duree_sem", "regime_duree_mois", "regime_duree_nsp"]:
    POPULATION_APPLICABLE[col] = ("regime_passe", [1])

# colonne dérivée : (colonne source, facteur)  ->  dérivée = source * facteur
PAIRES_DERIVEES = {
    "nb_cigarettes_sem": ("nb_cigarettes_jour", 7),
    "nb_cigares_sem": ("nb_cigares_jour", 7),
    "nb_pipes_sem": ("nb_pipes_jour", 7),
    "regime_duree_mois": ("regime_duree_sem", 12 / 52),
}



#####################################################
#  Main
#####################################################
def nettoyer_tout(dossier_tables="tables", info=False):
    """
    Lance tout le pipeline de nettoyage de description-indiv.csv.
    Renvoie (dfs_na, dfs_imp, resume, journal, non_imputees).
    """
    df = importer_donnees(dossier_tables=dossier_tables, info=info).copy()
    df = nettoyer_str(df)

    for func in FONCTIONS_NETTOYAGE:   # la liste existe déjà plus haut
        d_nett = func(df, bilan=False)
        colonnes = [c for c in d_nett.columns if c != "NOIND"]
        df[colonnes] = d_nett[colonnes]

    # Mesuré / déclaré : un indicateur à la place de 4 colonnes presque vides
    df["taille"] = np.select([df["taille"].notna(), df["taille_m"].notna(), df["taille_d"].notna()],
                             [df["taille"], df["taille_m"], df["taille_d"]],
                             default=np.nan)
    df["poids"] = np.select([df["poids"].notna(), df["poids_m"].notna(), df["poids_d"].notna()],
                            [df["poids"], df["poids_m"], df["poids_d"]],
                            default=np.nan)
    df = df.drop(columns=["taille_m", "taille_d", "poids_m", "poids_d"]).copy()
    df = supp_col_vides(df)

    return construire_tables(df)   # dfs_na, dfs_imp, resume, journal, non_imputees


if __name__ == "__main__":         # ne s'exécute que si on lance `python nettoyage.py`
    dfs_na, dfs_imp, resume, journal, non_imputees = nettoyer_tout()
    print(resume)
