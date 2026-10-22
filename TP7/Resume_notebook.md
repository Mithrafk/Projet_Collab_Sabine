### A modifier :
- repasser les graphs dans l'IA pour rendre jolie couleurs, transparence ...
Point à modifier :
- dans le journal (et en général dans tout le doc), il faut faire les moyenne sans les na et sans les -1 (rajouté par NA_APPL), sinon ca fausse ou rend NaN (comme le journal actuel)
- Je voudrais créer un 3ieme df, df_strict qui ne contiendrais aucun NA. Pour cela, il faudrait partir de dfs_imp, et supprimer soit les lignes, soit les colonnes qui contiennent les NA restant. Il faut decider de la méthode pour choisir entre supprimer ligne/colonnes (suivant l'impact sur l'ensemble des données)

### Liste des fonctions :
- **def importer_donnees(info=False):**
    '''
    Importe les données des différents fichiers CSV et les fusionner.
    Ne retenir que les lignes communes aux 4 fichiers.
    '''

- **def plot_diff_na(df1, df1_libelle, df2, df2_libelle):** 
    """Compare la répartition des colonnes selon leur pourcentage de valeurs manquantes."""

- **def sep_age(dataframe, forme_return="liste"):**  
    '''  
    Sépare le dataframe en trois sous-dataframes selon l'âge : adultes, nourrissons et enfants.
    Peut rentre 3 sous-dataframes ou un dictionnaire de dataframes selon le paramètre forme_return.
    '''  

- **def supp_col_vides(dataframe):** 
    '''  
    supprime les colonnes vides (toutes les valeurs sont NaN ou NA_APPL) du dataframe.  
    '''  

- **def afficher_info_colonnes(dataframe, sep=False):**  
    '''  
    Affiche les informations sur les colonnes contenant des valeurs manquantes,  
    triées par pourcentage de valeurs manquantes décroissant.  
    Info regroupées par tranche d'âge si sep=True, sinon sur l'ensemble du dataframe.  
    '''  

- **def plot_missing_values_distribution(dataframe, sep=False):**  
    '''  
    Afficher graphiquement la distribution des na dans les colonnes.  
    sep : bool - définir si on veut séparer entre les âges ou non.  
    Si on sépare par âge, on supprime les colonnes vides pour chaque sous-table. Sinon, on ne supprime que les colonnes   vides de la table entière.  
    '''  

- **def remplir(data, masque, colonnes, valeur):**  
    '''Écrit `valeur` dans `colonnes` pour les lignes de `masque`, UNIQUEMENT là où la case est vide (NaN).  
    On n'écrase donc jamais une vraie réponse.'''  

- **def connue(serie):**  
    '''Vrai là où on a une vraie valeur (ni NaN, ni non applicable)'''  

- **def sans_non_applicable(data):**  
    '''Copie avec NaN à la place de NA_APPL : à utiliser pour moyennes, régressions, imputations'''  

- **def bilan_na(avant, apres):**  
    '''Tableau NaN avant / NaN après / non applicable, colonne par colonne'''  

- **def nettoyer_`*`(dataframe, bilan=False):**  
    '''  
    Nettoie les données des colonnes liées aux *.  
    bilan = True donne en 2nd retrun le pd bilan_na  
    '''  
    **`*`** = {fume, regime, poids, femmes, profession, divers}  

$---$

Les fonctions d'imputations n'ont pas encore été vérifiées en détails, mais j'ai passé 10/20min a regarder et claude a bien compris ma lgique et je n'ai pas trouvé d'erreur grossières, je reprend ca demain


