import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import ARDRegression

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

from zipfile import ZipFile

import datetime
import json
import re
# import shutil

from tqdm import tqdm

# import seaborn as sns

# import ipywidgets
import os

# import tensorflow as tf
# import tensorflow.keras as keras

# functions to extract, transform and load data from API and zip files to dataframes

class Etl:
    # basic class to extract files
    # - poids
    # - exercices
    # - calories
    
    #----------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, repo:str = None):
        if repo == None:
            self.repo = "/home/benjamin/Downloads/"  # répertoire par défaut pour aller chercher les différents fichiers
        else:
            self.repo = repo
            
    #----------------------------------------------------------------------------------------------------------------------------------            
    def _extract_poids(self):
        """Retourne la dataframe de toutes les mesures de poids depuis le 1er Août 2020,
        (il peut y en avoir plusieurs par jour), avec MG% et BMR calculés. 
        Drope les rows sans Masse_Totale ou Masse_Grasse

        Raises:
            NameError: _description_
            NameError: _description_

        Returns:
            dataframe : toutes les mesures avec Masse_Totale et Masse_Grasse depuis 1er Août 2020
        """
        
        downloads_rep = self.repo
        liste_downloaded_files = os.listdir(downloads_rep)
        
        if len(liste_downloaded_files) == 0:
            raise NameError(f"Aucun fichier n'est présent dans {downloads_rep}")
        
        pattern_poids = "^data_BEN_[\d]+[.]zip"
        eng = re.compile(pattern_poids)
        liste_zip_poids = []
        liste_mtime_zips = []

        for f in liste_downloaded_files:
            m = eng.search(f)
            if m:  # si on trouve un data_BEN_dddd.zip, on note le nom du fichier et le temps de modif
                filename = m.group(0)
                liste_zip_poids.append(filename)
                mtime = os.stat(downloads_rep + f).st_mtime
                liste_mtime_zips.append(mtime)
                
        if len(liste_zip_poids) == 0:
            raise NameError(f'Aucun fichier de type data_BEN_xxxx.zip contenant \
                les données poids ne figure dans le répertoire {downloads_rep}')
            
        # print(liste_zip_poids)

        idx = np.argmax(liste_mtime_zips)
        filename_poids = liste_zip_poids[idx]

        print(f"\nLe fichier poids le plus récent est : {filename_poids}, parmi :")
        for f in liste_zip_poids:
            print(f"{f}")

        # Récupère données POIDS

        withings_filename = self.repo + filename_poids  # fichier zip le plus récent de HealthMate Withings

        with ZipFile(withings_filename, 'r') as weight_zip:
            weight_csv = weight_zip.extract('weight.csv', path=self.repo)

        # extrait la dataframe poids ----------------

        colnames = ['Date', 'Poids (kg)', 'Gras (kg)', 'Masse osseuse (kg)', 'Masse musculaire (kg)', 'Hydratation (kg)']

        df_weight = pd.read_csv(weight_csv, usecols=colnames)
        os.remove(weight_csv)
        df_weight.rename(columns = {'Poids (kg)' : 'Masse_Totale' , 
                                    'Gras (kg)' : 'Masse_Grasse',
                                    'Masse osseuse (kg)' : 'Masse_Osseuse',
                                    'Masse musculaire (kg)' : 'Masse_Musculaire',
                                    'Hydratation (kg)' : 'Masse_Hydrique'
                                    }, 
                        inplace=True)

        # transforme le champ str de Date en datetime object
        date_format = '%Y-%m-%d %H:%M:%S'
        df_weight['Date'] = df_weight['Date'].apply(lambda x : datetime.datetime.strptime(x, date_format).date())

        # garde les data au delà du 1er Août 2020 seulement
        start_date = datetime.date(2020, 8, 1)
        df_weight = df_weight[df_weight['Date'] >= start_date]

        # drope les rows sans données Masse_Totale ou Masse_Grasse
        subset = ['Masse_Totale', 'Masse_Grasse']
        df_weight.dropna(subset = subset, inplace=True)

        # calcule MG% et BMR suivant Katch Mac Ardle
        df_weight['MG%'] = df_weight['Masse_Grasse'] / df_weight['Masse_Totale']
        df_weight['BMR'] = 370 + 21.6 * (df_weight['Masse_Totale'] - df_weight['Masse_Grasse'])  # Katch Mac Ardle

        df_weight.sort_index(inplace=True)
        
        return df_weight
    
    #----------------------------------------------------------------------------------------------------------------------------------
    def _extract_food(self):
        """extrait et retourne une dataframe avec kcalories
        """
        
        # recherche fichiers FOOD : format File-Export-YYYY-MM-DD-to-YYYY-MM-DD.zip
        # https://www.myfitnesspal.com/reports
        # www.myfitnesspal.com ==> reports > export data ==> File-Export-date1-to-date2.zip

        # identifie les fichiers au bon format
        pattern_food = "^File-Export-[\d]{4}-[\d]{2}-[\d]{2}-to-[\d]{4}-[\d]{2}-[\d]{2}.*[.]zip"
        eng = re.compile(pattern_food)
        liste_zip_food = []
        liste_mtime_zips = []
        
        downloads_rep = self.repo
        liste_downloaded_files = os.listdir(downloads_rep)

        for f in liste_downloaded_files:
            m = eng.search(f)
            if m:  # si on trouve un File-Export-....zip, on note le nom du fichier et le temps de modif
                filename = m.group(0)
                liste_zip_food.append(filename)
                mtime = os.stat(downloads_rep + f).st_mtime
                liste_mtime_zips.append(mtime)
                
        if len(liste_zip_food) == 0:
            raise NameError(f'Aucun fichier de type File-Export-xxxx.zip contenant \
                les données food ne figure dans le répertoire {downloads_rep}')
        
        # identifie le fichier le plus récent dans la liste des fichiers éligibles
        idx = np.argmax(liste_mtime_zips)
        filename_food = liste_zip_food[idx]

        print(f"\nLe fichier food le plus récent est : {filename_food}, parmi")

        for f in liste_zip_food:
            print(f"{f}")
            
        # Récupère données FOOD
        mfp_filename = downloads_rep + filename_food

        with ZipFile(mfp_filename, 'r') as food_zip:
            output_dir = food_zip.namelist()
            target = 'Nutrition-Summary'
            for l in output_dir:
                if l[:len(target)] == target:
                    food_csv = food_zip.extract(l, path=self.repo)
                    break
                
        # construit la dataframe food
        colnames = ['Date', 'Meal', 'Calories', 'Fat (g)', 'Carbohydrates (g)', 'Protein (g)']
        df_food = pd.read_csv(food_csv, usecols=colnames)
        os.remove(food_csv)
        df_food.rename(columns = {'Fat (g)' : 'Lipides' , 'Carbohydrates (g)' : 'Glucides', 'Protein (g)' : 'Proteines'}, inplace=True)

        # transforme le champ str de Date en datetime object
        date_format = '%Y-%m-%d'
        df_food['Date'] = df_food['Date'].apply(lambda x : datetime.datetime.strptime(x, date_format).date())

        # ne garde que les données au delà du 1er Août 2020
        start_date = datetime.date(2020, 8, 1)
        df_food = df_food[df_food['Date'] >= start_date]
        df_food.dropna(inplace=True)
        df_food.sort_index(inplace=True)
        
        return df_food
    
    #----------------------------------------------------------------------------------------------------------------------------------
    def _extract_exos(self):
        """Extrait exercices du fichier Polar, concatène avec fichier exos_persos, retourne une dataframe
        """
        
        #---------------------------------------------------------------------------------
        def extract_data_training(exo_dict):
            # utility fonction pour récupérer : date, durée, type exercice et calories dépensées
            # exo_dict est au format JSON dans l'archive Polar
            
            # récupère date
            pattern = '2(\d){3,3}-(\d){2,2}-(\d){2,2}'  # on cherche une date du type 2xxx-yy-zz
            p = re.compile(pattern)
            m = p.search(exo_dict.get('startTime'))
            if m: 
                time_format = '%Y-%m-%d'
                exo_date = datetime.datetime.strptime( exo_dict.get('startTime')[:10], time_format ).date()
            
            # récupère durée
            pattern = '(\d)+[.]?(\d)*'  # on cherche une durée du type xxxx.yyyy
            p = re.compile(pattern)
            m = p.search(exo_dict.get('duration'))
            if m:
                exo_duration = float(m.group())
            
            # récupère type activité et calories dépensées
            d = exo_dict.get('exercises')[0]
            exo_type = d.get('sport')
            if d.get('kiloCalories'):
                exo_cals = float(d.get('kiloCalories'))
            else:
                exo_cals = 0.0
            
            return exo_date, exo_duration, exo_type, exo_cals
        #---------------------------------------------------------------------------------
        
        # recherche fichiers EXO par POLAR : format polar-user-data-export.zip
        # https://account.polar.com/#export ==> polar-user-data-export-xxxxx.zip

        # recherche et construit la liste des fichiers éligibles (= au bon format)
        pattern_polar = "^polar-user-data-export_.+[.]zip"
        eng = re.compile(pattern_polar)
        liste_zip_polar = []
        liste_mtime_zips = []
        
        downloads_rep = self.repo
        liste_downloaded_files = os.listdir(downloads_rep)

        for f in liste_downloaded_files:
            m = eng.search(f)
            if m:  # si on trouve un polar-user-data-export_....zip, on note le nom du fichier et le temps de modif
                filename = m.group(0)
                liste_zip_polar.append(filename)
                mtime = os.stat(downloads_rep + f).st_mtime
                liste_mtime_zips.append(mtime)
                
        if len(liste_zip_polar) == 0:
            raise NameError(f'Aucun fichier de type polar-user-data-export_xxxx.zip contenant \
                les données exercices de Polar ne figure dans le répertoire {downloads_rep}')
            
        # prend le fichier éligible le plus récent
        idx = np.argmax(liste_mtime_zips)
        filename_polar = liste_zip_polar[idx]
        print(f"\nLe fichier exercices le plus récent est : {filename_polar}, parmi :")
        for f in liste_zip_polar:
            print(f"{f}")
        
        # Récupère les données EXERCICE dans ce fichier
        polar_filename = self.repo + filename_polar  # données de Polar

        with ZipFile(polar_filename, 'r') as polar_zip:
            output_dir = polar_zip.namelist()
            
            target = 'training-session'
            dict_all_exos = {}
            id_training = 0
            
            for i, enr_name in enumerate(tqdm(output_dir)): # on parcourt la liste des archives
                if enr_name[:len(target)] == target:  # si c'est un enregistrement d'une session de training, on traite
                    enr_json = polar_zip.extract(enr_name, path=self.repo)   # extraction du json
                    with open(enr_json, 'r') as f:
                        exo_dict = json.load(f)
                        exo_date, exo_duration, exo_type, exo_cals = extract_data_training(exo_dict)
                        dict_all_exos[id_training] = [ exo_date, exo_duration, exo_type, exo_cals ]
                        id_training += 1
                    os.remove(enr_json)
            
            df_exos = pd.DataFrame.from_dict(dict_all_exos, orient='index', columns=['exo_date', 'exo_duree', 'exo_type', 'exo_cals_bruts']).sort_index()
            
        # Ajout à la main des exos depuis le xx Janvier inclus pour palier à la fréquence de rafraîchissement de l'extraction Polar

        # récupère fichier brut .csv
        exos_persos_filename = "/home/benjamin/Folders_Python/Weight_imports/Exos_Persos.csv"
        df_exos_persos = pd.read_csv(exos_persos_filename)

        # convertit la colonne Jour en datetime objects en colonne Date
        pattern_date = '[\d]{2}/[\d]{2}/[\d]{2}'
        p = re.compile(pattern_date)
        day_format = "%d/%m/%y"
        df_exos_persos['exo_date'] = df_exos_persos['Jour'].apply( lambda x : datetime.datetime.strptime(p.search(x).group(0), day_format).date() )

        # drope la colonne Jour
        df_exos_persos.drop(columns=['Jour'], inplace=True)

        # met au bon format la duree de l'exercice (minutes => secondes)
        df_exos_persos['exo_duree'] = df_exos_persos['Duree'] * 60
        df_exos_persos.drop(columns=['Duree'], inplace=True)
        
        # enfin, concatène avec le fichier issu de Polar
        df_exos_total = pd.concat( [df_exos, df_exos_persos], axis=0 )
        df_exos_total = df_exos_total.sort_values(by=['exo_date'])
        df_exos_total = df_exos_total.dropna()
        df_exos_total = df_exos_total.reset_index(drop=True)
        
        # maintenant rajoute des zeros aux jours où il n'y a pas eu entrainement
        # sinon les calculs de moyennes seront faux !
        df_exos_total.set_index('exo_date', inplace=True)
        
        start_date = min(df_exos_total.index)
        end_date = max(df_exos_total.index)
        add_indices = []
        d = start_date

        # méthode moche pour construire la liste des indices manquant. Il doit y avoir plus beau avec DateIndex etc. mais f**k it.
        while d <= end_date:
            if not (d in df_exos_total.index):
                add_indices.append(d)
            d = d + datetime.timedelta(days=1)
            
        add_df = pd.DataFrame(index=add_indices, columns=df_exos_total.columns).fillna(0)
        add_df.index.name='exo_date'
        df_exos_total = pd.concat([df_exos_total, add_df]).sort_index()

        return df_exos_total
    
    #----------------------------------------------------------------------------------------------------------------------------------
    def extract_store_all(self):
        """extrait toutes les données, et les store dans trois *.csv
        """
        
        # process et store les 3 dataframes brutes
        df_weight_raw = self._extract_poids()
        df_food_raw = self._extract_food()
        df_exos_raw = self._extract_exos()
        
        file_weight_raw = os.getcwd() + "/data/raw/weight_raw.csv"
        file_food_raw = os.getcwd() + "/data/raw/food_raw.csv"
        file_exos_raw = os.getcwd() + "/data/raw/exos_raw.csv"
            
        with open(file_weight_raw, 'w') as f:
            df_weight_raw.to_csv(file_weight_raw)
            
        with open(file_food_raw, 'w') as f:
            df_food_raw.to_csv(file_food_raw)
            
        with open(file_exos_raw, 'w') as f:
            df_exos_raw.to_csv(file_exos_raw)
            
        # créé maintenant une dataframe globale avec les données agrégées par jour
        df_food = df_food_raw.groupby(['Date']).sum(numeric_only=True)
        df_weight = df_weight_raw.groupby(['Date']).mean()
        df_exos_total = df_exos_raw.groupby(['exo_date']).sum(numeric_only=True)
        df_all = pd.concat( [df_weight, df_food, df_exos_total], axis=1 ).sort_index()

        # remplace les NaN dans les colonnes exercice par 0.0 : jour sans exercice
        df_all['exo_duree'].fillna(0.0, inplace=True)
        df_all['exo_cals_bruts'].fillna(0.0, inplace=True)

        # drop les jours où il manque des données autres que les exercices
        subset = ['Masse_Totale', 'Masse_Grasse', 'Calories']
        df_all.dropna(subset = subset, inplace=True)
        
        #---- utility function ---
        def exo_cals_nets(bmr, exo_duree, exo_cals_bruts):
            if exo_cals_bruts > 0:
                ecn = exo_cals_bruts - bmr / (24*60*60) * exo_duree
            else:
                ecn = 0.0
            return ecn
        #-------------------------

        df_all['exo_cals_nets'] = np.where(df_all['exo_cals_bruts'] > 0, df_all['exo_cals_bruts'] - df_all['BMR'] / (24*60*60) * df_all['exo_duree'], 0 )
        df_all['Depense_cal_totale'] = df_all['BMR'] + df_all['exo_cals_nets']
        df_all['cal_deficit'] = df_all['Calories'] - (df_all['BMR'] + df_all['exo_cals_nets'])
        
        file_all_save = os.getcwd() + "/data/full_dataset.csv"

        with open(file_all_save, 'w') as f:
            df_all.to_csv(file_all_save)

            # store toutes les datatframes brutes
            file_all_save = os.getcwd() + "/data/full_dataset.csv"
            
        return df_weight_raw, df_food_raw, df_exos_raw, df_all
