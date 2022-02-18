import datetime
from datetime import date
import csv
from zipfile import ZipFile
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class GetData():
    
    """
    Basic class to to all the data extraction work :
    - extracts the two files
    - replaces manually entered numbers for total mass and fat mass by calculated averages if available
    - provides method to export processed data
    """
    
    def __init__(self, zip_file_name, manual_file_name):
        self.zip_file_name = zip_file_name
        self.manual_follow_up_file_name = manual_file_name
        self.daily_mass_measurements = []
        self.daily_data = []
    
    def extract(self):
        # extract data from the two files
        # first : data_BEN.zip
        
        root = os.getcwd() + '/'
        weight_file_path = root + 'tmp'

        with ZipFile(self.zip_file_name, 'r') as fichier_zip:
            print(f'Extraction fichier Zip Healthmate dans {weight_file_path}...')
            fichier_zip.extractall(path = weight_file_path)
            print(f'... Done')
        
        weight_file_csv_name = weight_file_path + '/weight.csv'
        fields = ['date', 
                  'MT', 
                  'MG']

        self.raw_from_zip = self.__extract_data(weight_file_csv_name, 
                                                fields,
                                                skip=1, 
                                                delimiter=',')  # gets a list of dictionnaries
        
        # 1/ self.raw_from_zip is a list of dictionnaries, that's the set of all mass measurements from Withings scale
        # there can be several measurements per day (and there usually are)
        
        # format is :
        # [ { 'date' : 'YYYY-MM-DD time', 'MT' : str of total mass, 'MG' : str of fat mass, None : ['','','']},
        #   { 'date' : 'YYYY-MM-DD time', 'MT' : str of total mass, 'MG' : str of fat mass, None : ['','','']},
        #   ....,
        #   { 'date' : 'YYYY-MM-DD time', 'MT' : str of total mass, 'MG' : str of fat mass, None : ['','','']},
        # ]
                
        # second : Suivi_Poids.csv
        
        fields = ['date', 
                  'Masse_Totale', 
                  'Masse_Grasse', 
                  'Calories_in', 
                  'Glucides', 'Lipides', 'Proteines', 
                  'Calories_Exercices_Brut', 'Duree_totale_exercices',
                  'C_Ex_Cardio_Brut', 'Duree_Ex_Cardio',
                  'C_Ex_Strength_Brut', 'Duree_Ex_Strength',
                  'Verif_cal', 'Verif_durees'
                  ]
        
        self.raw_from_manual = self.__extract_data(self.manual_follow_up_file_name, 
                                                   fields,
                                                   skip=2, 
                                                   delimiter=';')  # returns and stores a list of disctionnaries
              
        
        # 2/ self.raw_from_manual is a list of dictionnaries, with one set of data per day, from a *.csv follow-up file
        
        # format is :
        # [ { 'date' : 'DD-month_name-YYYY', 'Masse_Totale' : str of total mass, 'Masse_Grasse' : str of fat mass,
        #     'Calories_In' : str of kcals, 'Glucides' : str, 'Lipides' : str, 'Proteines' : str, 'Calories_Exercice_Brut': str,
        #     'Duree_total_exercices', 
        #     'C_Ex_Cardio_Brut': str, 'Duree_Ex_Cardio': str, 
        #     'C_Ex_Strength_Brut': str, 'Duree_Ex_Strength' : str
        #     'Verif_cal', 'Verif_durees' : str, None: ['', '']
        #   },
        # ....
        # ]
        
        # ------------------------------
        # -- data consolidation --------
        # ------------------------------
        
        # 1/ creates an attribute self.daily_mass_measurements, which is a list of dictionnaries, containing all the mass measurements
        # per day, non empty, in float format - created from self.raw_from_zip
        
        # format is :
        
        # [ { 'date' : date_object, 'mt' : list of total mass measurements of the day (floats), 
        #     'mg' : list of total mass measurements of the day (floats) },
        #   { 'date' : date_object, 'mt' : list of total mass measurements of the day (floats), 
        #     'mg' : list of total mass measurements of the day (floats) },
        #   ...
        # ]
        
        for cdict in self.raw_from_zip:
            cdate = self.__conv_to_date_num(cdict.get('date'))
            cmt = self.__conv_to_float(cdict.get('MT'))
            cmg = self.__conv_to_float(cdict.get('MG'))
            self.__add_daily_mass_measurement(cdate, cmt, cmg)
            
        # 2/ creates an attribute self.daily_data, which is a list of dictionnaries, containing all the measurements of the day
        # plus the calories data from the *.csv file (self.raw_from_manual)
        # if available in self.daily_mass_measurements, then the mass data in self.daily_data is overwritten by the averages
        # in self.daily_mass_measurements
        
        # format is :
        
        # [ { 'date' : date_object, 'masse_totale' : total mass (float), 'masse_grasse' : fat mass (float),
        #     'calories_in' : kcals (float), 'glucides' : float, 'lipides' : float, 'proteines' : float, 
        #     'calories_exercice': float, 'duree_totale_exercices' : float,
        #     'calories_cardio': float, 'duree_ex_cardio' : float
        #     'calories_strength': float, 'duree_ex_strength' : float },
        # ....
        # ]
        
        for cdict in self.raw_from_manual:
            cdate = self.__conv_to_date_str(cdict.get('date'))
            found, cmt, cmg = self.get_daily_mass_measurement(cdate)  # is the mass data in the Withings file ?
            if found:
                cmt = np.mean(cmt)
                cmg = np.mean(cmg)
            else:
                cmt = self.__conv_to_float(cdict.get('Masse_Totale'))
                cmg = self.__conv_to_float(cdict.get('Masse_Grasse'))
            cals_in = self.__conv_to_float(cdict.get('Calories_in', '0.0'))
            glu = self.__conv_to_float(cdict.get('Glucides', '0.0'))
            lip = self.__conv_to_float(cdict.get('Lipides','0.0'))
            prot = self.__conv_to_float(cdict.get('Proteines', '0.0'))
            cals_ex = self.__conv_to_float(cdict.get('Calories_Exercices_Brut', '0.0'))
            cals_card = self.__conv_to_float(cdict.get('C_Ex_Cardio_Brut', '0.0'))
            cals_str = self.__conv_to_float(cdict.get('C_Ex_Strength_Brut', '0.0'))
            duree_ex = self.__conv_to_float(cdict.get('Duree_totale_exercices', '0.0'))
            duree_card = self.__conv_to_float(cdict.get('Duree_Ex_Cardio', '0.0'))
            duree_strength = self.__conv_to_float(cdict.get('Duree_Ex_Strength', '0.0'))
            new_rec = dict([ ('date', cdate), 
                             ('masse_totale', cmt),
                             ('masse_grasse', cmg),
                             ('calories_in', cals_in),
                             ('glucides', glu),
                             ('lipides', lip),
                             ('proteines', prot),
                             ('calories_exercice', cals_ex),
                             ('calories_cardio', cals_card),
                             ('calories_strength', cals_str),
                             ('duree_totale_exercices', duree_ex),
                             ('duree_ex_cardio', duree_card),
                             ('duree_ex_strength', duree_strength)
                             ])
            self.daily_data.append(new_rec)            
                    
    def __add_daily_mass_measurement(self, cdate : datetime.datetime, cmt : float, cmg : float):
        # utility : add cmt, cmg to the self.daily_mass_measurements records, or create it if first time
        found = False
        for i, daily_m_dict in enumerate(self.daily_mass_measurements):
            if daily_m_dict.get('date') == cdate:
                found = True
                self.daily_mass_measurements[i]['mt'].append(cmt)
                self.daily_mass_measurements[i]['mg'].append(cmg)
        if not found:
            new_rec = dict([ ('date', cdate), ('mt', [cmt]) , ('mg', [cmg]) ])
            self.daily_mass_measurements.append(new_rec)
            
    def get_daily_mass_measurement(self, cdate : datetime.datetime):
        # access to list of daily mass measurements
        # returns :
        # (True, list of mt measurements, list of mg measurements) if exist
        # (False, 0.0, 0.0) if does not exist
        found = False
        daily_mt = 0.0
        daily_mg = 0.0
        
        for i, daily_m_dict in enumerate(self.daily_mass_measurements):
            if daily_m_dict.get('date') == cdate:
                found = True
                daily_mt = daily_m_dict.get('mt')
                daily_mg = daily_m_dict.get('mg')
                
        return found, daily_mt, daily_mg
    
    def get_window_daily_mass_measurements(self, start_date : datetime.datetime, end_date : datetime.datetime):
        # return a list of dictionnaries, being the extract of daily_mass_measurements between the two dates
        # format is, for the relevant dates :
        # [ { 'date' : date_object, 'mt' : list of total mass measurements of the day (floats), 
        #     'mg' : list of total mass measurements of the day (floats) },
        #   { 'date' : date_object, 'mt' : list of total mass measurements of the day (floats), 
        #     'mg' : list of total mass measurements of the day (floats) },
        #   ...
        # ]
        
        if start_date >= end_date:
            raise NameError('Requesting window with start date posterior to end date')
        
        duration = (end_date - start_date).days
        dates_list = [ start_date + datetime.timedelta(days = d) for d in range(duration) ]
        window = []
        
        for cdate in dates_list:
            found, daily_mt, daily_mg = self.get_daily_mass_measurement(cdate)
            if found:
                window.append( dict([ ('date',cdate), ('mt',daily_mt), ('mg', daily_mg) ]) )
                
        return window
    
    def get_daily_data(self, cdate : datetime.datetime):
        # access to list of daily data
        
        # returns :
        # (True, dict of dialy data if exist
        # (False, {}) if does not exist
        found = False
        ret_dict = {}
        
        for i, daily_m_dict in enumerate(self.daily_data):
            if daily_m_dict.get('date') == cdate:
                found = True
                ret_dict = self.daily_data[i]
                
        return found, ret_dict
    
    def get_window_daily_data(self, start_date : datetime.datetime, end_date : datetime.datetime):
        # return a list of dictionnaries, being the extract of the daily data between the two dates
        # format is, for the relevant dates :
        
        # [ { 'date' : date_object, 'masse_totale' : total mass (float), 'masse_grasse' : fat mass (float),
        #     'calories_in' : kcals (float), 'glucides' : float, 'lipides' : float, 'proteines' : float, 'calories_exercice': float,
        #     'calories_cardio': float, 'calories_strength': float },
        # ....
        # ]
        
        if start_date >= end_date:
            raise NameError('Requesting window with start date posterior to end date')
        
        duration = (end_date - start_date).days
        dates_list = [ start_date + datetime.timedelta(days = d) for d in range(duration) ]
        window = []
        
        for cdate in dates_list:
            found, cdict = self.get_daily_data(cdate)
            if found:
                window.append( cdict )
                
        return window
    
    def __extract_data(self,
                       filename, 
                       champs,
                       skip=2,
                       delimiter=','):
        """
        Helper function that reads a csv file, returns a list of dictionnaries
        Each dict is a line, ie a daily measurement, in the file.
        Skips the first line(s)
        
        Parameters :
        filename (str) : name of the *.csv file
        fieldnames (sequence) : sequence of the columns names
        skip (int) : number of lines to skip at the beginning of the file
        delimiter (str) : character used as a delimiter
        """

        data = []
        with open(filename, 
                  newline='',
                  encoding='ISO-8859-1'
                  ) as csvfile:
            fichier = csv.DictReader(csvfile, fieldnames=champs, delimiter=delimiter)
            for i in range(skip):  # skip <skip> lines at the beginning of the file
                next(fichier)
            for row in fichier:
                data.append(row)

        return data
    
    def __conv_to_date_str(self, date_string : str) -> datetime.date:
        """
        helper function that converts string outputs of dates, 
        with format "YY-month_name-day", and returns a date object
        from datetime.
        """

        dict_mois = {'août' : 8, 
                     'sept.' : 9,
                     'oct.' : 10,
                     'nov.' : 11,
                     'déc.' : 12,
                     'janv.' : 1,
                     'févr.' : 2,
                     'mars' : 3,
                     'avr.' : 4,
                     'mai' : 5,
                     'juin' : 6,
                     'juil.' : 7
                    }
        d = date_string.split(' ')[0]  # récupère la date en début de string : 2xxx-MM-DD
        d = d.split('-')  # récupère year, month, day

        # print(d)

        try:
            day = int(d[0])
        except ValueError:
            raise NameError('problème de format dans un champ date (jour)')

        try:
            year = 2000 + int(d[2])
        except ValueError:
            raise NameError('problème de format dans un champ date (année)')

        try:
            month = int(d[1])
        except ValueError:
            try:
                month = dict_mois.get(d[1])
            except ValueError:
                raise NameError('problème de format dans un champ date (mois)')

        date_object = datetime.date(year, month, day)

        return date_object
    
    def __conv_to_date_num(self, date_string : str) -> datetime.date:
        """
        helper function that converts a "YYYY-MM-DD" string into a date object
        """

        d = date_string.split(' ')[0]  # récupère la date en début de string : 2xxx-MM-DD
        d = d.split('-')  # récupère year, month, day

        # print(d)

        try:
            day = int(d[2])
        except ValueError:
            raise NameError('problème de format dans un champ date (jour)')

        try:
            year = int(d[0])
        except ValueError:
            raise NameError('problème de format dans un champ date (année)')

        try:
            month = int(d[1])
        except ValueError:
            raise NameError('problème de format dans un champ date (mois)')

        date_object = datetime.date(year, month, day)

        return date_object
    
    def __conv_to_float(self, float_string:str) -> float:
        """
        conversion basique+ en float.

        renvoie 0 si string vide ou remplie d'espaces, ou string = '-'.
        """

        if type(float_string) is None:
            return 0

        float_string = float_string.replace(" ","")
        if not float_string:
            return 0
        if float_string == "-":
            return 0

        try:
            float_string = float_string.replace(" ","")
            valeur = float(float_string.replace(',','.'))
        except ValueError:
            raise NameError('une tentative de conversion en float a échouée car string non compatible')

        return valeur
    
    def get_dataframe(self):
        """Takes the data that has been extracted from the two files, returns a Dataframe

        Returns:
            df: dataframe
        """
        
        dict_ex = self.daily_data[0]  # assumes extraction has been done and self.daily_data has been created
        keys_names = list(dict_ex.keys())
        data_dict = { k: [] for k in keys_names} # initializes dictonnary to create dataframe
        for cdict in self.daily_data:
            for k,v in cdict.items():
                data_dict[k].append(v)
        df = pd.DataFrame.from_dict(data_dict)
        
        # removing rows with missing values in total mass, fat mass, or calories in.
        df.drop(df[df['masse_totale']==0.0].index, inplace=True)
        df.drop(df[df['masse_grasse']==0.0].index, inplace=True)
        df.drop(df[df['calories_in']==0.0].index, inplace=True)
        # keeping data after 1st september 2020
        df.drop(df[df['date']<datetime.date(2020,9,1)].index, inplace=True)
        
        return df
    
    
class Display():
    """
    class to provide graphical output based on data passed as a parameter
    """
    
    def __init__(self, window_daily_mass_measurements, window_daily_data):
        """Constructor. Uses the formats viewed  in the GetData class

        Args:
        
            window_daily_mass_measurements ([type]): 
            
                # format is :
                # [ { 'date' : date_object, 'mt' : list of total mass measurements of the day (floats), 
                #     'mg' : list of total mass measurements of the day (floats) },
                #   { 'date' : date_object, 'mt' : list of total mass measurements of the day (floats), 
                #     'mg' : list of total mass measurements of the day (floats) },
                #   ...
                # ]
                
            window_daily_data ([type]): [description]
            
                # format is :
                # [ { 'date' : date_object, 'masse_totale' : total mass (float), 'masse_grasse' : fat mass (float),
                #     'calories_in' : kcals (float), 'glucides' : float, 'lipides' : float, 'proteines' : float, 'calories_exercice': float,
                #     'calories_cardio': float, 'calories_strength': float },
                # ....
                # ]    
                
        """
        
        self.daily_mass_measurements = window_daily_mass_measurements
        self.daily_data = window_daily_data   
    
    def __basic_plot(self,
                     data_list, 
                     grid=True,
                     title='titre', perc=False,
                     rolling_average=False, n_avg=7, 
                     linear_regression=False, n_reg=30):
        
        """Private method for basic display of one data over time, along with rolling average and linear regression
        
        Parameters :
        ------------
        data_list (list of dictionnaries) : this is the data, format is [ { 'date' : datetime.date object, 'y' : float} ...]
        grid (bool, optional): [présence ou pas de la grille]. Defaults to True.
        title(string, optional): [titre]. Defaults to 'titre'
        rolling_average (bool, optional): [affiche ou pas la moyenne glissante]. Defaults to False.
        n_avg (int, optional): [fenêtre de calcul de la moyenne glissante]. Defaults to 7.
        linear_regression (bool, optional): [affiche ou pas la régression linéaire]. Defaults to False.
        n_reg (int, optional): [fenêtre de calcul de la régression linéaire]. Defaults to 30.

        Returns:
        --------
            [type]: [description]
        """
        
        X = [ cdict.get('date') for cdict in data_list ]
        y = [ cdict.get('y') for cdict in data_list ]
        
        if rolling_average or linear_regression:
            fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24,8))
        else:
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
            
        ax1.plot(X, y, color='blue', marker='x')
        ax1.set_title(title)
    
        if grid: 
            ax1.grid()
            
        # calcule le rolling average sur n_avg jours
        moy = np.zeros(shape=len(y))
        moy[0] = y[0]
        for i in range(1,len(y)):
            id = max(0, i-n_avg+1)
            moy[i] = np.mean(y[id:i])
    
        if rolling_average:
            # plot moyenne glissante sur les n_moy dernières valeurs
            ax1.plot(X, moy, color='red', marker='o')
            ax2.text(0.1,0.9,f'moyenne glissante sur {n_avg} jours')
    
        if linear_regression:
            # plot regression linéaire sur les n_reg dernières valeurs des valeurs rolling average
            X_num = np.array([ d.toordinal() for d in X]).reshape(-1,1)
            lr_model = LinearRegression().fit(X_num[-n_reg:], moy[-n_reg:])   # fit sur les n_reg dernières valeurs du rolling average
            reg_pred = lr_model.predict(X_num)
            ax1.plot(X, reg_pred, color='green', marker='+')
            pente =lr_model.coef_[0] * 30 # coefficient par mois
            if perc==True: pente *= 100
            coeff = lr_model.score(X_num[-n_reg:], moy[-n_reg:])
            ax2.text(0.1,0.8,f'régression calculée sur {n_reg} jours (de {X[-n_reg]} à {X[-1]}) sur les valeurs moyennées {n_avg} jours')
            if perc==True:
                str_pente = f'pente = {np.around(pente, decimals=3)}% / mois'
            else:
                str_pente = f'pente = {np.around(pente, decimals=3)} / mois'
            ax2.text(0.1,0.7,str_pente)
            ax2.text(0.1,0.6,f'coefficient régression = {np.around(coeff * 100,1)}%')
        
        return fig
    
    def plot_weight(self, **kwargs):
        """plot the total mass data
        """
        
        data_list = [ dict( [ ('date', cdict.get('date')), ('y', np.mean(cdict.get('mt'))) ] ) for cdict in self.daily_mass_measurements ]
        self.__basic_plot(data_list, 
                          title='total mass', 
                          **kwargs)
        
    def plot_fat(self, **kwargs):
        """plot the fat mass data
        """
        
        data_list = [ dict( [ ('date', cdict.get('date')), ('y', np.mean(cdict.get('mg'))) ] ) for cdict in self.daily_mass_measurements ]
        self.__basic_plot(data_list, 
                          title='Fat mass', 
                          **kwargs)
        
    def plot_lean_mass(self, **kwargs):
        """plot the lean mass data
        """
        
        data_list = [ dict( [ ('date', cdict.get('date')), ('y', np.mean(cdict.get('mt')) - np.mean(cdict.get('mg'))) ] ) for cdict in self.daily_mass_measurements ]
        self.__basic_plot(data_list, 
                          title='Lean mass', 
                          **kwargs)
        
    def plot_body_fat_percentage(self, **kwargs):
        """plot the body fat percentage data
        """
        
        data_list = [ dict( [ ('date', cdict.get('date')), ('y', np.mean(cdict.get('mg')) / np.mean(cdict.get('mt')) ) ] ) for cdict in self.daily_mass_measurements if np.mean(cdict.get('mt'))>0 ]
        self.__basic_plot(data_list, 
                          title='body fat percentage', 
                          perc=True,
                          **kwargs)
        
    def plot_calories_net(self, **kwargs):
        """plot the net calories data
        """
        
        data_list = [ dict( [ ('date', cdict.get('date')), ('y', np.mean(cdict.get('calories_in')) - np.mean(cdict.get('calories_exercice'))) ] ) for cdict in self.daily_data ]
        self.__basic_plot(data_list, 
                          title='net calories', 
                          **kwargs)
        
def plot_moyennes(df,
                  window=7, 
                  list_of_moyennes=['calories_exercice',
                        'calories_deficit',
                        'masse_totale',
                        'masse_seche',
                        'body_fat_percentage']
                  ):
    """Affiche les colonnes choisies d'une dataframe dans un format sympa,
    en moyennant sur un nombre de jours

    Args:
        df ([type], optional): [Dataframe source].
        window (int, optional): [nombre de jours pour la moyenne]. Defaults to 7.
        list_of_moyennes (list, optional): [liste des colonnes à traiter]. Defaults to ['calories_exercice', 'calories_deficit', 'masse_totale', 'masse_seche', 'bf_perc'].
    """
    
    df_test = df
    
    for m in list_of_moyennes:
        col_name = 'moyenne ' + m
        df_test[col_name] = df_test[m].rolling(window).mean().shift(1)
        
    df_test = df_test[window:]
    n = len(list_of_moyennes)
    
    fig, ax = plt.subplots(1,n,figsize=(n*8,8))
    
    for i,m in enumerate(list_of_moyennes):
        col_name = 'moyenne ' + m
        ax[i] = df[col_name].plot(ax = ax[i], title = f'moyenne {window}j ' + m + f' vs temps', grid=True)
    
    plt.show()
    
    
# Katch-MacArdle -- considérée la plus précise si on connait le bf%

def kma(masse_seche, *kwargs):
    """Calcule métabolisme de base suivant Katch-MacArdle

    Args:
        masse_seche ([float]): [masse sèche]

    Returns:
        [float]: métabolisme de base suivant KMA
    """
    return 370 + 21.6 * masse_seche

# Mifflin Saint Jeor -- la plus précise par défaut

def msj(masse_totale, taille = 181, age = 53, *kwargs):
    return 9.99 * masse_totale + 6.25 * taille - 4.92 * age + 5

# Oxford -- pour info

def oxf(masse_totale, *kwargs):
    return 14.2 * masse_totale + 593

# Schofield -- pour info

def sch(masse_totale, *kwargs):
    return 11.472 * masse_totale + 873.1


def plot_columns(df,
                 columns_list = ['masse_totale', 'masse_grasse', 'masse_seche', 'body_fat_percentage',
                                 'calories_in', 'rest_metabolism_rate', 
                                 'calories_cardio', 'calories_strength', 'calories_deficit']):
    """Affichage brut des colonnes choisies d'une dataframe

    Args:
        df ([dataframe]): dataframe source
        columns_list (list, optional): liste des colonnes à afficher. Defaults to ['masse_totale', 'masse_grasse', 'masse_seche', 'bf_perc', 'calories_in', 'rest_metabolism', 'calories_cardio', 'calories_strength', 'calories_deficit'].
    """
    
    nb_cols_per_row = 3
    n = len(columns_list)
    
    if (n % nb_cols_per_row) == 0:
        n_rows = n // nb_cols_per_row
        n_cols = nb_cols_per_row
    else:
        n_rows = n // nb_cols_per_row + 1
        n_cols = nb_cols_per_row
        
    row_size = 8
    col_size = 8
    
    fig, axs = plt.subplots(nrows = n_rows, 
                            ncols = n_cols,
                            figsize=(row_size * nb_cols_per_row, 
                                     col_size * n_rows))
    
    if axs.ndim == 1: axs = axs[np.newaxis,:]  # required to make sure the axs array is actually 2D...
     
    for i, name in enumerate(columns_list):
        row_i = i // nb_cols_per_row
        col_i = i - row_i * nb_cols_per_row
        # print(i, row_i, col_i)
        
        axs[row_i, col_i] = df[name].plot(ax=axs[row_i, col_i], title=name + f' vs temps',grid=True)
                
    plt.show()
    
def plot_trends(df,
                columns_list = ['masse_totale',
                                'masse_seche',
                                'body_fat_percentage']
                ):
    """Affichage colonnes choisies d'une dataframe avec tendances linéaires

    Args:
        df (dataframe): données source.
        columns_list (list, optional): colonnes sur lesquelles travailler. Defaults to ['masse_totale', 'masse_seche', 'bf_perc'].
    """
    
    nb_cols_per_row = 3
    n = len(columns_list)
    
    if (n % nb_cols_per_row) == 0:
        n_rows = n // nb_cols_per_row
        n_cols = nb_cols_per_row
    else:
        n_rows = n // nb_cols_per_row + 1
        n_cols = nb_cols_per_row
        
    row_size = 8
    col_size = 8
    
    fig, axs = plt.subplots(nrows = n_rows, 
                            ncols = n_cols,
                            figsize=(row_size * nb_cols_per_row, 
                                     col_size * n_rows))
    
    if axs.ndim == 1: axs = axs[np.newaxis,:]  # required to make sure the axs array is actually 2D...
    
    for i, name in enumerate(columns_list):
        row_i = i // nb_cols_per_row
        col_i = i - row_i * nb_cols_per_row
        
        #------------- display -----------------------------------------------------------------------
        #--- raw data --------------------------------------------------------------------------------
        axs[row_i, col_i] = df[name].plot(ax=axs[row_i, col_i], title=name + f' vs temps',grid=True)
        #--- overall trend ---------------------------------------------------------------------------
        y = df[name]
        x_ord = np.array([d.toordinal() for d in df[name].index]).reshape(-1,1)
        lr = LinearRegression().fit(x_ord, y)
        y_pred = lr.predict(x_ord)
        axs[row_i, col_i].plot(df[name].index,y_pred)
        r = lr.score(x_ord, y)
        #--- numbers ---------------------------------------------------------------------------------
        print(f'rythme mensuel moyen sur la période de ' + name + f' = {np.round(lr.coef_[0] * 30,3)} avec coeff correlation = {np.round(r*100)}%')
        
    plt.show()
    

def plot_boxes(df_seche,
               df_bulk,
               columns_list = ['calories_in', 
                               'rest_metabolism_rate', 
                               'calories_deficit']):
    """Affiche boxplots comparés des colonnes d'intérêt entre deux dataframes

    Args:
        df_seche (dataframe): période 1
        df_bulk (dataframe): période 2
        columns_list (list, optional): colonnes d'intérêt. Defaults to ['calories_in', 'rest_metabolism', 'calories_deficit'].
    """
    
    nb_cols_per_row = 3
    n = len(columns_list)
    
    if (n % nb_cols_per_row) == 0:
        n_rows = n // nb_cols_per_row
        n_cols = nb_cols_per_row
    else:
        n_rows = n // nb_cols_per_row + 1
        n_cols = nb_cols_per_row
    
    row_size = 6
    col_size = 6
    
    fig, axs = plt.subplots(nrows = n_rows, 
                            ncols = nb_cols_per_row,
                            figsize=(row_size * nb_cols_per_row, 
                                     col_size * n_rows))
    
    for i, name in enumerate(columns_list):
        row_i = i // nb_cols_per_row
        col_i = i - row_i * nb_cols_per_row
        
        data = [ df_seche[name],
                df_bulk[name]
                ]
        
        axs[row_i, col_i].set_title(name + ' par periode')
        axs[row_i, col_i].boxplot(data, labels = ['seche', 'bulk'], showmeans=True)
        axs[row_i, col_i].grid()
        
        m = np.round(df_seche[name].median())
        print(f'valeur médiane de ' + name + f' pendant la période seche = {m} kcals')
        m = np.round(df_bulk[name].median())
        print(f'valeur médiane de ' + name + f' pendant la période bulk = {m} kcals')
        print(f'---------------')
        
    plt.show()
    
    
def plot_moyennes_with_targets(df,
                  window=7, 
                  list_of_moyennes=['masse_totale',
                                    'masse_seche',
                                    'body_fat_percentage',
                                    'calories_in',
                                    'calories_exercice_net',
                                    'calories_deficit'
                                    ],
                  list_of_targets = {
                      'calories_in' : 1850.0,
                      'calories_exercice_net' : 350.0,
                      'calories_deficit' : -200.0
                      }
                  ):
    """Affiche les colonnes choisies d'une dataframe dans un format sympa,
    en moyennant sur un nombre de jours, avec des targets

    Args:
        df ([type], optional): [Dataframe source].
        window (int, optional): [nombre de jours pour la moyenne]. Defaults to 7.
        list_of_moyennes (list, optional): [liste des colonnes à traiter]. Defaults to ['calories_exercice', 'calories_deficit', 'masse_totale', 'masse_seche', 'body_fat_percentage'].
    """
    
    df_test = df
    
    for m in list_of_moyennes:
        col_name = 'moyenne ' + m
        df_test[col_name] = df_test[m].rolling(window).mean().shift(1)
        target = list_of_targets.get(m)
        if target is not None:
            name_target = 'target ' + m
            df_test[name_target] = target        
        
    df_test = df_test[window:]
    n = len(list_of_moyennes)
    
    fig, ax = plt.subplots(1,n,figsize=(n*8,8))
    
    for i,m in enumerate(list_of_moyennes):
        col_name = 'moyenne ' + m
        df[col_name].plot(ax = ax[i], title = f'moyenne {window}j ' + m + f' vs temps', grid=True)
        target = list_of_targets.get(m)
        if target is not None:
            name_target = 'target ' + m
            df[name_target].plot(ax = ax[i], title = f'moyenne {window}j ' + m + f' vs temps avec target', grid=True)
    
    plt.show()

    
