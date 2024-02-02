# fonctions base pour test RNN

import tensorflow as tf

### Classe génératrice de dataset

# Le dataset est hyperparamétré par :
# - PAST : le nombre de jours connus de l'historique
# - FUTURE : le nombre de jours à venir pour la prévision

# Les inputs sont les données connues, à savoir :
# - dans le PAST, les calories (food et exercices), et les masses (totale, grasse) mesurées
# - dans le FUTURE, les calories (food et exercices)

# Les outputs, que le modèle doit apprendre, sont alors :
# - dans le FUTURE, les masses (totale, grasse)

class CreateDataset():
    """Generic class to generate dataset in tf.dataset format
    """
    
    PAST = 7
    FUTURE = 3
    
    def __init__(self, dataframe, past=None, future=None, batch_size=1):
        """Constructor

        Args:
            dataframe (dataframe, optional): dataframe with dates as index, ['Calories', 'exo_cals_nets', 'Masse_Totale', 'Masse_Grasse'] as columns.
            past (_type_, optional): number of days in the past, including today. Defaults to 7.
            future (_type_, optional): number of days in the future, including tomorrow. Defaults to 3.
        """

        self.dataframe = dataframe.dropna()
        
        self.past = self.PAST if past is None else past
        self.future = self.FUTURE if future is None else future
        self.batch_size = 32 if batch_size is None else batch_size
        
    def create_dataset(self, batch_size=None):
        """engine to create the tf.dataset

        Args:
            batch_size (int, optional): _description_. Defaults to 32.
        """
        
        self.batch_size = 32 if batch_size is None else batch_size
        dataframe = self.dataframe
        # 
        # the target for the input window at index i is the dataframe[['Masse_Totale', 'Masse_Grasse']] from date i + PAST included to date i + PAST + FUTURE - 1 included (length = FUTURE )
        
        # prepare the input data
        
        # a) all values known in the PAST : the first piece of the input window at index i is an extract of the whole dataframe from date i included to date i+PAST-1 included (length = PAST)
        data_inputs1 = dataframe[['Calories', 'exo_cals_nets', 'Masse_Totale', 'Masse_Grasse']].to_numpy()        
        ds_inputs1 = tf.keras.utils.timeseries_dataset_from_array(
            data=data_inputs1,
            targets=None,
            sequence_length=self.past,
            batch_size=self.batch_size
        )
        
        # b) all values calories in the FUTURE : columns 'Calories' and 'exo_cals_nets' from i+PAST to i+PAST+FUTURE-1
        data_inputs2 = dataframe[['Calories', 'exo_cals_nets']].to_numpy()
        ds_inputs2 = tf.keras.utils.timeseries_dataset_from_array(
            data=data_inputs2,
            targets=None,
            sequence_length=self.future,
            start_index=self.past,
            batch_size=self.batch_size
        )             
                
        # # create the output data, by flattening a window of future data
        # imax = len(data_inputs1) - self.past - self.future # maximum indice from which to extract a window of length past+future
        # outputs_raw = dataframe[['Masse_Totale', 'Masse_Grasse']]
        # targets = np.zeros(shape=(imax+1,self.future * 2))
        # for i in range(len(targets)):
        #     targets[i] = outputs_raw.iloc[i+self.past:i+self.past+self.future,:].to_numpy().reshape(-1)
        # data_inputs1 = data_inputs1[:imax+1]
        
        # # create the dataset itself
        # ds_outputs = tf.keras.utils.timeseries_dataset_from_array(
        #     data=targets,
        #     targets=None,
        #     sequence_length=self.future,
        #     batch_size=None
        # )
        
        outputs = dataframe[['Masse_Totale', 'Masse_Grasse']].to_numpy()
        ds_outputs = tf.keras.utils.timeseries_dataset_from_array(
            data=outputs,
            targets=None,
            sequence_length=self.future,
            start_index=self.past,
            batch_size=self.batch_size
        )
        
        # final dataset
        ds_inputs = tf.data.Dataset.zip((ds_inputs1, ds_inputs2))
        ds = tf.data.Dataset.zip((ds_inputs, ds_outputs))
        
        # the final outputs is a tf.dataset of elements of specs :
        #         ((TensorSpec(shape=(None, 4), dtype=tf.float64, name=None), TensorSpec(shape=(None, 2), dtype=tf.float64, name=None)),  # INPUTS = PAST values of 4 features (exo_cals_nets, Calories, masse_totale, masse_grasse) + FUTURE values of 2 features (Calories, exo_cals_nets)
        #           TensorSpec(shape=(None, 6), dtype=tf.float64, name=None))                                                               # OUTPUTS = FUTURE values of 2 features (masse_totale, masse_grasse)
        return ds