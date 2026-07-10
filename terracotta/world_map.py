import numpy as np
import pandas as pd


def _format_parameter_string(s):
    """Formats the parameter string to match expected time:value format

    Parameters
    ----------
    s : string
        Input parameter string

    Returns
    -------
    formatted : string
        Formatted parameter string
    """

    if ":" in s:
        split_string = s.replace(",", ":").split(":")
        epochs = np.array(split_string[::2]).astype(float)
        parameters = np.array(split_string[1::2])
        sorted_epochs = np.argsort(epochs)
        formatted = ""
        if epochs[sorted_epochs[0]] != 0:
            formatted += "0:0,"
        for e in sorted_epochs:
            formatted += f"{epochs[e]}:{parameters[e]},"
        return formatted[:-1]
    elif s == "nan":
        raise RuntimeError(f"String `{s}` cannot be formatted.")
    return f"0:{s}"


class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
    connections : pandas.DataFrame
    samples : pandas.DataFrame
    epochs : numpy.ndarray
    """

    def __init__(self, demes, connections, samples=None):
        """Initializes the WorldMap object

        Parameters
        ----------
        demes : pandas.DataFrame
            From demes.tsv file
        connections : pandas.DataFrame
            From connections.tsv file
        samples : pandas.DataFrame
            From samples.tsv file (optional)

        Attributes
        ----------
        demes
        connections
        samples
        epochs
        parameters
        """

        self.parameters = ["coefficient"]

        self.demes = demes.copy()  
        self.demes["suitability"] = self.demes["suitability"].astype(str)
        formatted_suitability_strings = []
        for suitability in self.demes["suitability"]:
            formatted_suitability_strings.append(_format_parameter_string(suitability))
        self.demes["suitability"] = formatted_suitability_strings        

        self.connections = connections.copy()
        self.connections["migration_modifier"] = self.connections["migration_modifier"].astype(str)
        formatted_modifier_strings = []
        for modifier in self.connections["migration_modifier"]:
            formatted_modifier_strings.append(_format_parameter_string(modifier))
        self.connections["migration_modifier"] = formatted_modifier_strings
    
        epochs = [0]
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = float(epoch_assignment)
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        migration_modifier_variables = []
        for modifier_string in self.connections["migration_modifier"]:
            epoch_formatter = modifier_string.replace(",", ":").split(":")
            for t in range(0, len(epoch_formatter), 2):
                epoch_assignment = float(epoch_formatter[t])
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        self.epochs = np.sort(epochs)

        self.suitabilities = self._build_suitability_array()

        self.connection_modifiers = self._build_connection_modifiers_array()
        modifiers = np.unique(self.connection_modifiers)
        if len(modifiers) > 1:
            for mod in modifiers:
                if mod.isalpha():
                    self.parameters.append(mod)

        self.samples = None
        if samples is not None:
            self.samples = samples.copy()
    
    def get_deme_suitability_at_time(self, id, time):
        """Gets the type for specified deme

        Wrapper of pandas.DataFrame function. Assumes that deme IDs are unique.

        Parameters
        ----------
        id : int
            ID of deme
        time : int or float
            Specified time of map

        Returns
        -------
        deme_type : float
            Type of deme
        """

        deme_suitability = self.demes.loc[self.demes["id"]==id,"suitability"].iloc[0]
        epochs = deme_suitability.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return float(details[1])
        return float(epochs[-1].split(":")[1])

    def get_all_deme_suitabilities_at_time(self, time):
        """

        Parameters
        ----------
        time : int or float
            Specified time of world map

        Returns
        -------
        suitabilities_at_time : pd.Series
            Deme types ordered by the demes dataframe
        """

        return pd.Series([self.get_deme_suitability_at_time(id, time) for id in self.demes["id"]])

    def get_connection_migration_modifier_at_time(self, id, time):
        """Extracts the migration modifier for a connection at time

        Parameters
        ----------
        id : int
        time : int or float

        Returns
        -------
        migration_modifier : str
            Modifier of connection at specified time
        """
        connection_parameter = self.connections.loc[self.connections["id"]==id,"migration_modifier"].iloc[0]
        epochs = connection_parameter.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return details[1]
        migration_modifier = epochs[-1].split(":")[1]
        return migration_modifier

    def get_all_connection_migration_modifiers_at_time(self, time):
        return pd.Series([self.get_connection_migration_modifier_at_time(id, time) for id in self.connections["id"]])

    def build_transition_matrices(self, parameters):
        """Builds the transition matrix based on the world map and migration rate parameters

        Row is the target deme, column is the source deme backwards in time.

        Parameters
        ----------
        parameters : np.array or list
        
        Returns
        -------
        transition_matrix : np.array
        """

        if len(parameters) != len(self.parameters):
            raise RuntimeError("Length of `parameters` does not equal WorldMap.parameters. You must provide values for all parameters, in order.")

        if "coefficient" in self.parameters:
            m = parameters[self.parameters.index("coefficient")]
        else:
            m = 1
    
        transition_matrix = np.zeros((len(self.epochs), len(self.demes),len(self.demes)))
        for e in range(len(self.epochs)):
            for f,connection in self.connections.iterrows():
                i_0 = self.demes.loc[self.demes["id"]==connection["deme_0"]].index[0]
                i_1 = self.demes.loc[self.demes["id"]==connection["deme_1"]].index[0]
                denom = self.suitabilities[e][i_0]
                if denom == 0:
                    denom = 1e-99
                suit = (self.suitabilities[e][i_1] / denom)
                mod = self.connection_modifiers[e][f]
                if mod in self.parameters:
                    mod = parameters[self.parameters.index(mod)]
                else:
                    mod = float(mod)
                transition_matrix[e, i_1, i_0] = (m * suit) * mod
            diag = -np.sum(transition_matrix[e], axis=0)
            np.fill_diagonal(transition_matrix[e], diag)
        return transition_matrix

    def build_sample_locations_array(self):
        """Formats the sample locations into probability distribution vectors

        Returns
        -------
        sample_locations_array : numpy.ndarray
            Probability distribution vector for each sample location (generally 0 in all demes except one)
        sample_ids : numpy.ndarray
            Order of sample IDs for `sample_locations_array`
        """

        if (self.samples is None):
            raise RuntimeError("No samples provided. Check that you've added samples to your WorldMap.")
        sample_locations_array = np.zeros((len(self.samples), len(self.demes)), dtype="float64")
        sample_ids = np.array(self.samples["id"])
        for i, sample in self.samples.iterrows():
            sample_locations_array[i,self.demes.loc[self.demes["id"]==sample["deme"]].index[0]] = 1
        return sample_locations_array, sample_ids

    def _build_suitability_array(self):
        """Builds the suitability array
        """

        suitabilities = np.zeros((len(self.epochs), len(self.demes)), dtype="float")
        for i,epoch in enumerate(self.epochs):
            suitabilities[i] = self.get_all_deme_suitabilities_at_time(epoch)
        return suitabilities
    
    def _build_connection_modifiers_array(self):
        modifiers = np.zeros((len(self.epochs), len(self.connections)), dtype="object")
        for i,epoch in enumerate(self.epochs):
            modifiers[i] = self.get_all_connection_migration_modifiers_at_time(epoch)
        return modifiers