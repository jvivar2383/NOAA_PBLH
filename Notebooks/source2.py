#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

"""
Created on Wed Jun 29 17:59:35 2022

@author: jenifervivar
"""


def pbl_height_sub(df, stat="std", var_type="CNR", avetime="1H"):
    """
    input:
    This functions takes a dataframe with dates as index, and the statistic of interest you want o calculate per hour.
    The default is std.

    output:
    dataframe with mean, median, variance, or std deviation of data over avetime.

    """

    import pandas as pd

    # creates a copy to have the date as a column instead of an index
    df_cop = df.transpose()
    df_cop.index = df_cop.index.astype(int)

    if var_type == "CNR":
        # applying different statiscs values per hour
        if stat == "mean":
            std_series = df_cop.resample(avetime, axis=1).mean()
        elif stat == "median":
            std_series = df_cop.resample(avetime, axis=1).median()
        #elif stat == "var":
           # std_series = df_cop.resample(avetime, axis=1).var()
        else:
            std_series = df_cop.resample(avetime, axis=1).std()

    # gets values for the date of interest
    if var_type == "wind":
        std_series = df_cop.resample(avetime, axis=1).var()
    return std_series.transpose()


def pbl_height(df, stat="std", var_type=None):

    """
    input:
    This functions takes a dataframe with dates as index, and the statistic of interest you want o calculate per hour.
    The default is std.

    outpu:
    the functions outputs a tupple of two arrays, a datetime array and an array with the PBL heigth

    """
    std_series = pbl_height_sub(df, stat, var_type)
    # std_series = std_series.transpose()
    rval = []
    tval = []
    pbl = []

    for xval in std_series.iterrows():
        time = xval[0]
        temp = xval[1]

        if var_type == "wind":
            
            #the height is adopted at the first height where the var is lover than 0.16 m^2/s^2   
            temp1 = temp[temp < 0.16]
            if len(temp1) != 0:
                val = list(temp1)[0]
                maxindex = (temp == val).argmax()
            #else:
             #   maxindex = temp.argmin()
        else:
            # argmax returns index of place in series with largest value
            maxindex = temp.argmax()
        # get index value (height) of place of largest value

        pblht = int(temp.index[maxindex])
        # std = temp[maxindex]
        rval.append((time, pblht))
        pbl.append(pblht)
        tval.append(time)
        # stdra.append(std)
    return pd.to_datetime(tval), pbl


def dataframe_set(array1, time_array, day=None, columntype=str):
    """
    input:
    the function takes two arrays as inputs, an arrays of values and a datetime array. The fucntion also takes
    the day were insterested in (in case the files have more than one day available)

    output:
    the fucntion returns a dataframe with the heights as string columns, datetime array as index and the values for
    the date selected only
    """

    import numpy as np
    import pandas as pd

    # heights are set.
    columns = (np.array([i for i in range(200, 5200, 100)])).astype(columntype)
    df = pd.DataFrame(
        np.flip(array1).reshape(int(len(array1) / 50), 50),
        columns= np.flip(columns),
        index= np.flip(time_array.round("S")),
        )
    if day:
        df = df[df.index.date == pd.Timestamp(day).date()]
    return df


def pbl_lidar(df):

    """
    Function takes a dataframe object and returns a pandas core series with time as the index.
    The mean is computed from the values where the flag is equal to 30. The mean is taken per every hour and the values are
    rounded.
    """
    import pandas as pd

    h = []
    time = []
    for i in range(len(df)):
        for column in df.columns:
            if df[column][i] == 30:
                h.append(column)
                time.append(df.index[i])
    data = {"Time": time, "Heigth": [int(i) for i in h]}

    df_ = pd.DataFrame(data=data)
    return df_.resample("H", on="Time")["Heigth"].mean().round()


def plot_all(
    df_cnr=None,
    df_lidar=None,
    tup_mean=None,
    tup_std=None,
    tup_median=None,
    wind_std=None,
    date="",
    cbarlbl = "CNR (dB)",
    cmap = 'seismic',
    plot_type = 'contourf'
):

    """
    The function plots all calculated CNR values on a heatmap/contour plot

    input:
    cnr data frame
    pbl dataframe
    tuples for mean, median and std
    the date (as string) for title name

    outputs:
    heatmap graph with imposed graph lines
    """
    import matplotlib.pyplot as plt
    import numpy as np

    import seaborn as sns
    fig, ax =plt.subplots(figsize = (10,10))

    # the transpose does not take a long time. It is the pcolormesh that takes a lont time
    # to render all the points.
    
    pcnr= df_cnr.transpose()
    pcnr.index = pcnr.index.astype(int)
    if plot_type == 'pcolormesh':
        CS = plt.pcolormesh(pcnr.columns, pcnr.index.values, pcnr.values, cmap=cmap)
    if plot_type == 'contourf':
        CS = plt.contourf(pcnr.columns, pcnr.index.values, pcnr.values, cmap=cmap)
    else:
        CS = plt.pcolormesh(pcnr.columns, pcnr.index.values, pcnr.values, cmap=cmap)

    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(cbarlbl)
    fig.autofmt_xdate()
    
    tlist = (list, np.ndarray, tuple)
    if isinstance(tup_mean, tlist):
        plt.plot(
            tup_mean[0].hour, tup_mean[1], label="CNR-mean derived Values", color="lime"
        )
    if isinstance(tup_std, tlist):
        ax.plot(
            tup_std[0], tup_std[1], label="CNR-STD derived Values", color="deeppink"
        )
    if isinstance(tup_median, tlist):
        plt.plot(
            tup_median[0].hour,
            tup_median[1],
            label="CNR-Median derived Values",
            color="white",
        )
    if isinstance(wind_std, tlist):
        plt.plot(wind_std[0].hour, wind_std[1], label="Wind Variance", color="black")
    if isinstance(df_lidar, tlist):
        plt.plot(df_lidar.index.hour, df_lidar, label="LiDAR Values", color="yellow")
    plt.title("PBL Calculations for " + date)

    plt.ylabel("Heigth (m)")
    plt.xlabel("Time (UTC)")
    
    plt.legend()

    plt.show()
    
    
#will give it a better name :)

def do_the_thing(C, date):
    
    variables.extract()
    cnr_17 = variables.cnr_day
    time_17 = pd.to_datetime(variables.time_day, unit = 's', utc = True)
    
    july17_cnr = dataframe_set(cnr_17, time_17, date)

    july17_cnrmean = pbl_heigth(july17_cnr, stat = "mean", var_type = "CNR")
    july17_cnrstd = pbl_heigth(july17_cnr, var_type = "CNR")
    july17_cnrmedian = pbl_heigth(july17_cnr, stat = "median", var_type = "CNR")

    #Getting liDAR derived values for PBL
    pbl_17 = variables.atm_structures
    df_structures_17 = dataframe_set(pbl_17, time_17, date)
    lidar_pbl17 = pbl_lidar(df_structures_17)

    #lidar_pbl17 = lidar_pbl17[lidar_pbl17.index.date ==pd.Timestamp("2021-07-17").date()]
    windsp_17 = variables.ver_wind_speed
    df_wind = dataframe_set(windsp_17, time_17, date)
    
    wind_var = pbl_heigth(df_wind,  var_type = "wind")

    plot_all(july17_cnr,lidar_pbl17,july17_cnrmean, july17_cnrstd,july17_cnrmedian, wind_var,date)
    # plt.show()
    return ax

class VAREXTRACT:

    """
    This function extracts the variables from a netcdf file

    Input:
    Path to the netCDF file. It is assumed the data is containn in folders with individual files in the folder

    output:
    The function returns the following numpy arrays as attributes of the extract function:
    *CNR
    *time
    *Relatibe Beta
    *Doppler Spectral Width
    *atmospherical_structures_type (clouds, PBL, etc)
    *radial_wind_speed
    """

    def __init__(self, path):
        import numpy as np

        self.path = path
        self.time_day = np.array([])
        self.cnr_day = np.array([])
        self.relat_beta_day = np.array([])
        self.spectral_width_day = np.array([])
        self.atm_structures = np.array([])
        self.ver_wind_speed = np.array([])
        self.range_day = np.array([])

    def extract(self):

        import numpy as np
        import netCDF4 as nc
        import os

        for file in sorted(os.listdir(self.path)):

            # skips dummy files in mac
            if file != ".DS_Store":
                data = nc.Dataset(
                    self.path + file, mode="r"
                )  # this will open the netCDF data

                # gets the name of the group that contains the data as it changes in every file
                sweep_file = data[list(data.groups.keys())[1]]

                # THE [:] IS NEEDED TO GET ALL THE VALUES
                dumm_cnr = sweep_file.variables["cnr"][:]
                dumm_time = sweep_file.variables["time"][:]
                dumm_rela_beta = sweep_file.variables["relative_beta"][:]
                dumm_spectral = sweep_file.variables["doppler_spectrum_width"][:]
                dum_struct = sweep_file.variables["atmospherical_structures_type"][:]
                dumm_wind = sweep_file.variables["radial_wind_speed"][:]
                dumm_range = sweep_file.variables["range"][:]

                # appending all the values into one numpy array
                self.cnr_day = np.append(self.cnr_day, dumm_cnr)
                self.time_day = np.append(self.time_day, dumm_time)
                self.relat_beta_day = np.append(self.relat_beta_day, dumm_rela_beta)
                self.spectral_width_day = np.append(self.spectral_width_day, dumm_spectral)
                self.atm_structures = np.append(self.atm_structures, dum_struct)
                self.ver_wind_speed = np.append(self.ver_wind_speed, dumm_wind)
                self.range_day = np.append(self.range_day, dumm_range)
