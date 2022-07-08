#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:59:35 2022

@author: jenifervivar
"""

def pbl_heigth(df, stat = "std", var_type = None):
    
    """
    input:
    This functions takes a dataframe with dates as index, and the statistic of interest you want o calculate per hour.
    The default is std.
    
    outpu:
    the functions outputs a tupple of two arrays, a datetime array and an array with the PBL heigth
    
    """
    
    import pandas as pd
    
    
    
  
    #creates a copy to have the date as a column instead of an index
    df_cop = df.transpose()
    df_cop.index = df_cop.index.astype(int)
    
    if var_type == "CNR":
        #applying different statiscs values per hour     
        if stat == "mean":
            std_series = df_cop.resample('1H', axis=1).mean()
        elif stat == "median":
            std_series = df_cop.resample('1H', axis=1).median()
        else:
            std_series = df_cop.resample('1H', axis=1).std()

    #gets values for the date of interest
    if var_type == "wind":
        std_series = df_cop.resample('1H', axis=1).var()
    
    std_series = std_series.transpose()
    rval = []
    tval = []
    pbl = []
   
    for xval in std_series.iterrows(): 
        time = xval[0]
        temp = xval[1]
        
        if var_type =="wind":
            
            temp1 = temp[temp<0.16]
            if len(temp1) !=0:
                val = list(temp1)[0]
                maxindex = (temp == val).argmax()
            else:
                maxindex = temp.argmin()
        else:    
        # argmax returns index of place in series with largest value
            maxindex = temp.argmax()
        # get index value (height) of place of largest value

        pblht = int(temp.index[maxindex])
        #std = temp[maxindex]
        rval.append((time,pblht))
        pbl.append(pblht)
        tval.append(time)
        #stdra.append(std)
    return pd.to_datetime(tval),pbl





                

def dataframe_set(array1, time_array, day):
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
    
    columns = (np.array([i for i in range(200, 5200, 100)])).astype(str)
    df = pd.DataFrame(np.flip((array1)).reshape(int(len(array1)/50),50), columns = np.flip(columns), index =(time_array.round('S')))
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
    time= []
    for i in range(len(df)):
        for column in df.columns:
            if df[column][i]==30:
                h.append(column)
                time.append(df.index[i])
    data = {'Time': time, 'Heigth': [int(i) for i in h]}
    
    df_ = pd.DataFrame(data = data)
    return df_.resample("H", on='Time')['Heigth'].mean().round() 


    
def plot_all(df_cnr = None, df_lidar= None, tup_mean = None, tup_std = None, tup_median= None,wind_std = None, date = ""):
    
    """
    The function plots all calculated CNR values on a heatmap/contour plot
    
    input:
    cnr data frane
    pbl datafrane
    tuples for mean, median and std
    the date (as string) for title name
    
    outputs:
    heatmap graph with imposed graph lines
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax =plt.subplots(figsize = (10,10))
    pcnr= df_cnr.transpose()
    pcnr.index = pcnr.index.astype(int)
    #pcnr.columns = pcnr.columns.hour
    #CS = plt.contourf(pcnr.columns, np.flip(pcnr.index.values),np.flip(pcnr.values), cmap= "seismic")
    CS = plt.pcolormesh(pcnr.columns, pcnr.index.values,pcnr.values, cmap= "seismic")
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('CNR (dB)')
    fig.autofmt_xdate() 
    if isinstance(tup_mean, (list, np.ndarray)): 
        plt.plot(tup_mean[0].hour, tup_mean[1], label = "CNR-mean derived Values", color = "lime")
        plt.plot(tup_std[0].hour, tup_std[1], label = "CNR-STD derived Values", color = "deeppink")
        plt.plot(tup_median[0].hour, tup_median[1], label = "CNR-Median derived Values", color = "white")
        plt.plot(wind_std[0].hour, wind_std[1], label = "Wind Variance", color = "black")
        plt.plot(df_lidar.index.hour,df_lidar, label = "LiDAR Values", color = "yellow")
        plt.title("PBL Calculations for "+ date)
        plt.ylabel("Heigth (m)")
        plt.xlabel("Time (UTC)")
        plt.legend()
        plt.gcf()
    plt.show()
    
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
            
            #skips dummy files in mac
            if file != '.DS_Store':
                data = nc.Dataset(self.path + file , mode ='r')#this will open the netCDF data

                #gets the name of the group that contains the data as it changes in every file
                sweep_file = data[list(data.groups.keys())[1]]

                #THE [:] IS NEEDED TO GET ALL THE VALUES
                dumm_cnr = sweep_file.variables['cnr'][:]
                dumm_time= sweep_file.variables['time'][:]
                dumm_rela_beta = sweep_file.variables["relative_beta"][:]
                dumm_spectral = sweep_file.variables["doppler_spectrum_width"][:]
                dum_struct = sweep_file.variables["atmospherical_structures_type"][:]
                dumm_wind = sweep_file.variables["radial_wind_speed"][:]
                dumm_range = sweep_file.variables["range"][:]
                
                #appending all the values into one numpy array
                self.cnr_day =np.append(self.cnr_day, dumm_cnr)
                self.time_day = np.append(self.time_day,dumm_time)
                self.relat_beta_day = np.append(self.relat_beta_day, dumm_rela_beta)
                self.spectral_width_day = np.append(self.spectral_width_day, dumm_spectral)
                self.atm_structures = np.append(self.atm_structures, dum_struct)
                self.ver_wind_speed = np.append(self.ver_wind_speed, dumm_wind)
                self.range_day = np.append(self.range_day, dumm_range)
               
