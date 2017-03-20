#!/usr/bin/env python
""" Observed Ozone Spatial Interpolator

Description:
    This script reads in a CSV containing maximum daily averaged 8-hour ozone (MDA8 O3) values across the Continental
    United States (CONUS) and interpolates the the data points across the domain. Interpolation methods are either
    linear or nearest neighbor.

Usage:
    python Interpolte_MDA8_Ozone.py

    or

    ./Interpolte_MDA8_Ozone.py

Dataset:
    EPA's AQS Network
    MDA8 Ozone
    July 1, 2012
    http://aqsdr1.epa.gov/aqsweb/aqstmp/airdata/download_files.html#Daily

TODO:
    Mask out non-CONUS regions including Mexico, Cananda, and water where skewing occurs in the distribution
    of interpolated concentrations vs. observed concentration distribution

"""
import matplotlib
matplotlib.use('TkAgg')  # Display plots in GUI on Mac OS X
import numpy as np
import pandas as pd
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# ~~~~ USER OPTIONS ~~~~ #

# Specify path to observation file (example is CONUS MDA8 O3 for July 1, 2012)
obs_file = './data/AQS_MDA8O3_CONUS_20120701.csv'

# Specify interpolation method: nn (nearest neighbor), linear
interpolation_method = 'nn'

# ~~~~ END USER OPTIONS ~~~~ #


def define_grid(obs_df):
    print("Defining Domain")
    # Assign lat/lons to variables
    lats = obs_df['Latitude']
    lons = obs_df['Longitude']
    
    # Define basemap extent
    lcrnrlon = min(lons)
    llcrnrlat = min(lats)-5
    urcrnrlon = max(lons)+5
    urcrnrlat = max(lats)+2
    
    return lcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat
    

def mk_basemap(obs_df, lcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat):
    print("Creating Basemap")

    
    # Create basemap
    m = Basemap(llcrnrlon=lcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                rsphere=(6378137.00, 6356752.3142),
                resolution='h', area_thresh=1000., projection='lcc',
                lat_1=45., lat_2=33, lon_0=-97.)
    
    # Map lat lons to basemap
    obs_df['lon'], obs_df['lat'] = m(*(obs_df.Longitude.values, obs_df.Latitude.values))

    print("Creating Grid")
    # Create grid to interpolate data on
    xi_lin = np.linspace(obs_df['lon'].min(), obs_df['lon'].max(), 1000)
    yi_lin = np.linspace(obs_df['lat'].min(), obs_df['lat'].max(), 1000)
    
    # Create x/y grid space
    xi, yi = np.meshgrid(xi_lin, yi_lin)

    return m, obs_df, xi, yi


def interp_obs(data, xi, yi, interpolation_method):
    print("Interpolating Observed Data Points")
    # Set up grid/values to conduct interpolation on
    x, y, obs_arr = data['lon'].values, data['lat'].values, data['1st Max Value'].values*1000
    
    # Interpolate MDA8 ozone using nearest neighbor
    ozone_interp = griddata(x, y, obs_arr, xi, yi, interp=interpolation_method)
    
    return x, y, obs_arr, ozone_interp


def mk_plots(m, llcrnrlat, urcrnrlat, lcrnrlon, urcrnrlon, data, xi, yi, ozone_interp, obs_arr,interpolation_method):
    print("Generating plots")
    # Define plot size
    # plt.figure(figsize=(12, 10))

    # Define global subplot options
    colorbar_shrink = 0.7
    colorbar_size = 8
    title_size = 10
    tick_size = 8
    colorbar_padding = 0.1

    # Contour & Contour Filled & Observed Data Subplot
    plt.subplot(2, 2, 2)
    # Draw shape files
    m.drawmapboundary()
    m.fillcontinents()
    m.drawcountries()

    # Draw lat/lon lines
    m.drawparallels(np.arange(round(llcrnrlat), round(urcrnrlat), 4.),
                    color='black', linewidth=0.5, labels=[True, False, False, False], size=tick_size)
    m.drawmeridians(np.arange(round(lcrnrlon), round(urcrnrlon), 8.),
                    color='0.25', linewidth=0.5, labels=[False, False, False, True], size=tick_size)

    # contour plot
    CS = plt.contour(xi, yi, ozone_interp, 20, linewidths=0.5, colors='k')
    contourfilled = m.contourf(xi, yi, ozone_interp, zorder=4, alpha=0.6, cmap="jet")
    # scatter plot
    m.scatter(data['lon'], data['lat'], alpha=.75, cmap="jet", zorder=4)

    cbar = plt.colorbar(contourfilled, shrink=colorbar_shrink, orientation='horizontal', pad=colorbar_padding)
    cbar.ax.tick_params(labelsize=colorbar_size)
    cbar.set_label("(ppb)", size=8)
    plt.title("Interpolated MDA8 Ozone with AQS Locations \n July 1, 2012", size=title_size)

    # Observed data only subplot
    plt.subplot(2, 2, 1)
    # Draw shape files
    m.drawmapboundary()
    m.fillcontinents()
    m.drawcountries()

    # Draw lat lon lines
    m.drawparallels(np.arange(round(llcrnrlat), round(urcrnrlat), 4.),
                    color='black', linewidth=0.5, labels=[True, False, False, False], size=tick_size)
    m.drawmeridians(np.arange(round(lcrnrlon), round(urcrnrlon), 8.),
                    color='0.25', linewidth=0.5, labels=[False, False, False, True], size=tick_size)

    # scatter plot
    m.scatter(data['lon'], data['lat'], 10, data['1st Max Value']*1000, alpha=.75, cmap="jet", zorder=4)

    cbar = plt.colorbar(contourfilled, shrink=colorbar_shrink, orientation='horizontal', pad=colorbar_padding)
    cbar.set_label("(ppb)", size=8)
    cbar.ax.tick_params(labelsize=colorbar_size)
    plt.title("Observed MDA8 Ozone \n July 1, 2012", size=title_size)

    # Contour Only Subplot
    plt.subplot(2, 2, 3)

    # Draw shape files
    m.drawmapboundary()
    m.fillcontinents()
    m.drawstates()
    m.drawcountries()
    m.drawcoastlines()

    # Draw lat/lon lines
    m.drawparallels(np.arange(round(llcrnrlat), round(urcrnrlat), 4.),
                    color='black', linewidth=0.5, labels=[True, False, False, False], size=tick_size)
    m.drawmeridians(np.arange(round(lcrnrlon), round(urcrnrlon), 8.),
                    color='0.25', linewidth=0.5, labels=[False, False, False, True], size=tick_size)
    # Contour Plots
    contourfilled = m.contourf(xi, yi, ozone_interp, zorder=4, alpha=0.5, cmap="jet")

    # Colorbar
    cbar = plt.colorbar(contourfilled, shrink=colorbar_shrink, orientation='horizontal', pad=colorbar_padding)
    cbar.set_label("(ppb)", size=8)
    cbar.ax.tick_params(labelsize=colorbar_size)
    plt.title("Interpolated Observed MDA8 Ozone \n July 1, 2012", size=title_size)

    # Subplot 4: Overview of distribution of observed data vs. interpolated
    plt.subplot(2, 2, 4)

    # Get rid of any bad/nan interpolated values
    ozone_interp_clean = ozone_interp[ozone_interp > 0]

    # Histogram comparison of observed data distribution vs. interpolated data
    plt.hist(obs_arr, bins=50, histtype='stepfilled',
             normed=True, color='b', label='Observed MDA8')
    plt.hist(ozone_interp_clean, bins=50, histtype='stepfilled',
             normed=True, color='r', alpha=0.5, label='Interpolated MDA8')

    plt.legend(prop={'size': 8}, loc='upper left')
    plt.title("MDA8 Ozone Distribution \n July 1, 2012", size=title_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig("./images/Interpolated_%s_MDA8O3_20120701.png" % interpolation_method)

if __name__ == '__main__':
    # Read CSV with data points as pandas dataframe
    obs_df = pd.read_csv(obs_file, sep=',')
    # Get corners of spatial domain
    lcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = define_grid(obs_df)
    # Create basemap object along with grid
    m, obs_df, xi, yi = mk_basemap(obs_df, lcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)
    # Interpolate observation data to domain
    x, y, obs_arr, ozone_interp = interp_obs(obs_df, xi, yi, interpolation_method)
    # Create plots based on interpolation
    mk_plots(m, llcrnrlat, urcrnrlat, lcrnrlon, urcrnrlon, obs_df, xi, yi, ozone_interp, obs_arr, interpolation_method)
