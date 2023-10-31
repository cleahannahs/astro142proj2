#
# Clea Hannahs
# Astro 142 UCLA Fall 2023
# Project 2
#

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vo import Vizier
import logging

# Set up the logging facility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Hubble UDF image data
def load_udf_data(image_path):
    """Load the Hubble UDF image data.

    Args:
        image_path (str): The path to the Hubble UDF image FITS file.

    Returns:
        image_data (numpy.ndarray): The image data.
        wcs (astropy.wcs.WCS): The WCS information for the image.
    """
    try:
        hdu = fits.open(image_path)
        image_data = hdu[0].data
        header = hdu[0].header
        wcs = WCS(header)
        hdu.close()
        return image_data, wcs
    except Exception as e:
        logging.error(f"Error loading Hubble UDF image: {str(e)}")
        raise

# Create a 3-color RGB image
def create_rgb_image(data):
    """Create a 3-color RGB image.

    Args:
        data (numpy.ndarray): Image data to be used for RGB channels.

    Returns:
        rgb_image (numpy.ndarray): The RGB image.
    """
    r = data
    g = data
    b = data
    rgb_image = make_lupton_rgb(r, g, b, stretch=1.0, Q=1.0)
    return rgb_image

# Load the galaxy catalog with photo-z measurements
def load_photometric_catalog(catalog_path):
    """Load the galaxy catalog with photometric redshift measurements.

    Args:
        catalog_path (str): The path to the photometric catalog FITS file.

    Returns:
        catalog (astropy.table.Table): The photometric catalog.
    """
    try:
        catalog = Table.read(catalog_path)
        return catalog
    except Exception as e:
        logging.error(f"Error loading photometric catalog: {str(e)}")
        raise

# Load the galaxy catalog with spectroscopic redshifts
def load_spectroscopic_catalog(catalog_path):
    """Load the galaxy catalog with spectroscopic redshift measurements.

    Args:
        catalog_path (str): The path to the spectroscopic catalog FITS file.

    Returns:
        catalog (astropy.table.Table): The spectroscopic catalog.
    """
    try:
        catalog = Table.read(catalog_path)
        return catalog
    except Exception as e:
        logging.error(f"Error loading spectroscopic catalog: {str(e)}")
        raise

# Load data from a Virtual Observatory (VO) catalog
def load_vo_data(vo_catalog_name, field_coordinates, vo_radius):
    """Load data from a Virtual Observatory (VO) catalog.

    Args:
        vo_catalog_name (str): The name of the VO catalog to query.
        field_coordinates (str): Field coordinates in the format "RA Dec" (e.g., "53.16079 -27.79236").
        vo_radius (float): The query radius in degrees.

    Returns:
        data_table (astropy.table.Table): The data from the VO catalog.
    """
    try:
        # Define the VO catalog service and query
        catalog_service = Vizier(columns=['RAJ2000', 'DEJ2000', 'Halpha', 'Hbeta', 'OIII', 'NII'])
        catalog_service.TIMEOUT = 600  # Increase timeout if necessary

        # Query the catalog for your data
        query_result = catalog_service.query_region(field_coordinates, radius=vo_radius, catalog=[vo_catalog_name])

        # Access the resulting data table
        data_table = query_result[0]
        return data_table
    except Exception as e:
        logging.error(f"Error loading data from VO catalog: {str(e)}")
        raise

# Overplot galaxy detections with indicators for photo-z and spectro-z
def plot_galaxy_detections(rgb_image, photometric_catalog, spectroscopic_catalog, wcs, subregions):
    """Plot galaxy detections with indicators for photometric and spectroscopic redshifts.

    Args:
        rgb_image (numpy.ndarray): The RGB image.
        photometric_catalog (astropy.table.Table): The photometric catalog.
        spectroscopic_catalog (astropy.table.Table): The spectroscopic catalog.
        wcs (astropy.wcs.WCS): The WCS information for the image.
        subregions (list): List of subregions with coordinates and inset positions.

    Returns:
        None
    """
    fig, ax = plt.subplots(subplot_kw={'projection': wcs})
    ax.imshow(rgb_image, origin='lower')

    # Convert catalog coordinates to SkyCoord for cross-matching
    photometric_coords = SkyCoord(photometric_catalog['RA'], photometric_catalog['Dec'], unit=(u.degree, u.degree))
    spectroscopic_coords = SkyCoord(spectroscopic_catalog['RA'], spectroscopic_catalog['Dec'], unit=(u.degree, u.degree))

    # Loop through the photometric catalog and plot galaxy detections
    for i, galaxy in enumerate(photometric_catalog):
        # Check if the galaxy's coordinates are in the spectroscopic catalog
        match = photometric_coords.separation(SkyCoord(galaxy['RA'], galaxy['Dec'], unit=(u.degree, u.degree))) <= 1 * u.arcsec

        # Extract galaxy coordinates and other relevant information
        ra = galaxy['RA']
        dec = galaxy['Dec']
        x, y = wcs.wcs_world2pix(ra, dec, 0)

        if match.any():
            ax.plot(x, y, 'o', markersize=5, label=f'z={galaxy["PhotoZ"]}', color='blue', markerfacecolor='white', markeredgewidth=1, markeredgecolor='blue', alpha=0.8)
            ax.annotate("S", (x, y), xycoords='data', textcoords='offset points', xytext=(0, 5), color='blue', fontsize=10)
        else:
            ax.plot(x, y, 'o', markersize=5, label=f'z={galaxy["PhotoZ"]}', color='red', alpha=0.8)

        # Create an inset view for selected subregions
        for subregion in subregions:
            if subregion['x1'] <= x <= subregion['x2'] and subregion['y1'] <= y <= subregion['y2']:
                inset_axes = fig.add_axes([subregion['left'], subregion['bottom'], subregion['width'], subregion['height'], i+1])
                inset_axes.imshow(rgb_image, extent=[subregion['x1'], subregion['x2'], subregion['y1'], subregion['y2']], origin='lower')
                inset_axes.set_xticks([])
                inset_axes.set_yticks([])
                inset_axes.annotate(f'({chr(97 + i)})', (0.02, 0.95), xycoords='axes fraction', fontsize=12, color='white')

    ax.set_xlabel('Right Ascension (J2000)')
    ax.set_ylabel('Declination (J2000)')
    plt.legend(loc='upper right')

    plt.show()

def main():
    try:
        # Set the paths and parameters. Replace with real things
        image_path = "udf_image.fits"
        catalog_path_photometric = "photometric_catalog.fits"
        catalog_path_spectroscopic = "spectroscopic_catalog.fits"

        image_data, wcs = load_udf_data(image_path)
        rgb_image = create_rgb_image(image_data)
        photometric_catalog = load_photometric_catalog(catalog_path_photometric)
        spectroscopic_catalog = load_spectroscopic_catalog(catalog_path_spectroscopic)

        # Define the subregions as dictionaries with coordinates and inset positions
        subregions = [
            {'x1': 10, 'x2': 100, 'y1': 20, 'y2': 110, 'left': 0.1, 'bottom': 0.6, 'width': 0.2, 'height': 0.2},
            {'x1': 200, 'x2': 300, 'y1': 200, 'y2': 300, 'left': 0.6, 'bottom': 0.6, 'width': 0.2, 'height': 0.2}
        ]

        plot_galaxy_detections(rgb_image, photometric_catalog, spectroscopic_catalog, wcs, subregions)
    except Exception as e:
        logging.error(f"An error occurred: {str(e}")

if __name__ == '__main__':
    main()
