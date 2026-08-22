"""Real dataset loaders (Sandia, spectra) - STUBBED FOR FUTURE USE.

Currently intentionally not implemented. These will be uncommented and
wired up once real datasets are ready for integration.
"""

# TODO: Sandia validation dataset loader
# Files: src/RK4TRAIN/ml/data/sandia_validation/PV_Module_Operating_Temperature_Data.nc
# Format: NetCDF with temperature measurements
# To implement:
#   - Load NetCDF file
#   - Extract time, location, weather, T_operating
#   - Align with RK4TRAN format
#
# class SandiaLoader:
#     """Load Sandia PV validation dataset."""
#     def __init__(self, nc_path):
#         pass
#
#     def load(self):
#         pass


# TODO: Spectral data loader
# Files: src/RK4TRAIN/ml/data/spectra/AM0AM1_5.xls
# Format: Excel with spectral irradiance
# To implement:
#   - Load Excel file
#   - Extract spectral data
#   - Convert to broadband irradiance
#
# class SpectraLoader:
#     """Load spectral irradiance data."""
#     def __init__(self, xls_path):
#         pass
#
#     def load(self):
#         pass


def load_real_datasets():
    """Placeholder for loading all real datasets.

    Returns:
        Dict with dataset loaders (empty for now)
    """
    return {}
