import cuxrft
import xarray as xr

try:
    @xr.register_dataset_accessor("cuxrft")
    class cuxrft:
        def __init__(self, xarray_obj):
            self._obj = xarray_obj
        def fft_cellwise(self, *kwargs):
            return cuxrft.fft_cellwise(self._obj, *kwargs)
        def ifft_cellwise(self, *kwargs):
            return cuxrft.ifft_cellwise(self._obj, *kwargs)
    @xr.register_dataarray_accessor("cuxrft")
    class cuxrft:
        def __init__(self, xarray_obj):
            self._obj = xarray_obj
        def fft_cellwise(self, *kwargs):
            return cuxrft.fft_cellwise(self._obj, *kwargs)
        def ifft_cellwise(self, *kwargs):
            return cuxrft.ifft_cellwise(self._obj, *kwargs)
except AttributeError:
    print('Could not provide cuxrft methods as internal functions of xarray.')
