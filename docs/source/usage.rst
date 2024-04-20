Usage
=====

Installation
------------

To use cuxrft, first install it using pip (via terminal):

.. code-block:: console

   $ pip install cuxrft

Please note, that cuxrft is not yet available via conda/Anaconda.

Basics and Scope
----------------
The package performs fast fourier transformations on graphics cards making use of cuda.
Therefor the data will be chunked, according to the maximal numpy array size that each graphic card (if multiple are available) can handle.
Making use of a recursive algorithm the computation of the FFT is going to be prepared in a delayed dask.array.
This allows each chunk to be computed on a different graphic card, for speeding up the calculations.
If only one graphic card is available, all chunks are going to be computed in a row.

For use of xarray see the well maintained `xarray documentation <https://docs.xarray.dev/en/stable/user-guide/index.html>`_.

Tutorial
========

Importing and making use of definitions
---------------------------------------
.. code-block:: python

   from cuxrft import fft_cellwise, ifft_cellwise
   import xarray as xr
   dataset = xr.open_dataset(path_to_dataset)
   datasetFFT = fft_cellwise(dataset)
   datasetiFFT = ifft_cellwise(dataset)

``fft_cellwise`` will perform FFTs for all data_vars and for all dimensions if no further arguments are passed to the method.
All dimensions that are transformed will be relabeled. A '_freq' is going to be appended to the dimension name. 
``ifft_cellwise`` will perform inverse FFTs in the same manner as ``fft_cellwise``. The names of the dimensions that have been transformed will be appended by '_ifft'.
If a '_freq' as a substring is found in a dimension name, this will be removed.

Importing as definition of xarray
---------------------------------
.. code-block:: python

   import cuxrft.xarray
   import xarray as xr
   dataset = xr.open_dataset(path_to_dataset)
   datasetFFT = dataset.fft_cellwise()
   datasetiFFT = dataset.ifft_cellwise()

The methods can now be used how mentioned in the paragraph above.

Arguments that can be passed to the definitions
-----------------------------------------------
   
``data``
""""""""

| *xarray.Dataset* or *xarray.DataArray*,
|   *xarray.Dataset*:             
|                       *xarray.Dataset* containing the provided ``FFT_dims`` and ``data_vars``.
|   *xarray.DataArray*:
|                       *xarray.DataArray* containing the provided ``FFT_dims``. ``data_vars`` will be ignored.

``chunks``
""""""""""

| *str* or *dict*,
|   *str*:
|           Must be ``chunks='auto'``. Chunk size will be determined automatically using the size of the numpy array and the smallest amount of memory of all gaphic cards supplied.
|   *dict*:
|           Shall contain the name of the dimension(s) to chunk along as the key and the size as the value.
|   Argument to define chunk size. 

``FFT_dims``
""""""""""""

| *str*, *list* or *dict*,
|   *str* or *list*:
|                       The dimension(s) to calculate the (i)FFT(s) along in data_var(s).
|   *dict*:
|                       The dimensions(s) to calculate the (i)FFT(s) along. Keys of ``FFT_dims`` are used internally as ``data_vars`` and values of ``FFT_dims`` as the dimensions to transform along.
|                       *Str* and *list* as values allowed.
|   Argument to define the dimension(s) to calculate the (i)FFT(s) along.
|   The returned *xarray.Dataset* or *xarray.DataArray* will contain these dimension(s) with the name(s) being prolonged by '_freq' - for FFT and by '_ifft' for iFFT.
|   If multiple data_vars are present in the ``data`` and the others are dependent on ``FFT_dims`` and will not be transformed a new dimension will be created.

``data_vars``
"""""""""""""

| *str* or *list*,
|   The name(s) of the data_var(s) to calculate the (i)FFT(s) from. Will be ignored if ``data`` is a *xarray.DataArray*.
|   If no value is provided, all data_vars in ``data`` will be used. Will be ignored if ``FFT_dims`` is a *dict*.

        
``delayed``
"""""""""""

| *bool*,
|   Wherether the returned *xarray.Dataset* or *xarray.DataArray* should be made up by *dask.delayed* arrays.
|   The GPU(s) are going to be reserved from python until the computation has been executed.

``multiple_GPUs``
"""""""""""""""""

| *bool*,
|   If multiple GPUs should be used. Even if ``GPUs`` is a *list*, this flag needs to be set ``multiple_GPUs=True`` to use all of the GPUs provided in the *list*.
|   If ``multiple_GPUs=True`` and the argument ``GPUs`` is not set or gets only one GPU, all GPUs will be used.
|   This flag starts a controlling server that manages the access of the GPUs.

``GPUs``
""""""""

| *list* or *int*,
|   Contains the index/indices of the GPU(s) to use.

``keepGPUcontrollingServerRunning``
"""""""""""""""""""""""""""""""""""
| *bool*,
|   Whether the GPU controlling server should continue to run, or not. Sometimes cuda does not closes itself properly.
|   By setting ``keepGPUcontrollingServerRunning=True`` this can be resolved for the case that multiple FFTs are supposed to be computed during different calls
|   of this method. This flag is only relevant, if ``multiple_GPUs=True``. Otherwise no GPU controlling server is going to be started.

``sel or isel``
"""""""""""""""

| *dict*,
|   Selects data in the same manner as xarray, as soon as FFT of the dim is done. Key must be the result dim of an FFT dim (*_freq).

Further defs:
-------------

``closeGPUController()``
""""""""""""""""""""""""

|   Closes the GPU controlling server, after making use of ``keepGPUcontrollingServerRunning=True`` or ``delayed=True``, when last operation that could require to calculate a chunk is done.

.. code-block:: python

   datasetFFT = fft_cellwise(dataset, delayed=True)
   someArbitraryMethods(datasetFFT)
   closeGPUController()
