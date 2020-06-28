# PyEMEBSDDI_wrapper
[**PyEMEBSDDI_wrapper**](https://github.com/Darkhunter9/PyEMEBSDDI_wrapper) is a high level wrapper for [**PyEMEBSDDI**](https://github.com/EMsoft-org/EMsoft/tree/develop/Source/EMsoftWrapperLib/DictionaryIndexing) module
under [**EMsoft**](https://github.com/EMsoft-org/EMsoft), providing Python wrappers for Dictionary Indexing (DI) functions.

The objective is to allow users to apply dictionary indexing and refinement conveniently inside Python.

While **PyEMEBSDDI** maintain the same arguments as original functions written in Fortran, **PyEMEBSDDI_wrapper** offers a more intuitive method to pass all parameters, high efficient multi-GPU support, and process EBSD patterns.


## Features
1. Using nml file to pass all parameters;
2. Provide useful EBSD pattern processing methods;
3. High efficient multi-GPU support for PyEMEBSDDI;


## Source
1. [**EMsoft**](https://github.com/EMsoft-org/EMsoft) github page;
2. [**PyEMEBSDDI**](https://github.com/EMsoft-org/EMsoft/tree/develop/Source/EMsoftWrapperLib/DictionaryIndexing) github page (under **EMsoft**)
3. [**PyEMEBSDDI_wrapper**](https://github.com/Darkhunter9/PyEMEBSDDI_wrapper) github page;
4. [**PyEMEBSDDI_wrapper**](https://pypi.org/project/PyEMEBSDDI-wrapper/) PyPI page;

[This site](http://vbff.materials.cmu.edu/EMsoft) conveniently brings all EMsoft stuff together in one place.

<!-- To obtain higher-level Python wrappers for PyEMEBSDDI, please refer to the project repository [**EBSDDI-CNN**](https://github.com/Darkhunter9/EBSDDI_CNN). -->


## Contributors
- [**Zihao Ding**](https://github.com/Darkhunter9)
- [**Elena Pascal**](https://github.com/elena-pascal)


## Installation
You should have **EMsoft** with **PyEMEBSDDI** built before installing **PyEMEBSDDI_wrapper**. Refer to [this site](https://github.com/EMsoft-org/EMsoft) and [this site](https://github.com/EMsoft-org/EMsoft/tree/develop/Source/EMsoftWrapperLib/DictionaryIndexing) for building directions.


Besides prerequisites in building [**EMsoft**](https://github.com/EMsoft-org/EMsoft) and [**EMsoftSuperbuild**](https://github.com/EMsoft-org/EMsoftSuperbuild), the following packages are also required:

| Package  | Version  |
| :------------ |:---------------|
| [Python](https://www.python.org/)      | &ge; 3.7    |
| [numpy](https://numpy.org/)            | &ge; 1.18.1 |
| [h5py](http://docs.h5py.org/en/stable/)            | &ge; 2.10.0   |
| [f90nml](https://pypi.org/project/f90nml/)         | &ge; 1.2      |
| [matplotlib](https://matplotlib.org/)              | &ge; 3.2.2    |
| [opencv](https://pypi.org/project/opencv-python/)  | &ge; 4.2.0.34 |

They can be installed through pip or conda to current Python environment.

Three ways to install PyEMEBSDDI_wrapper package (You need to provide the directory to EMsoft Bin folder):
1. Without downloaded package:

`pip install PyEMEBSDDI_wrapper --install-option="--EMsoftBinDIR=abs/dir/to/EMsoft/Bin"`

2. With downloaded package:

`pip install . --install-option="--EMsoftBinDIR=abs/dir/to/EMsoft/Bin"`

3. With downloaded package:

`python3 setup.py --EMsoftBinDIR="abs/dir/to/EMsoft/Bin"`


## How to use?

To import **PyEMEBSDDI_wrapper** into Python program:
```python
import PyEMEBSDDI_wrapper
# utils functions
from PyEMEBSDDI_wrapper.utils import *
# two main high-level wrapper functions
from PyEMEBSDDI_wrapper import PyEMEBSDDI_wrapper, PyEMEBSDRefine_wrapper
```


## API Reference
The module is composed of two parts: high-level wrapper functions and utility functions. 

Utility functions include conversion between Euler angles and quaternions, and many useful pattern processing methods.

API list:

**`PyEMEBSDDI_wrapper.utils`**
1. `eu2qu(eu)`
2. `qu2eu(qu)`
3. `binning(img, bsize=(2, 2))`
4. `circularmask(img)`
5. `squaremask(img)`
6. `bac(img, a=1, b=0)`
7. `autobac(img)`
8. `gamma_trans(img, gamma)`
9. `autogamma(img)`
10. `clahe(img, limit=10, tileGridSize=(10, 10))`
11. `poisson_noise(img, c=1.)`

**`PyEMEBSDDI_wrapper`**
1. `PyEMEBSDDI_wrapper(epatterns, dpatterns, orientations, nml_dir, epatterns_processing=None, dpatterns_processing=None, gpu=None, refine=False)`
2. `PyEMEBSDRefine_wrapper(epatterns, startOrientations, startdps, variants, nml_dir, h5_dir, epatterns_processing=None)`

**`PyEMEBSDDI_wrapper.utils`**
--------------------------------

`eu2qu(eu)`

Convert Euler angles group to quaternions. Default value of eps = 1.

Input:
- Euler angles: array-like, float, (3,);
  
Output:
- Quaternions, 1darray, float, (4,);

`qu2eu(qu)`

Convert quaternions to Euler angles group. Default value of eps = 1.

Input:
- Quaternions: array-like, float, (4,);

Output:
- Euler angles, 1darray, float, (3,);

`binning(img, bsize=(2, 2))`

Bin images.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);
- bsize: list of 2 ints, bin size at axis of row and column;

Output:
- img_binning: 3darray, 8bit, (n, floor(numsx/bsize[0]), floor(numsy/bsize[1]));

`circularmask(img)`

Apply circularmask to images.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);

Output:
- img_with_mask: 3darray, 8bit, (n, numsx, numsy);

`squaremask(img)`

Apply squaremask to images. Will change the size.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);

Output:
- img_with_mask: 3darray, 8bit, (n, numsx_mask, numsy_mask);

`bac(img, a=1, b=0)`

Adjust brigtness and contrast of images. img = a*img+b.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);
- a: contrast coefficient, float;
- b: brigtness coefficient, float;

Output:
- img_bac: 3darray, 8bit, (n, numsx, numsy);

`autobac(img)`

Automatically adjust brigtness and contrast of images.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);

Output:
- img_bac: 3darray, 8bit, (n, numsx, numsy);

`gamma_trans(img, gamma)`

Adjust gamma value of images.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);
- gamma: gamma coefficient;

Output:
- img_gamma: 3darray, 8bit, (n, numsx, numsy);

`autogamma(img)`

Automatically adjust gamma value of images.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);

Output:
- img_gamma: 3darray, 8bit, (n, numsx, numsy);

`clahe(img, limit=10, tileGridSize=(10, 10))`

Apply contrast limit adaptive histogram equalization.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);
- limit: limit of count in each value bin;
- tileGridSize: tuple of 2 ints, number of grid at axis of row and column;

Output:
- img_clahe: 3darray, 8bit, (n, numsx, numsy);

`poisson_noise(img, c=1.)`

Apply poisson noise on images.

Input:
- img: 3darray, 8bit, (n, numsx, numsy);
- c: poisson coefficient. Smaller c brings higher noise level;

Output:
- img_poisson: 3darray, 8bit, (n, numsx, numsy);


**`PyEMEBSDDI_wrapper`**
--------------------------------
Two NML templates `PyEMEBSDDI.nml` and `PyEMEBSDRefine.nml` are provided in the package.

`PyEMEBSDDI_wrapper(epatterns, dpatterns, orientations, nml_dir, epatterns_processing=None, dpatterns_processing=None, gpu=None, refine=False)`

High-level wrapper for PyEMEBSDDI.

Input:
- epatterns: experimental patterns, 3darray, 8bit, (n, numsx, numsy);
- dpatterns: dictionary patterns, 3darray, 8bit, (n, numsx, numsy);
- orientations: orientations of dictionary patterns, unit quaternions, 2darray, float, (n, 4);
- nml_dir: PyEMEBSDDI nml file dir, string;
- epatterns_processing: img processing recipe for epatterns, list of strings;
- dpatterns_processing: img processing recipe for dpatterns, list of strings;

  All methods in `PyEMEBSDDI_wrapper.utils` can be applied;

  Example: `['clahe(10, (4, 4))','circularmask()',]` (exclude `img` parameter from each method);

- gpu: multiple gpu choice, None or list of device id (int);

  If single gpu, `gpu = None` or `len(gpu) = 1`, actual gpu used is determined by devid (ipar[5]);

  If multiple gpu, gpu is the list of devid of all gpu used;

  Attention: full use of multiplt gpu needs more cpu resource and memory!

- refine: whether to refine indexing, bool, currently unavailable;

Output:

`[pred, resultmain]`
- pred: orientations output by PyEMEBSDDI, 2darray, float, (n,4);
- resultmain: dot products for each orientation prediction, 1darray, float, (n,);

`PyEMEBSDRefine_wrapper(epatterns, startOrientations, startdps, variants, nml_dir, h5_dir, epatterns_processing=None)`

High-level wrapper for PyEMEBSDRefine.

Input:
- epatterns: experimental patterns, 3darray, 8bit, (n, numsx, numsy);
- startOrientations: initial orientation predictions in form of unit quaternions, 2darray, float, (n, 4);
- startdps: initial dot products for each orientation prediction, 1darray, float, (n,);
- variants: quaternions defining the potential pseudosymmetry variants, 2darray, float, (m, 4);

  default value is `np.array([[1,0,0,0]])` (pseudosymmetry not involved);

- nml_dir: nml file dir, string;
- h5_dir: h5 file dir that stores master pattern data, generated by EMEBSD, string;
- epatterns_processing: img processing recipe for epatterns, list of strings;

Output:
- orientations output by PyEMEBSDRefine, 2darray, float, (n,4);


## Contribute
Have you spotted a typo, would you like to fix a bug, or is there something you’d like to suggest? You are welcome to open a pull request. We will do our best to review your proposal in due time.

In addition, you can also email [**Zihao**](mailto:ding@cmu.edu) should you have any questions or suggestions.

## Credits
We want to express our sincere thanks to those who provided help during the development of this project (in no particular order):

- Saransh Singh
- Michael Jackson

## License
```
BSD 3-Clause License

Copyright (c) 2020, Zihao Ding, Marc De Graef Research Group/Carnegie Mellon University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```