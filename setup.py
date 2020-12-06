"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

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

Three ways to install PyEMEBSDDI_wrapper package:
With downloaded package:
(1) python3 setup.py install --EMsoftBinDIR="abs/dir/to/EMsoft/Bin"
(2) pip install . --install-option="--EMsoftBinDIR=abs/dir/to/EMsoft/Bin"
Without downloaded package:
(3) pip install PyEMEBSDDI_wrapper --install-option="--EMsoftBinDIR=abs/dir/to/EMsoft/Bin"
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools.command.install import install
# To use a consistent encoding
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

class InstallCommand(install):
    user_options = install.user_options + [
        # ('someopt', None, None), # a 'flag' option
        #('someval=', None, '<description for this custom option>') # an option that takes a value
        ('EMsoftBinDIR=', None, 'directory of EMsoft Bin folder.'),
    ]                                      

    def initialize_options(self):
        install.initialize_options(self)
        # self.someopt = None
        # self.someval = None
        self.EMsoftBinDIR = None

    def finalize_options(self):
        print("Directory of EMsoft Bin folder is:", self.EMsoftBinDIR)
        install.finalize_options(self)

    def run(self):
        if not self.EMsoftBinDIR:
            raise ValueError("EMsoftBinDIR not input: Please use pip install PyEMEBSDDI_wrapper --install-option='--EMsoftBinDIR=abs/dir/to/EMsoft/Bin'")
        elif not path.exists(self.EMsoftBinDIR):
            raise ValueError('EMsoftBinDIR does not exist: %s' % self.EMsoftBinDIR)
        elif path.exists(path.join(self.EMsoftBinDIR, 'PyEMEBSDDI.so')):
            with open(path.join(here, 'PyEMEBSDDI_wrapper', 'EMsoft_DIR.txt'), mode='w') as f:
                f.write(self.EMsoftBinDIR)
        elif path.exists(path.join(self.EMsoftBinDIR, 'Bin', 'PyEMEBSDDI.so')):
            with open(path.join(here, 'PyEMEBSDDI_wrapper', 'EMsoft_DIR.txt'), mode='w') as f:
                f.write(path.join(self.EMsoftBinDIR, 'Bin'))
        else:
            raise ValueError('PyEMEBSDDI.so does not exist under EMsoftBinDIR: %s' % self.EMsoftBinDIR)
        install.run(self)


# solve EMsoftBinDIR using sys.argv
# prepared for python3 setup.py install --EMsoftBinDIR dir
# not required if rewrite install func

# if '--EMsoftBinDIR' in sys.argv:
#     index = sys.argv.index('--EMsoftBinDIR')
#     sys.argv.pop(index)
#     EMsoftBinDIR = sys.argv.pop(index)
#     with open(path.join(here, 'PyEMEBSDDI_wrapper', 'EMsoft_DIR.txt'), mode='w') as f:
#         f.write(EMsoftBinDIR)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='PyEMEBSDDI_wrapper',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Python high level wrapper for PyEMEBSDDI under EMsoft',  # Optional

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/Darkhunter9/PyEMEBSDDI_wrapper',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='Zihao Ding',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='ding@cmu.edu',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3.7',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='',  # Optional

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir=dict(),  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=['PyEMEBSDDI_wrapper', 'PyEMEBSDDI_wrapper/utils'],  # Required

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.7',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'h5py', 'f90nml', 'matplotlib', 'opencv-python'],  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    # extras_require={  # Optional
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={  # Optional
        # '': ['README.md', 'LICENSE', 'requirements.txt'],
        'PyEMEBSDDI_wrapper': ['README.md', 'LICENSE', 'requirements.txt', 'PyEMEBSDDI.nml', 'PyEMEBSDRefine.nml', 'EMsoft_DIR.txt'],
    },

    # data_files=[('PyEMEBSDDI_wrapper', ['README.md', 'LICENSE', 'requirements.txt', 'PyEMEBSDDI.nml', 'PyEMEBSDRefine.nml']),],

    include_package_data=True,

    cmdclass={'install': InstallCommand},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/Darkhunter9/PyEMEBSDDI_wrapper/issues',
        'Funding': 'http://vbff.materials.cmu.edu/EMsoft/',
        'Source': 'https://github.com/Darkhunter9/PyEMEBSDDI_wrapper',
    },
)