from distutils.core import setup, Extension
import numpy

shift = Extension('PotentialFilter',
                  ['potential_filter.c'],
                  include_dirs=[numpy.get_include()]
)

setup(name='PotentialFilter',
      ext_modules=[shift]
)