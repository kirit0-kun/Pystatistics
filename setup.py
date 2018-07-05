from distutils.core import setup

setup(name='Pystatistics', version="0.1", author="Moustafa Essam",
      description="A module to calculate statistical variables as well as statistical models",
      py_modules=["pystatistics"], requires=['PIL', 'matplotlib', 'numpy'])
