# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:49:16 2017

@author: Ali Aghaeifar
"""


from setuptools import setup


setup(name='recotwix', # this will be name of package in packages list : pip list 
      version='0.2.0',
      description='Reconstruction utilities for siemens twix data',
      keywords='twix,reconstruction,mri,nifti',
      author='Ali Aghaeifar',
      author_email='ali.aghaeifar [at] tuebingen.mpg [dot] de',
      license='MIT License',
      packages=['recotwix'],
      install_requires = ['tqdm','numpy','nibabel','torch','twixtools']
     )
