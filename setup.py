from setuptools import setup

setup(
    name             = 'framenet',
    version          = '0.1.0',
    packages         = ['framenet'],
    install_requires = ['pandas>=0.18', 'multipledispatch>=0.4.8', 'decorator'],
    url              = 'https://github.com/icsi-berkeley/framenet',
    license          = 'MIT',
    author           = 'Sean Trott, Luca Gilardi',
    author_email     = 'seantrott@berkeley.edu, lucag@icsi.berkeley.edu',
    description      = 'A FrameNet/ECG Exploration Tool'
)
