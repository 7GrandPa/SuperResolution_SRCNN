# SuperResolution_SRCNN
Super Resolution with SRCNN

+++++++++++++UPDATES++++++++++++++++
Version 1.0
1. complete basic function

Version 1.1
1. add some parser to set the test set data path.

Version1.2
1. we add a parser function to the code, allows you to specify test or validation phase, and do it
in which file.
parameters:
--phase: 'test' or 'validation'
--data_path: the path you want to do super resolution

e.x.
python validation.py --phase test --data_path G:\Jupyter\ImagePro\Code\Super_Resolution\Test\SetReal

+++++++++++PROBLEM LEFT++++++++++++++++
1. only do super resolution on factor 2
2. some border effect remains.

