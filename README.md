# EN: Classification of Histopathologic Scans of Lymph Node 
Sections Using Machine Learning

# HR: Klasifikacija histopatoloških snimaka dijelova limfnih čvorova pomoću strojnog učenja

Analiza i generiranje slike spadaju pod tipične probleme računalnog vida i strojnog učenja. S nedavnim razvojem dubokih modela pomaknute su granice izvedivoga i računala često uspijevaju dosenguti ljudsku točnost u analizi slika. Jedan od problema koji su uključeni u ovo područje jest klasifikacija slika.
Tema ovog rada je napraviti prototipno programsko rješenje na temelju slika histopatoloških skenova područja limfnih čvorova obavlja  binarnu klasifikaciju slika na temelju prisutnosti tkiva s metastazama.
U okviru rada potrebno je proučiti i opisati postojeće pristupe ovome problemu, ostvatiri prototipnu implementaciju sustava koji izvodi ovu zadaću te prikazati i ocijeniti dobivene rezultate. 
Radu priložiti izvorni kod razvijenih postupaka uz potrebna objašnjenja i dokumentaciju. Predložiti pravce budućeg razvoja. Citirati korištenu literaturu i navesti dobivenu pomoć.

## Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
For building this project system needs to have installed the following:
- [```git```](https://git-scm.com/)
- [```virtualenv```](https://virtualenv.pypa.io/en/latest/installation/)
- [```python3.6```](https://www.python.org/downloads/release/python-360/)
- [```pip```](https://pypi.org/project/pip/)


### Building and running example

Guide for setting up the project. 

It is recommended to work in a virtual environment and keep a list of required dependencies in a ```requirements.txt``` file. 

Commands to setup virtual environment with requirements.
```
virtualenv -p python3.6 env
source env/bin/activate
pip install -r requirements.txt
python setup.py
```

If you intend to develop part of this project you should use following command instead of last one.
```
python setup.py develop
```

### Windows specifics
1. install python 3.6 64 bit with pip
(if needed update pip ``` python3 -m pip install --upgrade pip ```)
2. install and create virtual environment  
```
pip3 install virtualenv
virtualenv -p python3 env
```
3. Activate environment  
```
\path\to\env\Scripts\activate.bat -- using CMD
\path\to\env\Scripts\activate.ps1 -- using PowerShell
```

Note: To create a virtualenv under a path with spaces in it on Windows, you’ll need the win32api library installed.

3. Install requirements and run tests
Install pytorch.  
```
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
```
Install other requirements.  
```
pip install -r requirements.txt
python setup.py install
```

4. Deactivate environment when needed  
```
.\env\Scripts\deactivate.bat
```

## Usage examples
For usage examples see examples in [histo/examples](https://github.com/domi385/FER/tree/master/histo/examples)

## Code style standards
In this repository we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/) as a standard for documentation and Flake8 for code sytle. Code style references are [Flake8](http://flake8.pycqa.org/en/latest/) and [PEP8](https://www.python.org/dev/peps/pep-0008/).

Commands to check flake8 compliance for written code and tests.
```
flake8 takepod
flake8 test
```

## Author

* Author: Domagoj Pluščec
* Project represents Master Thesis at Faculty of Electrical Engineering and Computing, University of Zagreb  
* Academic year 2018/2019  
* Faculty website: https://www.fer.unizg.hr/en/ 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## List of References