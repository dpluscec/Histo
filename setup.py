from setuptools import setup, find_packages
import histo

setup(name='histo',
      version=histo.__version__,
      description='Classification of Histopathologic Scans of Lymph Node Sections Using '\
                  'Machine Learning',
      author='Domagoj Pluscec',
      author_email='domagoj.pluscec@fer.hr',
      license='MIT',
      packages=find_packages(),
      url="https://github.com/domi385/FER",
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      package_data={},
      zip_safe=False)
