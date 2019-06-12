"""
Classification of Histopathologic Scans of Lymph Node Sections Using Machine Learning

When analyzing histopathological images, there is a need for an automated system that
could help doctors in image analysis and diagnostics. Such a system could increase the
accuracy and speed of analysis and diagnostics.
Within this paper, an overview of the histopathological image analysis area is provided
and a program implementation for the classification of lymph nodes based on
machine learning has been developed. 
PatchCamelyon dataset has been used for training and testing of chosen
machine learning models. 
The results of the following deep learning models have been studied: AlexNet, ResNet,
DenseNet and Inceptionv3. Also, the influence of different data augmentation methods on
the model performance was investigated.
"""

import logging
import os
import time
__name__ = "histo"
__version__ = "0.1.0"


LOG_FILENAME = os.path.join('logs', f"{str(int(time.time()))}.lg")
LOG_FORMAT = "%(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.FileHandler(filename=LOG_FILENAME, encoding="utf-8"))
LOGGER.setLevel(logging.INFO)
