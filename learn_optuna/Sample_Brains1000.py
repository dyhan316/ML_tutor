import logging
import os
import io
import requests
import numpy as np
from tvb.simulator.lab import *

logger = logging.getLogger(__name__)


import os
from glob import glob
from scipy.io.matlab import loadmat
import numpy as np
import pandas as pd
import re

#data_root = os.path.abspath(
#        os.path.join(
#            os.path.dirname(
#                os.path.dirname(__file__)
#            ),
#            'data'
#        )
#)

data_root = '/scratch/connectome/dyhan316/TVB/TVB_showcase/virtual_aging_showcase/data'

class Brains1000Dataset: #(DataProxyConnectivityDataset):
    """
    Caspers, S. et al (2021).
    1000BRAINS study, connectivity data. 

    v1.0: https://doi.org/10.25493/61QA-KP8
    v1.1: https://doi.org/10.25493/6640-3XH
    """

    ds_root     = os.path.join(data_root, "external", "Julich")
    ds_external = os.path.join(data_root, "external")

    def __init__(self):
        self.data_root = data_root
        
    def list_subjects(self):
        d = os.path.join(self.ds_root)
        return [subj for subj in os.listdir(d) if not subj.startswith(".")]

    
    def load_sc(self, subj, log10=False):
        
        d = os.path.join(self.ds_root)
        
        separator          = ''
        file_julich        = separator.join([d,'/',subj,'/ses-1/SC/',subj,'_SC_Schaefer_7NW100p.txt']) 
        file_julich_no_log = separator.join([d,'/',subj,'/ses-1/SC/',subj,'_SC_Schaefer7NW100p_nolog10.txt']) 
        SC                 = np.loadtxt(file_julich)
        SC_nolog           = np.loadtxt(file_julich_no_log)
        
        if log10:
            return SC
        else:
            return SC_nolog

    def get_connectivity(self, subject, scaling_factor=124538.470647693):
        SC = self.load_sc(subject)
        SC = SC / scaling_factor
        conn = connectivity.Connectivity(
                weights = SC,
                tract_lengths=np.ones_like(SC),
                centres = np.zeros(np.shape(SC)[0]),
                speed = np.r_[np.Inf]
        )
        conn.compute_region_labels()
        logger.warning("Placeholder region names!")
        return conn


