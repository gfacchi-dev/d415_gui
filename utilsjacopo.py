import os 
import pickle
import numpy as np

def check_calibration()-> bool:
    """
    Checks if the calibration file is present in the calibration folder.
    :return: True if the file is present, False otherwise.
    """
    dirs = os.listdir("./calibrations")
    if len(dirs) == 0:
        return False
    else:
        return True

def get_calibration()-> tuple:
    """
    Returns the last calibration matrices.
    :return: (M_L2C, M_R2C)
    """
    dirs = os.listdir("./calibrations")
    dirs.sort(reverse=True)
    matrice_calibrazione_L2C = pickle.load(open("./calibrations/"+dirs[0]+"/cal_mat_L2C.mat", "rb"))
    matrice_calibrazione_R2C = pickle.load(open("./calibrations/"+dirs[0]+"/cal_mat_R2C.mat", "rb"))
    # TODO: check if the last calibration is enough recent
    return matrice_calibrazione_L2C, matrice_calibrazione_R2C


calibrate_text = """"Assicurarsi di posizionare l'oggetto di calibrazione in modo che sia visibile da tutte le camere.\n
Quando si è pronti, premere il tasto "Calibra" per iniziare la procedura di calibrazione.\n"""

acquire_text = """"Assicurarsi che il soggetto sia visibile da tutte le camere.\n"""