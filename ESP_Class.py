# -*- coding: utf-8 -*-

"""
The ESP calculation module contains two classes for computing inputs performance under single-phase water/viscous
fluid flow or gas-liquid two-phase flow conditions.

The two-phase model was originally proposed by Dr. Zhang, TUALP ABM (2013) and later revised by Zhu (2017). Only the 
simplified version is programmed.

Original version:    1st, Aug, 2017
Developer:  Jianjun Zhu

Current version:    30, Mar, 2022
Developer: Haiwen Zhu
"""

import sys
import os
path_abs = os.path.dirname(os.path.abspath(__file__))   # current path
path = os.path.join(path_abs, 'C:\\Users\\haz328\\Desktop\\Github') # join path
sys.path.append(path)  # add to current py
from Common import *
from Utility import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# G = 9.81
# pi = np.pi
# E1 = 10e-3   # for gas
# E2 = 1e-7   # for single phase
# psi_to_pa = 1.013e5 / 14.7
# psi_to_ft = 2.3066587368787
# bbl_to_m3 = 0.15897
# bpd_to_m3s = 0.15897 / 24.0 / 3600.0

class ESP_TUALP(object):
    def __init__(self, ESP_GEO='Default', QBEM=5000, Coef='Default', 
        EMULSION_MODEL='tualp_2022', SGL_MODEL='tualp_2022', GL_MODEL='tualp_2022',F_MODEL='tualp_2022'):

        '''Flow condition'''
        self.QBEM = QBEM
        self.DENW = 997.                 # water density
        self.VISW = 1e-3                 # water viscosity

        '''ESP geometry'''
        if ESP_GEO == 'Default':
            ESP_GEO = {
                "R1": 0.017496,     "R2": 0.056054,     "TB": 0.00272,      "TV": 0.00448,      "RD1": 0.056,
                "RD2": 0.017496,    "YI1": 0.012194,    "YI2": 0.007835,    "VOI": 0.000016119, "VOD": 0.000011153,
                "ASF": 0.00176464,  "ASB": 0.00157452,  "AB": 0.001319,     "AV": 0.001516,     "ADF": 0.001482,
                "ADB": 0.000935,    "LI": 0.076,        "LD": 0.08708,      "RLK": 0.056209,    "LG": 0.00806,
                "SL": 0.00005,      "EA": 0.000254,     "ZI": 5,            "ZD": 9,            "B1": 19.5,
                "B2": 24.7,         "NS": 1600,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3500,          "SGM": 0.3,
                "QL": 2700,         "QG": 500,           "GVF": 10,          "WC": 0.,           "SN": 1,
                "P": 150,           "T":100,            "SGL":'zhang_2016', "GL":'None'
            }

        self.R1 = ESP_GEO['R1']
        self.R2 = ESP_GEO['R2']
        self.RD1 = ESP_GEO['RD1']
        self.RD2 = ESP_GEO['RD2']
        self.TB = ESP_GEO['TB']
        self.TV = ESP_GEO['TV']
        self.YI1 = ESP_GEO['YI1']
        self.YI2 = ESP_GEO['YI2']
        self.VOI = ESP_GEO['VOI']
        self.VOD = ESP_GEO['VOD']
        self.ASF = ESP_GEO['ASF']
        self.ASB = ESP_GEO['ASB']
        self.AB = ESP_GEO['AB']
        self.AV = ESP_GEO['AV']
        self.ADF = ESP_GEO['ADF']
        self.ADB = ESP_GEO['ADB']
        self.LI = ESP_GEO['LI']
        self.LD = ESP_GEO['LD']
        self.RLK = ESP_GEO['RLK']
        self.LG = ESP_GEO['LG']
        self.SL = ESP_GEO['SL']
        self.EA = ESP_GEO['EA']
        self.ZI = ESP_GEO['ZI']
        self.ZD = ESP_GEO['ZD']
        self.B1 = ESP_GEO['B1'] * (pi / 180.0)
        self.B2 = ESP_GEO['B2'] * (pi / 180.0)
        self.NS = ESP_GEO['NS']
        self.SN = ESP_GEO['SN']
        a1 = 2 * pi * self.R1 / self.ZI * np.sin(self.B1)
        b1 = self.YI1
        a2 = 2 * pi * self.R2 / self.ZI * np.sin(self.B2)
        b2 = self.YI2
        self.DH1 = 2 * a1 * b1 / (a1 + b1)
        self.DH2 = 2 * a2 * b2 / (a2 + b2)
        self.DH = (self.DH1 + self.DH2) / 2.0
        self.beta = (self.B1 + self.B2) / 2.0
        self.YI = (self.YI1 + self.YI2) / 2.0
        self.RI = (self.R1 + self.R2) / 2.0
        self.AIW = self.AB + self.ASF + self.ASB
        self.ADW = self.AV + self.ADF + self.ADB

        
        # self.F_MODEL = F_MODEL
        self.EMULSION_MODEL = EMULSION_MODEL
        self.SGL_MODEL = SGL_MODEL
        self.GL_MODEL = GL_MODEL
        self.F_MODEL = F_MODEL

        
        '''single phase coefficient'''
        if Coef == 'Default':
            Coef = {
                "FTI": 3, "FTD": 3, "FTI_coef": 0.01, "FTD_coef": 0.01, "F_leakage":0.25, "SGMU_coef":175, 
                "SGM_coef":0.1 , "SGM_coef_2":0.25*2, "SGMU_coef_2": 0.5, "QBEM_VISL_coef":0.01,
                "emulsion_E": 3.0, "emulsion_SN": 0.01, "emulsion_WE":0.1, "emulsion_RE": 0.1, "emulsion_ST1": 2.5, "emulsion_ST2":0.2,
                "alphaG_crit_coef":4, "alphaG_crit_critical":0.25, "factor": 1.4, "DB_GVF_EFF": 1.02, "DB_NS_EFF": 2, "CD_Gas_EFF": 0.62,    
                "CD_Liquid_EFF": 1, "CD_INT_EFF": 1.5, "CD_GV_EFF": 2, "SGL":'zhu_2018', "GL":'None', "flg": 'F', "train_type": 'ALL_geometry'
            }

        self.FTI = Coef["FTI"]       # Original turning factor 3
        self.FTD = Coef["FTD"]       # Original turning factor 3
        self.FTI_coef = Coef["FTI_coef"]       # Original turning factor coef 0.1
        self.FTD_coef = Coef["FTD_coef"]       # Original turning factor coef 0.1
        self.F_leakage = Coef["F_leakage"]       # Original leakage head loss coef 0.25
        self.SGM_coef = Coef["SGM_coef"]    # ori 0.1
        self.SGM_coef_2 = Coef["SGM_coef_2"]    # ori 0.25
        self.SGMU_coef = Coef["SGMU_coef"]    # ori 175
        self.SGMU_coef_2 = Coef["SGMU_coef_2"]    # ori 0.01
        self.QBEM_VISL_coef = Coef["QBEM_VISL_coef"]    # ori 0.01

        '''emulsion coefficient'''
        self.emulsion_E = Coef["emulsion_E"]    # ori 3.0 
        self.emulsion_SN = Coef["emulsion_SN"]    # ori 0.01 
        self.emulsion_WE = Coef["emulsion_WE"]    # ori 0.1
        self.emulsion_RE = Coef["emulsion_RE"]    # ori 0.1 
        self.emulsion_ST1 = Coef["emulsion_ST1"]    # ori 2.5 
        self.emulsion_ST2 = Coef["emulsion_ST2"]    # ori 0.2

        '''GL ccoefficient'''
        self.factor = Coef["factor"]     # Zhu DB 1.4 for all;        2.1/1.8 for Flex31 surging/GVF erosion compare , 1.2 TE2700 surging
        self.alphaG_crit_coef = Coef["alphaG_crit_coef"]
        self.alphaG_crit_critical = Coef["alphaG_crit_critical"]     #0.5 - alphaG_crit_critical*(np.exp(-(N/3500.0)))**alphaG_crit_coef

        self.DB_GVF_EFF = Coef["DB_GVF_EFF"]        # 1.02 good for all
        self.DB_NS_EFF = Coef["DB_NS_EFF"]    ## factor used in gas bubble size  2 good for all      

        self.CD_Gas_EFF = Coef["CD_Gas_EFF"]        # 1 good for all, 0.65 good for Flex31 GVF
        self.CD_Liquid_EFF = Coef["CD_Liquid_EFF"]       # Original drag coefficiet for gas velocity in modified Sun for INT flow (GVF same to CD_gas)
        self.CD_INT_EFF = Coef["CD_INT_EFF"]         # drag coefficient      1.5 for all,    1.2 for surging,  3 for Flex31 GVF, 1.8 TE2700 surging
        self.CD_GV_EFF = Coef["CD_GV_EFF"]           # drag coefficient for GV

        '''model selection'''
        self.sgl_model = Coef['SGL']
        self.flg = Coef['flg']

    def f0_Churchill_1977(self, Q, DENL, VISL):
        NRe = self.DH * Q * DENL / (2 * np.pi * self.RI * self.YI * VISL * np.sin(self.beta))

        # if NRe < 2300:
        #     return 64.0 / NRe
        # else:
        #     return 8 * (2.457 * np.log(1. / ((7. / NRe) ** 0.9 + 0.27 * (self.EA / self.DH)))) ** (-2)

        f1 = 64.0 / NRe
        f2 = 8 * (2.457 * np.log(1. / ((7. / NRe) ** 0.9 + 0.27 * (self.EA / self.DH)))) ** (-2)

        x = NRe - 2300
        LG_fnc = 1/(1+np.exp((x)/1000))
        f = f1*LG_fnc + f2*(1-LG_fnc)

        return f

    def f0_sun2003_shape_effect(self, Q, DENL, VISL):
        # a channel width, b channel height
        a1 = 2 * np.pi * self.R1 / self.ZI
        b1 = self.YI1
        a2 = 2 * np.pi * self.R2 / self.ZI
        b2 = self.YI2
        ll = (np.min([a1, b1]) / np.max([a1, b1]) + np.min([a2, b2]) / np.max([a2, b2])) / 2.0
        NRe = self.DH * Q * DENL / (2 * np.pi * self.RI * self.YI * VISL * np.sin(self.beta))

        # if NRe < 2300:
        #     return 1 / (2. / 3. + 11. / 24. * ll * (2 - ll))
        # else:
        #     return 1 / (2. / 3. + 11. / 24. * ll * (2 - ll)) ** 0.25

        f1 = 1 / (2. / 3. + 11. / 24. * ll * (2 - ll))
        f2 = 1 / (2. / 3. + 11. / 24. * ll * (2 - ll)) ** 0.25

        x = NRe - 2300
        LG_fnc = 1/(1+np.exp((x)/1000))
        f = f1*LG_fnc + f2*(1-LG_fnc)

        return f

    def f0_sun2003_curvature_effect(self, Q, DENL, VISL):
        # radius of curvature of radial type centrifugal pump used in Sun (2003) dissertation
        RC = 1. / np.sin(self.beta) * (1. / np.tan(self.beta) / self.RI -
                                        (self.B1 - self.B2) / (self.R1 - self.R2)) ** (-1)
        rH = self.DH / 2.
        NRe = self.DH * Q * DENL / (2 * np.pi * self.RI * self.YI * VISL * np.sin(self.beta))

        if RC / rH >= 860:
            NRec = 2300.
        else:
            NRec = 2e4 * (rH / RC) ** 0.32

        # if NRe < NRec:
        #     if RC / rH >= 860:
        #         return 1.
        #     else:
        #         return 0.266 * NRe ** 0.389 * (rH / RC) ** 0.1945
        # else:
        #     if NRe * (rH / RC) ** 2 >= 300:
        #         return (NRe * (rH / RC) ** 2) ** 0.05
        #     elif (NRe * (rH / RC) ** 2 > 0.0034) and (NRe * (rH / RC) ** 2 < 300):
        #         return 0.092 * (NRe * (rH / RC) ** 2) ** 0.25 + 0.962
        #     else:
        #         return 1.

        # if NRe < NRec:
        f10 = 0.266 * NRe ** 0.389 * (rH / RC) ** 0.1945
        f11 = 1.
        
        x0 = RC-860*rH
        LG_fnc0 = 1/(1+np.exp((x0)/1000))

        f1 = f10*(LG_fnc0) + f11*(1-LG_fnc0)
        f1 = f10

        # else:
        xxx = NRe * (rH / RC) ** 2
        x1 = NRe * (rH / RC) ** 2 - 0.0034
        x2 = NRe * (rH / RC) ** 2 - 300
        LG_fnc1 = 1/(1+np.exp((x1)/1000)) 
        LG_fnc2 = 1/(1+np.exp((x2)/1000))

        f21 = 0.092 * (NRe * (rH / RC) ** 2) ** 0.25 + 0.962
        f22 = (NRe * (rH / RC) ** 2) ** 0.05

        
        #     if (NRe * (rH / RC) ** 2 < 0.0034)
        f20 = 1.
        #     elif (NRe * (rH / RC) ** 2 > 0.0034) and (NRe * (rH / RC) ** 2 < 300):
        f2_1 = f20*(LG_fnc1) + f21*(1-LG_fnc1)
        f2_1 = f21
        #     if NRe * (rH / RC) ** 2 >= 300:
        f2 = f2_1*(LG_fnc2) + f22*(1-LG_fnc2)

        # combine
        x = NRe - NRec
        LG_fnc = 1/(1+np.exp((x)/1000))
        f = f1*(LG_fnc) + f2*(1-LG_fnc)
        # print (f1, (1-LG_fnc))
        # f = f2
        return f

    def f0_sun2003_rotation_effect(self, Q, DENL, VISL, OMEGA):
        NRe_omega = OMEGA * self.DH ** 2 * DENL / VISL
        NRe = self.DH * Q * DENL / (2 * np.pi * self.RI * self.YI*VISL * np.sin(self.beta))

        if NRe_omega >= 28:
            NRe_omegac = 1070 * NRe_omega ** 0.23
        else:
            NRe_omegac = 2300

        # if NRe < NRe_omegac:
        #     KLaminar = NRe * NRe_omega
        #     if (KLaminar <= 220) and (NRe_omega / NRe < 0.5):
        #         return 1.
        #     elif (KLaminar > 220) and (NRe_omega / NRe < 0.5) and (KLaminar < 1e7):
        #         fr = 0.0883 * KLaminar ** 0.25 * (1. + 11.2 * KLaminar ** (-0.325))
        #         if fr > 1:
        #             return 1.
        #         else:
        #             return fr
        #     else:
        #         fr = (0.0672 * NRe_omega ** 0.5) / (1. - 2.11 * NRe_omega ** (-0.5))
        #         if fr > 1:
        #             return 1.
        #         else:
        #             return fr
        # else:
        #     KTurbulent = NRe_omega ** 2 / NRe
        #     if KTurbulent < 1.:
        #         return 1.
        #     elif (KTurbulent) >= 1. and (KTurbulent < 15.):
        #         return 0.942 + 0.058 * KTurbulent ** 0.282
        #     else:
        #         return 0.942 * KTurbulent ** 0.05


        # if NRe < NRe_omegac:
        KLaminar = NRe * NRe_omega
        if (KLaminar <= 220) and (NRe_omega / NRe < 0.5):
            # f1 = 1.
            f1 = 0.0883 * KLaminar ** 0.25 * (1. + 11.2 * KLaminar ** (-0.325))
            f1 = fr
        elif (KLaminar > 220) and (NRe_omega / NRe < 0.5) and (KLaminar < 1e7):
            fr = 0.0883 * KLaminar ** 0.25 * (1. + 11.2 * KLaminar ** (-0.325))
            f1 = fr
            # if fr > 1:
            #     f1 = 1.
            # else:
            #     f1 = fr
        else:
            fr = (0.0672 * NRe_omega ** 0.5) / (1. - 2.11 * NRe_omega ** (-0.5))
            f1 = fr
            # if fr > 1:
            #     f1 = 1.
            # else:
            #     f1 = fr
        # else:
        KTurbulent = NRe_omega ** 2 / NRe
        if KTurbulent < 1.:
            f2 = 1.
        elif (KTurbulent) >= 1. and (KTurbulent < 15.):
            f2 = 0.942 + 0.058 * KTurbulent ** 0.282
        else:
            f2 = 0.942 * KTurbulent ** 0.05

        
        x = NRe - NRe_omegac
        LG_fnc = 1/(1+np.exp((x)/1000))
        f = f1*LG_fnc + f2*(1-LG_fnc)
        # f = f2
        return f

    def f_Thin2008(self, Q):
    # Thin et al. (2008) friction loss model
        a2 = 2 * np.pi * self.R2 / self.ZI
        b2 = self.YI2
        rH = 2 * a2 * b2 / (a2 + b2)
        C1M = Q / ((2.0 * np.pi * self.R1 - self.ZI * self.TB) * self.YI1)
        C2M = Q / ((2.0 * np.pi * self.R2 - self.ZI * self.TB) * self.YI2)
        W1 = C1M / np.sin(self.B1)
        W2 = C2M / np.sin(self.B2)
        h_friction = b2 * (2 * self.R2 - 2 * self.R1) * (W1 + W2) ** 2 / (8 * 9.81 * np.sin(self.B2) * rH)
        return h_friction

    def f_Bing2012(self, Q):
    # Bing et al. (2012) friction model
        fff = self.f0_Churchill_1977(Q)
        Fshape = self.f0_sun2003_shape_effect(Q)
        Fcurve = self.f0_sun2003_curvature_effect(Q)
        Frotation = self.f0_sun2003_rotation_effect(Q)
        friction_factor = fff * Fshape * Fcurve * Frotation
        C1M = Q / ((2.0 * np.pi * self.R1 - self.ZI * self.TB) * self.YI1)
        C2M = Q / ((2.0 * np.pi * self.R2 - self.ZI * self.TB) * self.YI2)
        W1 = C1M / np.sin(self.B1)
        W2 = C2M / np.sin(self.B2)
        h_friction = friction_factor * self.LI / self.DH * (W1 ** 2 + W2 ** 2) / (4 * 9.81)
        return h_friction

    def f_sun2003(self, Q, DENL, VISL, OMEGA):
        fff = self.f0_Churchill_1977(Q, DENL, VISL)
        Fshape = self.f0_sun2003_shape_effect(Q, DENL, VISL)
        Fcurve = self.f0_sun2003_curvature_effect(Q, DENL, VISL)
        Frotation = self.f0_sun2003_rotation_effect(Q, DENL, VISL, OMEGA)
        friction_factor = fff * Fshape * Fcurve * Frotation
        bm = (self.YI1 + self.YI2) / 2.
        h_friction = friction_factor * (Q ** 2 /(4 * np.pi ** 2 * bm ** 2 * 
                    (np.sin(self.beta)) ** 3 * self.R1 * self.R2)) * (self.R2 - self.R1) / (2 * 9.81 * self.DH )
        
        return h_friction

    def TUALP_2022(self, Q, DENL, VISL, OMEGA):
    
        h_friction1 = self.f_sun2003(Q, DENL, VISL, OMEGA)

        AI = self.VOI / self.LI
        VI = (Q) / self.ZI / AI
        DI = 4.0 * self.VOI / self.AIW
        C1M = Q / ((2.0 * np.pi * self.R1 - self.ZI * self.TB) * self.YI1)
        C2M = Q / ((2.0 * np.pi * self.R2 - self.ZI * self.TB) * self.YI2)
        W1 = C1M / np.sin(self.B1)
        W2 = C2M / np.sin(self.B2)
        REI = DENL * (W1 + W2) * DI / VISL / 2.0
        EDI = self.EA / DI
        fff = self.friction_unified(REI, EDI)
        h_friction2 = 4.0 * fff * VI ** 2 * self.LI / (2.0 * G * DI)

        x = REI - 2300
        LG_fnc = 1/(1+np.exp((x)/1000))
        h_friction = h_friction1*LG_fnc + h_friction2*(1-LG_fnc)

        return h_friction

    def friction_impeller (self, Q, DENL, VISL, OMEGA):
        if self.F_MODEL == 'sun2003':
            return self.f_sun2003(Q, DENL, VISL, OMEGA)
        elif self.F_MODEL == 'Thin2008':
            return self.f_Thin2008(Q)
        elif self.F_MODEL == 'Bing2012':
            return self.f_Bing2012(Q)
        elif self.F_MODEL == 'tualp_2022':
            return self.TUALP_2022(Q, DENL, VISL, OMEGA)

    @staticmethod
    def friction_unified(REM, ED):
        # friction factor used in unified model
        lo = 1000.
        hi = 3000.
        Fl = 16.0 / REM
        Fh = 0.07716 / (np.log(6.9 / REM + (ED / 3.7)**1.11))**2.0

        if REM < lo:
            return Fl
        elif REM > hi:
            return Fh
        else:
            return (Fh * (REM - lo) + Fl * (hi - REM)) / (hi - lo)

    def friction_leakage (self, re, VLK, OMEGA):
        '''
        OMEGA: rotational speed RPM
        RI: leakage radius (rotational effect, assume equal to RI=R1+R2)
        VLK: axial velocity in the leakage area
        LG: leakage length
        SL: leakage width
        '''
        # friction factor based on Childs 1983 and Zhu et al. 2019 10.4043/29480-MS
        fff = self.LG/self.SL*0.066*re**-0.25*(1+OMEGA**2*self.RI**2/4/VLK**2)**0.375
        return fff
        
    def emulsion (self, VISO=0.1, DENO=900, WC=0, STOW=0.035, RPM=3500, QL=0.001, SN=1):
        """
        The model is based on Brinkman (1952) correlation and Zhang (2017, Fall)
        :param VOI: volume of impeller (m3)
        :param R2:  impeller outer radius (m)
        :param VISO:viscosity of oil (kg/m-s)
        :param VISW:viscosity of water (kg/m-s)
        :param DENO:density of oil (kg/m3)
        :param DENW:density of water (kg/m3)
        :param WC:  water cut (%)
        :param ST:  surface tension (N/m)
        :param N:   rotational speed (rpm)
        :param Q:   flow rate (m3/s)
        :param SN:  stage number (-)
        :param mod: select different model type: tualp, banjar, or zhu
        :return: miu in Pas
        """
        Q = QL
        E = self.emulsion_E  # exponential index
        # E = 4  # exponential index
        # self.VOI = 6e-6
        # self.R2 = 0.04
        VISW = self.VISW
        DENW = self.DENW
        f = RPM / 60.
        miu_tilda = VISO / VISW
        phi_OI = miu_tilda ** (1. / E) / (1. + miu_tilda ** (1. / E))
        phi_WI = 1. - phi_OI
        phi_OE = 1. - (VISW / VISO) ** (1. / E)
        i = 0.

        get_C = lambda SN, WE, RE, ST: SN ** 0.01 * WE ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)

        if self.EMULSION_MODEL == "tualp":
            get_C = lambda SN, WE, RE, ST: (SN * WE * RE) ** 0.15 / (10 * ST ** 0.5)
        elif self.EMULSION_MODEL == "banjar":
            get_C = lambda SN, WE, RE, ST: (SN * WE * RE) ** 0.1 / (10 * ST ** 0.2)
        elif self.EMULSION_MODEL == "tualp_2022":
            # get_C = lambda SN, WE, RE, ST: (SN * WE) ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)
            get_C = lambda SN, WE, RE, ST: SN ** 0.01 * WE ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)
            get_C = lambda SN, WE, RE, ST: SN ** self.emulsion_SN * WE ** self.emulsion_WE * RE ** self.emulsion_RE / (self.emulsion_ST1 * ST ** self.emulsion_ST2)

        # find the inversion point
        St = f * self.VOI / Q
        i = 0.3
        miu_A_OW = 1
        miu_A_WO = 0
        while np.abs(miu_A_OW - miu_A_WO) / miu_A_OW > 0.01:
        # for i in np.arange(1000) / 1000.:
            # if i == 0:
                # continue
            if i <= 0: 
                i=0
                break
            if i >=1:
                i=1
                break

            rouA = i * DENW + (1 - i) * DENO
            We = rouA * Q ** 2 / (STOW * self.VOI)
            miu_M = VISW / (1 - (1 - i) * phi_OE) ** E

            # assume oil in water
            Re = rouA * Q / (VISW * 2 * self.R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISW / (1 - (1 - i)) ** E
            miu_A_OW = C * (miu_E - miu_M) + miu_M

            # assume water in oil
            Re = rouA * Q / (VISO * 2 * self.R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISO / (1 - i) ** E
            miu_A_WO = C * (miu_E - miu_M) + miu_M

            if miu_A_OW<miu_A_WO: i*=0.9
            else: i*=1.05

            # if np.abs(miu_A_OW - miu_A_WO) / miu_A_OW < 0.01:
            #     break

        if WC > i:
            # oil in water
            rouA = WC * DENW + (1 - WC) * DENO
            We = rouA * Q ** 2 / (STOW * self.VOI)
            miu_M = VISW / (1 - (1 - WC) * phi_OE) ** E

            Re = rouA * Q / (VISW * 2 * self.R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISW / (1 - (1 - WC)) ** E
            miu_A = C * (miu_E - miu_M) + miu_M
            miu_A_2 = WC * VISW + (1 - WC) * VISO
            if miu_A_2>miu_A:
                miu_A=miu_A_2
        else:
            # water in oil
            rouA = WC * DENW + (1 - WC) * DENO
            We = rouA * Q ** 2 / (STOW * self.VOI)
            miu_M = VISW / (1 - (1 - WC) * phi_OE) ** E

            Re = rouA * Q / (VISO * 2 * self.R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISO / (1 - WC) ** E
            miu_A = C * (miu_E - miu_M) + miu_M
            miu_A_2 = WC * VISW + (1 - WC) * VISO
            if miu_A_2>miu_A:
                miu_A=miu_A_2
        return miu_A

    def QBEM_EQN (self, QBEM, RPM, VISL, VISW):
        QBEM = QBEM * ( RPM / 3600.0) * (VISL / VISW) ** (self.QBEM_VISL_coef * (3448 / self.NS)**4) # QBEM = QBEM * (N / 3600) * (VISL / VISW) ** (0.01 * (3448 / NS)**4)
        return QBEM
        # return QBEM * ( RPM / 3600.0) * (VISO / VISW) ** (self.QBEM_VISL_coef * (3448 / self.NS)**4)           # QBEM = QBEM * (N / 3600) * (VISL / VISW) ** (0.01 * (3448 / NS)**4)

    def SGMU_EQN (self, VISL, VISW):
        # ori SGMU = 1 - np.sqrt(np.sin(self.B2)) / self.ZI ** (1.5* (3448 / NS) ** 0.4)
        SGMU = 1 - (np.sqrt(np.sin(self.B2)) / self.ZI ** (1.6)) * ((VISL / VISW / self.SGMU_coef) ** self.SGMU_coef_2 )** ( (self.NS / 2975) ** 4 )
        # SGMU = 1
        return SGMU
    
    def SGM_EQN (self, REC, VISW, VISL):
        SGM = (VISW / VISL) **  self.SGM_coef / (10.0 + 0.02 * REC ** self.SGM_coef_2)     #SGMU_coef=0.1 default,  SGM_coef_2=0.25
        # SGM = 0.03
        return SGM

    def TURN_EQN (self, VISL, VISW):
        FTI = self.FTI * ( (VISL / VISW) ** self.FTI_coef )           #zimo HTI = FTI * VI ** 2 / (2.0 * G)   SGM = (VISW / VISL) ** 0.1 / (10.0 + 0.02 * REC ** 0.25)
        FTD = self.FTD * ( (VISL / VISW) ** self.FTD_coef )           #zimo HTD = FTD * VD ** 2 / (2.0 * G)
        # FTI = self.FTI            #zimo HTI = FTI * VI ** 2 / (2.0 * G)   SGM = (VISW / VISL) ** 0.1 / (10.0 + 0.02 * REC ** 0.25)
        # FTD = self.FTD           #zimo HTD = FTD * VD ** 2 / (2.0 * G)
        return FTI, FTD
        
    def CD_EQN (self, VSR, DB, GF, GV, FGL, DENL, VISL, DENG, VISG, RPM, QL, QLK, QG):
        Q = QL
        # GV = GF
        GV = GV # GV = GV or GF, that is a question
        alphaG_crit = self.HG_critical(RPM)

        REB = DENL*VSR*DB/VISL
        
        # if VSR == 0: VSR = 1e-6    # eliminate error
        # SR = DB * (2.0 * np.pi * RPM / 60.0) / VSR
        
        # CD1 = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.55 * SR**2.0)
        
        # '''slug flow'''
        # C1M_L = (QL+QLK) / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI)
        # W1_L = C1M_L / np.sin((self.B1+self.B2/2))
        # C1M_G = QG / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI)
        # W1_G = C1M_G / np.sin((self.B1+self.B2/2))


        # W1L = W1_L / (1 - GV)
        # W1G = W1_G / GV
        # mium1 = (1 - GV) * VISL + GV * VISG
        # DENM1 = (1 - GV) * DENL + GV * DENG
        # CD2 = (12. * mium1 * (self.CD_INT_EFF*9.13e7 * GV ** self.CD_GV_EFF / W1_G ** self.CD_Gas_EFF * W1_L ** self.CD_Liquid_EFF ) / (np.abs(W1G - W1L) * DENM1) * (RPM/3600)**2)*DB

        # x = GV - alphaG_crit
        # LG_fnc = 1/(1+np.exp((x*10)))
        # CD = CD1*LG_fnc + CD2*(1-LG_fnc)

        if VSR == 0: VSR = 1e-6    # eliminate error
        SR = DB * (2.0 * np.pi * RPM / 60.0) / VSR
        
        if FGL == 'BUB':
            '''bubble flow'''
            CD = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.55 * SR**2.0)
        
        else:
            '''slug flow'''
            C1M_L = (QL+QLK) / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI)
            W1_L = C1M_L / np.sin((self.B1+self.B2/2))
            C1M_G = QG / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI)
            W1_G = C1M_G / np.sin((self.B1+self.B2/2))


            W1L = W1_L / (1 - GV)
            W1G = W1_G / GV
            mium1 = (1 - GV) * VISL + GV * VISG
            DENM1 = (1 - GV) * DENL + GV * DENG
            CD = (12. * mium1 * (self.CD_INT_EFF*9.13e7 * GV ** self.CD_GV_EFF / W1_G ** self.CD_Gas_EFF * W1_L ** self.CD_Liquid_EFF ) / (np.abs(W1G - W1L) * DENM1) * (RPM/3600)**2)*DB
    
        return CD

    def DB_EQN (self, HP, GV, GF, STLG, DENL, DENG, QL):
        # GV = GF
        GV = GV # GV = GV or GF, that is a question
        Q = QL
        # use DB or DBMax to calculate bubble flow?
        if HP*Q<0.000001: 
            HP=1
            Q=0.0001    # eliminate convergence and error problem
        DB = self.factor*6.034 * GV**self.DB_GVF_EFF * (STLG / DENL)**0.6 * (HP * Q *(0.05/self.R2)**self.DB_NS_EFF/ (DENL * self.ZI * self.VOI))**(-0.4) * (DENL / DENG)**0.2  #original 2021-10
        if DB > 1 : DB = 1
        DBMAX = DB/0.43
        # DBMAX = DB/0.43
        # DBMAX = DB/0.6
        return DB, DBMAX
  
    def HG_critical (self, RPM):
        return 0.5 - self.alphaG_crit_critical*(np.exp(-(RPM/3500.0)))**self.alphaG_crit_coef   #Fotran code selection

    def GVF_Critical (self, HP, DENL, DENG, VISL, VISG, RPM, STLG, QL, QG, QLK):
        '''dispersed bubble to bubble'''
        # critical bubble size in turbulent flow (Barnea 1982)
        DBCrit = 2.0 * (0.4 * 0.073 / (DENL - DENG) / ((2.0 * pi * RPM / 60.0)**2 * self.RI))**0.5

        if HP*QL<0.000001: 
            # HP*Q = 1
            LambdaC1 = DBCrit / (6.034 / 0.6 * (STLG / DENL)**0.6 * (1 / (DENL * self.ZI * self.VOI)) **  \
                             (-0.4) * (DENL / DENG)**0.2)
        else:
            LambdaC1 = DBCrit / (6.034 / 0.6 * (STLG / DENL)**0.6 * (HP * QL / (DENL * self.ZI * self.VOI)) **  \
                             (-0.4) * (DENL / DENG)**0.2)
        
        '''bubble to intermittent'''

        # rotary speed effect
        # alphaG_crit = pi / 6.0 - (pi / 6.0 - 0.25) * np.exp(-(N / 3600.0)**4)
        GF = QG/(QG+QL)
        GV = GF     # HG gas holdup, gas insitu void fraction
        VSR = 0.1
        LambdaC2 = 0.1
        ABV = 1.0
        icon = 0

        AIW = self.AB + self.ASF + self.ASB
        DI = 4.0 * self.VOI / AIW
        EDI = self.EA / DI
        AI = self.VOI / self.LI

        while ABV > E1:
            alphaG_crit = self.HG_critical(RPM)
            '''bubble shape effect'''    
            if icon > 1000:
                break
            else:
                icon += 1

            DB, DBMAX = self.DB_EQN(HP, GV, GF, STLG, DENL, DENG, QL)
            
            CD = self.CD_EQN(VSR, DB, GF, GV, 'BUB', DENL, VISL, DENG, VISG, RPM, QL, QLK, QG)

            VSR = np.sqrt(4.0 * DB * (DENL - DENG) * self.RI / (3.0 * CD * DENL)) * (2.0 * pi * RPM / 60.0)
            RS = 2*VSR * (2.0 * pi * self.RI - self.ZI * self.TB) * self.YI / QL
            if RS >= 1: RS = 0.9999
            elif RS <=0: RS = 0.0001        # eliminate error and non convergence problem
            GV = (RS - 1.0 + np.sqrt((1.0 - RS)**2 + 4.0 * RS * LambdaC2)) / (2.0 * RS)


            ABV = np.abs((GV - alphaG_crit) / alphaG_crit)

            if GV > alphaG_crit:
                LambdaC2 *=0.9
            else:
                LambdaC2 *=1.1

            if LambdaC2 < 0:
                LambdaC2 = -LambdaC2
                break

            if icon > 1e5:
                break
            else:
                icon += 1

        '''intermittent to segregated'''
        # not considered in this version
        return LambdaC1, LambdaC2

    def HG_CAL(self,HP,FGL, QL, QG, QLK, RPM, DENL, DENG, VISL, VISG, STLG):
        GF = QG / (QL+QLK + QG)
        DB = 0.00001
        OMEGA = 2.0 * pi * RPM / 60.0
        icon=0
        AIW = self.AB + self.ASF + self.ASB
        DI = 4.0 * self.VOI / self.AIW
        EDI = self.EA / DI
        AI = self.VOI / self.LI
        REB = 0.0
        RS = 0.0
        SR = 0.0
        VSRN = 0.0
        ABV = 1.0
        VSR = 0.1

        '''bubble shape effect'''

        if FGL == 'BUB':
            # Based on Sun drag coefficient
            GV  = 0.005
           
        else:
            # Based on Sun drag coefficient
            GV  = 0.5


        while(ABV > E1):
            if icon > 1000:
                GV = GF
                break
            else:
                icon += 1
            
            '''bubble shape effect'''
            # alphaG_crit = alphaG_crit_critical * 0.7**(VSR**alphaG_crit_coef) -0.25*np.exp(-(N/3500.0)**4.0)   #Fotran code selection
            alphaG_crit = self.HG_critical(RPM)
            if GV < alphaG_crit:
                FGL = 'BUB'
            else:
                FGL = 'INT'
            DB, DBMAX = self.DB_EQN(HP, GV, GF, STLG, DENL, DENG, QL)
            CD = self.CD_EQN(VSR, DB, GF, GV, FGL, DENL, VISL, DENG, VISG, RPM, QL, QLK, QG)
                
            VSRN    = np.sqrt(4.0*DB * (DENL - DENG) * self.RI / (3.0 * CD * DENL)) * OMEGA
            ABV     = np.abs((VSRN-VSR)/VSR)
            VSR     = (VSRN + 9.0*VSR)/10.0
            RS      = VSR*(2.0*np.pi*self.RI-self.ZI*self.TB)*self.YI/(QL+QLK)
            
            if(GF < 0.0): 
                GV  = 0.0001
            else:
                GV  = (RS-1.0+np.sqrt((1.0-RS)**2+4.0*RS*GF))/(2.0*RS)
                # GV  = (0.5*GV+0.5*GV1)
        
        return GV, CD, DB, VSR, FGL

    def Single_phase(self, RPM_in=3600, QL_in=5000, DENO_in=900, VISO_in=0.1, WC_in = 0, STOW_in = 0.035):

        ''' original version '''
        """
        Based on previous single-phase model (Jiecheng Zhang) with new development on pressure losses
        :param Q:   flow rate in m3/s
        :param QEM: best match flow rate in m3/s
        :param DENL: liquid density in kg/m3
        :param DENW: water density in kg/m3
        :param N:   rotational speed in rpm
        :param NS:  specific speed based on field units
        :param SGM: tuning factor
        :param SN:  stage number
        :param ST:  surface tension Nm
        :param VISL: liquid viscosity in Pas
        :param VISM: water viscosity in Pas
        :param WC:  water cut in %
        :return: HP, HE, HF, HT, HD, HRE, HLK, QLK in filed units
        """
        # updata flow condition
        
        RPM = RPM_in
        DENW = 997.                 # water density
        DENO = DENO_in
        VISW = 1e-3                 # water viscosity
        VISO = VISO_in
        WC = WC_in/100
        STOW = STOW_in
        QL = QL_in
        OMEGA = 2.0 * pi * RPM_in / 60.0
        QBEM = self.QBEM
        
        # calculation
        QLK = 0.02 * QL
        HP = 10.
        HE, HEE = 0., 0
        HFI, HFD = 0., 0
        HTI, HTD = 0., 0
        HLK = 0.
        icon = 0
        HP_new, QLK_new = 1., 1.
        # OMEGA = 2.0 * pi * RPM / 60.0

        ABH = np.abs((HP - HP_new) / HP_new)
        ABQ = np.abs((QLK - QLK_new) / HP_new)

        if WC==1:
            VISL=VISW
            DENL=DENW
        elif WC > 0.:
            VISL = self.emulsion (VISO, DENO, WC, STOW, RPM, QL, self.SN)
            DENL = (DENW*WC+DENO*(1-WC))
        elif WC == 0.:
            VISL = VISO
            DENL = DENO
        else:
            print ('wrong water cut, please input water cut in %')

        # slip factor compared to Wiesner (1961)
        # SGMU = 1 - np.sqrt(np.sin(self.B2)) / self.ZI ** (1.5 * (3448 / NS) ** 0.4)   # JJZ
        SGMU = 1    # non-slip
        SGMU = self.SGMU_EQN(VISL, VISW)

        # new QBEM due to liquid viscosity
        QBEM = self.QBEM_EQN(QBEM, RPM, VISL, VISW)

        while (ABH > E2) or (ABQ > E2):

            C1M = (QL + QLK) / ((2.0 * pi * self.R1 - self.ZI * self.TB) * self.YI1)
            C2M = (QL + QLK) / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)
            U1 = self.R1 * OMEGA
            U2 = self.R2 * OMEGA
            W1 = C1M / np.sin(self.B1)
            W2 = C2M / np.sin(self.B2)
            C1 = np.sqrt(C1M ** 2 + (U1 - C1M / np.tan(self.B1)) ** 2)
            C2 = np.sqrt(C2M ** 2 + (U2 - C2M / np.tan(self.B2)) ** 2)
            CMB = QBEM / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)
            C2B = np.sqrt(CMB ** 2 + (U2 - CMB / np.tan(self.B2)) ** 2)

            # Euler head
            # HE=(U2**2-U1**2+W1**2-W2**2+C2**2-C1**2)/(2.0*G)
            # HE = (U2 ** 2 - U2 * C2M / np.tan(self.B2)) / G
            HE = (U2 ** 2 * SGMU - U2 * C2M / np.tan(self.B2)) / G

            # head loss due to recirculation
            if (QL+QLK) <= QBEM:
                VSH = U2 * (QBEM - (QL+QLK)) / QBEM
                C2F = C2B * (QL+QLK) / QBEM
                DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI
                REC = DENL * VSH * DC / VISL
                SGM = self.SGM_EQN(REC,VISW,VISL) # SGM: shear factor due to viscosity
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                C2E = C2F + SGM * (C2P - C2F)
                HEE = HE + (C2E ** 2 - C2 ** 2) / (2.0 * G)
            else:
                VSH = U2 * (QL+QLK - QBEM) / QBEM
                C2F = C2B * (QL+QLK) / QBEM
                DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI
                REC = DENL * VSH * DC / VISL
                SGM = self.SGM_EQN(REC,VISW,VISL)   # SGM: shear factor due to viscosity
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                C2E = C2F + SGM * (C2P - C2F) * (QL+QLK - QBEM) / QBEM   # ori
                # C2E = (C2**2+C2F**2-VSH**2)/2/C2F   # 2021-8 by HWZ
                HEE = HE + (C2E ** 2 - C2 ** 2) / (2.0 * G)

            # friction loss
            # impeller
            HFI = self.friction_impeller(QL+QLK, DENL, VISL, OMEGA)   

            # diffuser
            AD = self.VOD / self.LD
            VD = QL / self.ZD / AD
            DD = 4.0 * self.VOD / self.ADW
            RED = DENL * VD * DD / VISL
            EDD = self.EA / DD
            FFD = self.friction_unified(RED, EDD)
            HFD = 4.0 * FFD * VD ** 2 * self.LD / (2.0 * G * DD)

            # turn loss   # original
            # FTI = 3.0
            # FTD = 3.0
            # zimo
            AI = self.VOI / self.LI
            VI = (QL+QLK) / self.ZI / AI
            FTI, FTD = self.TURN_EQN(VISL, VISW)
            HTI = FTI * VI ** 2 / (2.0 * G)                #zimo HTI = FTI * VI ** 2 / (2.0 * G)   SGM = (VISW / VISL) ** 0.1 / (10.0 + 0.02 * REC ** 0.25)
            HTD = FTD * VD ** 2 / (2.0 * G)              #zimo HTD = FTD * VD ** 2 / (2.0 * G)

            # new pump head
            HP_new = HEE - HFI - HFD - HTI - HTD

            # calculate leakage
            UL = self.RLK * OMEGA
            HIO = HEE - HFI - HTI
            HLK = HIO - (U2 ** 2 - UL ** 2) / (8.0 * G)
            if HLK >= 0:
                VL = np.abs(QLK) / (2.0 * pi * self.RLK * self.SL)
                REL = DENL * VL * self.SL / VISL
                EDL = self.EA
                FFL = self.friction_leakage(REL, VL, OMEGA)
                VL = np.sqrt(2.0 * G * HLK / (1.5 + 4.0 * FFL * self.LG / self.SL))
                QLK_new = 2.0 * pi * self.RLK * self.SL * VL
            else:
                VL = np.abs(QLK / (2.0 * pi * self.RLK * self.SL))
                REL = DENL * VL * self.SL / VISL
                EDL = self.EA
                FFL = self.friction_leakage(REL, VL, OMEGA)
                VL = np.sqrt(2.0 * G * np.abs(HLK) / (1.5 + 4.0 * FFL * self.LG / self.SL))
                QLK_new = -2.0 * pi * self.RLK * self.SL * VL

            ABQ = np.abs((QLK_new - QLK) / QLK_new)
            QLK = QLK_new
            HLKloss = self.F_leakage/2/G*(QLK/AI)**2        # by Haiwen Zhu
            HP_new -= HLKloss
            ABH = np.abs((HP_new - HP) / HP_new)
            HP = HP_new

            if icon > 500:
                break
            else:
                icon += 1

        # return pressure in psi, flow rate in bpd
        HP = HP * G * DENL / psi_to_pa
        HE = HE * G * DENL / psi_to_pa
        HEE = HEE * G * DENL / psi_to_pa
        HF = (HFI + HFD) * G * DENL / psi_to_pa
        HT = (HTI + HTD) * G * DENL / psi_to_pa
        HI = (HFI + HTI) * G * DENL / psi_to_pa
        HD = (HFD + HTD) * G * DENL / psi_to_pa
        HRE = HE - HEE
        HLKloss = HLKloss * G * DENL / psi_to_pa
        HLK = HLK * G * DENL / psi_to_pa
        QLK = QLK * 24.0 * 3600.0 / bbl_to_m3

        # return in psi
        # print ( HP, HE, HRE, HF, HT, HLKloss, int(QL/bpd_to_m3s), int(QLK))
        return HP, HE, HF, HT, HD, HRE, HLKloss, QLK

    def Gas_Liquid_phase(self, RPM_in=3600.0, QL_in=5000, QG_in=500, 
        DENO_in=900, DENG_in = 1.225, VISO_in=0.1, VISG_in=0.000018, 
        WC_in = 0, STOW_in = 0.035, STLG_in = 0.073):

        """
        Calcualte gas-liquid performance of ESP
        :param QL:  liquid flow rate in m3/s
        :param QG:  gas flow rate m3/s
        :param QEM: best match flow rate in m3/s
        :param flg: 'Z': Zhu model; 'S': Sun model; 'B': Barrios model
        :param DENG: gas density in kg/m3
        :param DENL: liquid density in kg/m3
        :param DENW: water density in kg/m3
        :param N:   rotational speed in rpm
        :param NS:  specific speed based on field units
        :param SGM: tuning factor
        :param SN:  stage number
        :param ST:  surface tension Nm
        :param VISG: gas viscosity in Pas
        :param VISL: liquid viscosity in Pas
        :param VISM: water viscosity in Pas
        :param WC:  water cut in %
        :return: PP, PE, PF, PT, PD, PRE, PLK, QLK, GV in field units
        """
        # update flow condition
        
        RPM = RPM_in
        OMEGA = 2.0 * pi * RPM_in / 60.0
        QL = QL_in
        QG = QG_in
        DENW = 997.                 # water density
        DENO = DENO_in
        DENG = DENG_in
        VISW = 1e-3                 # water viscosity
        VISO = VISO_in
        VISG = VISG_in
        WC = WC_in/100
        STOW = STOW_in
        STLG = STLG_in
        QBEM = self.QBEM

        if WC==1:
            VISL=VISW
            DENL=DENW
        elif WC > 0.:
            VISL = self.emulsion (VISO, DENO, WC, STOW, RPM, QL, self.SN)
            DENL = (DENW*WC+DENO*(1-WC))
        elif WC == 0.:
            VISL = VISO
            DENL = DENO
        else:
            print ('wrong water cut, please input water cut in %')

        # slip factor compared to Wiesner (1961)
        # SGMU = 1 - np.sqrt(np.sin(self.B2)) / self.ZI ** (1.5 * (3448 / NS) ** 0.4)   # JJZ
        SGMU = 1    # non-slip
        SGMU = self.SGMU_EQN(VISL, VISW)

        # new QBEM due to liquid viscosity
        QBEM = self.QBEM_EQN(QBEM, RPM, VISL, VISW)


        FGL = 'Other'   # flow pattern
        # # a tricky adjusting of rotational speed in Sun's model
        # if (flg == 'S') and N < 3500:
        #     N += 600

        # run single-phase calculation to initialize
        HP, HE, HF, HT, HD, HRE, HLKloss, QLK= self.Single_phase(RPM, QL, DENO, VISO, WC_in, STOW)

        if HP < 0:
            HP = HP
        # convert filed units
        PP = HP * psi_to_pa
        # QG = 
        GF = QG / (QL + QG)

        icon = 0
        ABP = 1.
        PE, PEE = 0., 0
        PFI, PFD = 0., 0
        PTI, PTD = 0, 0
        GV = 0.
        PLK = 0.
        HLKloss = 0.
        QLK = 0.02 * QL

        AIW = self.AB + self.ASF + self.ASB
        ADW = self.AV + self.ADF + self.ADB
        AI = self.VOI / self.LI
        AD = self.VOD / self.LD
        DI = 4.0 * self.VOI / AIW
        DD = 4.0 * self.VOD / ADW
        EDI = self.EA / DI
        EDD = self.EA / DD
        DEND = DENL * (1.0 - GF) + DENG * GF
        VISD = VISL * (1.0 - GF) + VISG * GF
        self.DENG = DENG
        CF = 0
        
        # while ABP > E1 or ABQ > E1:
        while ABP > E1 or ABQ >E1:
            VI = (QL + QG + QLK) / self.ZI / AI     # add QG by Haiwen Zhu
            VD = (QL + QG) / self.ZD / AD       # add QG by Haiwen Zhu
            C1M = (QL + QG + QLK) / ((2.0 * pi * self.R1 - self.ZI * self.TB) * self.YI1)        # add QG by Haiwen Zhu
            C2M = (QL + QG + QLK) / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)        # add QG by Haiwen Zhu
            U1 = OMEGA * self.R1
            U2 = self.R2 * OMEGA
            W1 = C1M / np.sin(self.B1)
            W2 = C2M / np.sin(self.B2)
            C1 = np.sqrt(C1M ** 2 + (U1 - C1M / np.tan(self.B1)) ** 2)
            C2 = np.sqrt(C2M ** 2 + (U2 - C2M / np.tan(self.B2)) ** 2)
            CMB = QBEM / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)
            C2B = np.sqrt(CMB ** 2 + (U2 - CMB / np.tan(self.B2)) ** 2)

            HE = (U2 ** 2 * SGMU - U2 * C2M / np.tan(self.B2)) / G

            if self.GL_MODEL == 'Homogenous':
                GV = GF
            elif self.GL_MODEL == 'Zhu_2017':
                GV, DBMAX = GF, 0.3
                CD = 0.
                ABV = 1.
                VSR = 0.1
                counter = 0
                OMEGA = 2.0 * pi * RPM / 60.0

                while ABV > E1:
                    counter += 1
                    if counter > 10000:
                        return GV, CD/DBMAX

                    DB, DBMAX = self.DB_EQN(HP, GV, GF)
                    REB = DENL * VSR * DB / VISL
                    SR = DB * OMEGA / VSR

                    if REB < 50.0:
                        CD = 24.0 / REB * (1.0 + 0.15 * REB ** 0.687) * (1.0 + 0.3 * SR ** 2.5)
                    else:
                        CD = 24.0 / REB * (1.0 + 0.15 * REB ** 0.687) * (1.0 + 0.55 * SR ** 2.0)
                    VSRN = np.sqrt(4.0 * DB * (DENL - DENG) * self.RI / (3.0 * CD * DENL)) * OMEGA
                    ABV = np.abs((VSRN - VSR) / VSR)
                    VSR = VSRN
                    RS = VSR * (2.0 * pi * self.RI - self.ZI * self.TB) * self.YI / (QL + QLK)

                    if GV < 0.0:
                        GV = 0.0
                    else:
                        GV = (RS - 1.0 + np.sqrt((1.0 - RS) ** 2 + 4.0 * RS * GF)) / (2.0 * RS)
                        
            elif self.GL_MODEL == 'tualp_2022':
                GVC1, GVC2 = self.GVF_Critical(HP,DENL, DENG, VISL, VISG, RPM, STLG, QL, QG, QLK)
                if GF < GVC2:
                    FGL = 'BUB'
                else:
                    FGL = 'INT'
                GV, CD, DB, VSR, FGL= self.HG_CAL(HP, FGL,QL, QG, QLK, RPM, DENL, DENG, VISL, VISG, STLG)

            DENI = DENL * (1.0 - GV) + DENG * GV
            VISI = VISL * (1.0 - GV) + VISG * GV        # revised by Haiwen zhu, should consider gas effect to viscosity
            # DEND = DENL * (1.0 - (0.9*GF+0.1*GV)) + DENG * GF       # correction for GV in diffuser? try
            PE = HE * DENI * G
            VLR = (QL + QLK) / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI * (1.0 - GV))
            VGR = QG / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI * GV)

            if (QL + QLK) <= QBEM:
                VSH = U2 * (QBEM - (QL + QLK)) / QBEM
                C2F = C2B * (QL + QLK) / QBEM
                DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)

                # default, no viscosity effect
                C2E = C2F  # ori

                # # consider viscosity effect
                # REC = DENI * VSH * DC / VISL    # SPSA
                # SGM = self.SGM_EQN(REC)
                # C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F) # SPSA
                # C2E = C2F + SGM * (C2P - C2F)

                PEE = PE + DENI * (C2E ** 2 - C2 ** 2) / 2.0
            else:
                VSH = U2 * (QL + QLK - QBEM) / QBEM
                C2F = C2B * (QL + QLK) / QBEM
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)

                # default, no viscosity effect
                SGM = 0.3
                C2E = C2F + SGM * (C2P - C2F) * (QL + QLK - QBEM) / QBEM

                # # consider viscosity effect
                # DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI # SPSA
                # REC = DENI * VSH * DC / VISI     # SPSA
                # SGM = self.SGM_EQN(REC)
                # C2E = C2F + SGM * (C2P - C2F) * (QL + QLK - QBEM) / QBEM

                PEE = PE + DENI * (C2E ** 2 - C2 ** 2) / 2.0

            REI = DENI * (W1 + W2) * DI / VISI / 2.0 / 2.0

            RED = DEND * VD * DD / VISD / 2.0

            
            # impeller
            HFI = self.friction_impeller(QL+QLK, DENI, VISI, OMEGA)   
            PFI = HFI * DENI * G 
            # diffuser
            AD = self.VOD / self.LD
            VD = QL / self.ZD / AD
            DD = 4.0 * self.VOD / self.ADW
            RED = DENL * VD * DD / VISL
            EDD = self.EA / DD
            FFD = self.friction_unified(RED, EDD)
            PFD = 2.5 * 4.0 * FFD * DEND * VD ** 2 * self.LD / (2.0 * DD)



            # FFI = self.friction_unified(REI, EDI)
            # FFD = self.friction_unified(RED, EDD)
            # PFI = 2.5 * 4.0 * FFI * DENI * VI ** 2 * self.LI / (2.0 * DI)
            # PFD = 2.5 * 4.0 * FFD * DEND * VD ** 2 * self.LD / (2.0 * DD)

            # ori
            # FTI, FTD = 3
            # zimo
            FTI, FTD = self.TURN_EQN(VISL, VISW)
            PTI = FTI * DENI * VI ** 2 / (2.0 )                #zimo HTI = FTI * VI ** 2 / (2.0 * G)   SGM = (VISW / VISL) ** 0.1 / (10.0 + 0.02 * REC ** 0.25)
            PTD = FTD * DEND * VD ** 2 / (2.0 )              #zimo HTD = FTD * VD ** 2 / (2.0 * G)


            PPN = PEE - PFI - PFD - PTI - PTD

            UL = self.RLK * OMEGA
            PIO = PEE - PFI - PTI
            PLK = PIO - DENI * (U2 ** 2 - UL ** 2) / 8.0

            if PLK >= 0.:
                VL = np.abs(QLK) / (2.0 * pi * self.RLK * self.SL)
                REL = np.abs(DEND * VL * self.SL / VISD)    # changed VISL to VISD by Haiwen Zhu
                EDL = 0.0
                FFL = self.friction_leakage(REL, VL, OMEGA)      # by Haiwen Zhu
                VL = np.sqrt(2.0 * PLK / (1.5 + 4.0 * FFL * self.LG / self.SL) / DEND)
                QLKN = 2.0 * pi * self.RLK * self.SL * VL
            else:
                VL = np.abs(QLK / (2.0 * pi * self.RLK * self.SL))
                REL = DENL * VL * self.SL / VISL
                EDL = 0.0
                FFL = self.friction_leakage(REL, VL, OMEGA)      # by Haiwen Zhu
                VL = np.sqrt(2.0 * np.abs(PLK) / (1.5 + 4.0 * FFL * self.LG / self.SL) / DEND)
                QLKN = -2.0 * pi * self.RLK * self.SL * VL

            HLKloss = self.F_leakage* DENI*(QLK/AI)**2        # by Haiwen Zhu
            PPN = PEE - PFI - PFD - PTI - PTD - HLKloss                              # by Haiwen Zhu
            ABQ = np.abs((QLKN - QLK) / QLK)
            QLK = QLKN
            ABP = np.abs((PPN - PP) / PPN)

            if icon > 200:
                break
                # PP = 0.9*PPN+0.1*PP
                # icon += 1
            else:
                PP = 0.5*PPN+0.5*PP
                icon += 1




        # return pressure in psi, flow rate in bpd
        PP = PP / psi_to_pa
        PE = PE / psi_to_pa
        PEE = PEE / psi_to_pa
        PF = (PFI + PFD) / psi_to_pa
        PT = (PTI + PTD) / psi_to_pa
        PD = (PFD + PTD) / psi_to_pa
        PRE = np.abs(PE - PEE)
        HLKloss = HLKloss / psi_to_pa
        QLK = QLK * 24.0 * 3600.0 / bbl_to_m3
        if FGL == 'BUB':
            FGL == 'BUB'
        # print('QL:', round(QL* 24.0 * 3600.0 / bbl_to_m3, 6), 'PP', round(PP,6), 'PRE', round(PRE,6), 'PF', round(PF,6), 'PT', round(PT,6),
        #              'PD', round(PD,6), 'HLKloss', round(HLKloss,6), 'GF', round(GF,6), 'GV', round(GV,6), 'FGL', FGL, 'icon', icon, 'CF', CF)

        # if QL/bbl_to_m3*24*3600 >8000:
        #     DB = factor*6.034 * GF**DB_GVF_EFF * (ST / self.DENL)**0.6 * (HP *psi_to_pa * QL * (N / 3500 * 0.05/self.R2)**DB_NS_EFF / (self.DENL * self.ZI * self.VOI))**(-0.4) * (self.DENL / self.DENG)**0.2


        # print (HP, HE, HF, HT, HD, HRE, HLKloss, QLK)
        # print (PP, PE, PF, PT, DB, PRE, HLKloss, QLK, GV, FGL)
        return PP, PE, PF, PT, DB, PRE, HLKloss, QLK, GV, FGL

class ESP_validation(object):
    def __init__(self, ESP_GEO, QBEM, Exp_data, pump_name='default', bx=None):
        self.ESP = ESP_GEO
        self.QBEM = QBEM
        self.Exp_data = Exp_data
        self.bx = bx
        self.pump_name = pump_name

    @staticmethod   # no self as input
    def gasdensity(p, t, gamma_std, h):
        """
        gas density based on CIPM-81 (Davis, 1972) correlations
        :param p: pressure in psig
        :param t: temperature in Fahrenheit
        :param gamma_std: gas specific gravity (compared to air)
        :param h: humidity in %
        :return: gas density in kg/m3
        """
        A = 1.2811805e-5
        B = -1.9509874e-2
        C = 34.04926034
        D = -6.3536311e3
        alpha = 1.00062
        beta = 3.14e-8
        gamma = 5.6e-7
        a0 = 1.62419e-6
        a1 = -2.8969e-8
        a2 = 1.0880e-10
        b0 = 5.757e-6
        b1 = -2.589e-8
        c0 = 1.9297e-4
        c1 = -2.285e-6
        d = 1.73e-11
        e = -1.034e-8
        R = 8.31441
        Ma = 28.9635  # air molecular weight, g/mol
        Mg = gamma_std*Ma
        Mv = 18  # water molecular weight, g/mol

        Pabs = (p + 14.7) * 6894.76
        Tt = (t - 32) / 1.8
        Tabs = Tt + 273.15
        psv = 1.0 * np.exp(A * (Tabs) ** 2.0 + B * Tabs + C + D / Tabs)
        f = alpha + beta * Pabs + gamma * (Tt) ** 2
        xv = h / 100.0 * f * psv / Pabs
        Z = 1.0 - Pabs / Tabs * (a0 + a1 * Tt + a2 * (Tt) ** 2.0 + (b0 + b1 * Tt) * xv + (c0 + c1 * Tt) * (xv) ** 2) + \
            (Pabs / Tabs) ** 2.0 * (d + e * (xv) ** 2.0)
        return Pabs * Mg / 1000.0 / (Z * R * Tabs) * (1.0 - xv * (1.0 - Mv / Mg))

    def ESP_curve(self, QL=np.arange(0.001, 1.1, 0.002) *5000, QG=np.arange(0.01, 1.1, 0.02) * 0, 
                VISO_in = 0.5, DENO_in = 950, DENG_std = 1.225, WC=80, RPM = 3600,
                VISG_std = 0.000018, O_W_ST = 0.035, G_L_ST = 0.073, P = 350, T=288, 
                curve_type = 'SGL', remove_nective_hp = True):
        '''
        QL, QG in bpd
        WC in %
        '''
        
        QL = QL * bpd_to_m3s
        QG = QG *bpd_to_m3s
        QBEM = self.QBEM * bpd_to_m3s
        gamma_std = DENG_std/1.225
        
        DENG = self.gasdensity(P, T, gamma_std, 0)  # 1 is specific gravity of air
        
        if curve_type == 'GL' :
            gl_cal = np.vectorize(ESP_TUALP(ESP_GEO=self.ESP,QBEM=QBEM).Gas_Liquid_phase)

            PP, PE, PF, PT, DB, PRE, HLKloss, QLK, GV, FGL = gl_cal (RPM_in=RPM, 
                            QL_in=QL, QG_in=QG, DENO_in=DENO_in, DENG_in=DENG, VISO_in=VISO_in, VISG_in=VISG_std,
                            WC_in=WC, STOW_in=O_W_ST, STLG_in=G_L_ST)
             
            return QL/bpd_to_m3s, PP   #psi
            
        elif curve_type == 'SGL':
            sgl_cal = np.vectorize(ESP_TUALP(ESP_GEO=self.ESP,QBEM=QBEM).Single_phase)
            HP, HE, HF, HT, HD, HRE, HLKloss, QLK= sgl_cal(RPM, QL, DENO_in, VISO_in, WC, O_W_ST)

            return QL/bpd_to_m3s, HP   #psi

        else:
            print('wrong curve type, neither single phase flow or gas liquid flow')

    def Oil_validation(self):
        VISL = self.Exp_data['TargetVISL_cp'].unique()
        DENL = self.Exp_data['DENL_kgm3'].mean()
        if self.bx != None:
            bx = self.bx
        else:
            fig2, (bx) = plt.subplots(dpi = dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
        for visl in VISL:
            df_plot = self.Exp_data[self.Exp_data['TargetVISL_cp'] == visl].copy()
            QL = df_plot.QL_bpd.to_numpy()
            QL_pri, HP_pri = self.ESP_curve(QL=QL, QG=0,VISO_in=visl/1000, DENO_in=DENL, DENG_std=1.225, WC=0, RPM=self.Exp_data['RPM'].mean(),
                        VISG_std=0.000018,O_W_ST=0.035,G_L_ST=0.073,P=100,T=40, curve_type='SGL')

            bx.scatter(QL, df_plot['DP_psi'],linewidths=0.75, s=8)
            bx.plot(QL_pri, HP_pri, label=str(int(visl))+' cP', linewidth=0.75)
            bx.set_ylim(0)
            bx.legend(frameon=False)

        bx.set_xlabel('QL bpd', fontsize=8)
        bx.set_ylabel('Head psi', fontsize=8)
        bx.legend(frameon=False, fontsize=5)
        if self.pump_name == 'Flex31':
            title='Oil performance MTESP at '+str(int(self.Exp_data['RPM'].mean()))+' RPM'
        else:
            title='Oil performance '+self.pump_name+' at '+str(int(self.Exp_data['RPM'].mean()))+ ' RPM'
        bx.set_title(title, fontsize=8)
        bx.xaxis.set_tick_params(labelsize=8)
        bx.yaxis.set_tick_params(labelsize=8)

        if self.bx != None:
            pass
        else:
            fig2.savefig('test/'+str(title)+'.jpg')
        
        return bx

    def water_validation(self):
        df_data = self.Exp_data
        if self.bx != None:
            bx = self.bx
        else:
            fig2, (bx) = plt.subplots(dpi = dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
        
        maxRPM = self.Exp_data['RPM'].mean()
        for RPM in [1800, 2400, maxRPM]:
            QL = df_data.QL_bpd
            QL_pri, HP_pri = self.ESP_curve(QL=QL, QG=0,VISO_in=0.001, DENO_in=1000, DENG_std=1.225, WC=100, RPM=RPM,
                VISG_std=0.000018,O_W_ST=0.035,G_L_ST=0.073,P=100,T=40, curve_type='SGL')
        
            bx.scatter(QL*RPM/maxRPM, df_data.DP_psi*(RPM/maxRPM)**2, label=('Catalog %d RPM' %RPM),linewidths=0.75, s=8)
            bx.plot(QL_pri, HP_pri, label=('Model %d RPM' %RPM), linewidth=0.75)

        bx.set_ylim(0)
        bx.set_xlabel('QL bpd', fontsize=8)
        bx.set_ylabel('Head psi', fontsize=8)
        bx.legend(frameon=False, fontsize=5)
        if self.pump_name == 'Flex31':
            title='Water performance MTESP'
        else:
            title='Water performance '+self.pump_name
        bx.set_title(title, fontsize=8)
        bx.xaxis.set_tick_params(labelsize=8)
        bx.yaxis.set_tick_params(labelsize=8)

        if self.bx != None:
            pass
        else:
            fig2.savefig('test/'+str(title)+'.jpg')
        
        return bx

    def GL_validation(self):
        if self.bx != None:
            bx = self.bx
        else:
            fig2, (bx) = plt.subplots(dpi = dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
        
        QG = self.Exp_data['TargetQG_bpd'].unique()
        
        for qg in QG:
            print(qg)
            df_plot = self.Exp_data[self.Exp_data['TargetQG_bpd'] == qg].copy()
            df_plot.sort_values(by=['QL_bpd'])
            QL = np.arange(0.05, 1.1, 0.02) * df_plot.QL_bpd.max()
            QL_pri, HP_pri = self.ESP_curve(QL=QL, QG=df_plot.QG_bpd.mean(),
                    VISO_in=self.Exp_data['TargetVISL_cp'].mean(), DENO_in=self.Exp_data['DENL_kgm3'].mean(), DENG_std=1.225, 
                    WC=self.Exp_data['TargetWC_%'].mean(), RPM=self.Exp_data['RPM'].mean(),
                    VISG_std=0.000018,O_W_ST=0.035,G_L_ST=0.073,P=self.Exp_data['Ptank_psi'].mean(),T=40, curve_type='GL')


            bx.scatter(df_plot.QL_bpd, df_plot.dp12_psi,linewidths=0.75, s=8) # mistake in SQLite, it is ft rather than psi
            bx.plot(QL_pri, HP_pri, label='QG: '+str(int(df_plot.QG_bpd.mean()))+' bpd', linewidth=0.75)
            bx.set_ylim(0)
            bx.legend(frameon=False)

        bx.set_xlabel('Qw bpd', fontsize=8)
        bx.set_ylabel('Head psi', fontsize=8)
        bx.legend(frameon=False, fontsize=5)
        if self.pump_name == 'Flex31':
            title='Mapping performance MTESP at '+str(self.Exp_data['TargetRPM'].mean())+' RPM'
        else:
            title='Mapping performance '+self.pump_name+' at '+str(self.Exp_data['TargetRPM'].mean())+ ' RPM'
        bx.set_title(title, fontsize=8)
        bx.xaxis.set_tick_params(labelsize=8)
        bx.yaxis.set_tick_params(labelsize=8)

        if self.bx != None:
            pass
        else:
            fig2.tight_layout()
            fig2.savefig('test/'+str(title)+'.jpg')
        
        return bx

    def error_analysis(self):

        if self.bx != None:
            bx = self.bx
        else:
            fig2, (bx) = plt.subplots(dpi = dpi, figsize = (3.33,2.5), nrows=1, ncols=1)

        try:
            if self.Exp_data['QG_bpd'].mean() < 5:
                curve_type = 'SGL'
            else:
                curve_type = 'GL'
        except:
            curve_type = 'SGL'

        _, HP = self.ESP_curve(QL=self.Exp_data['QL_bpd'], QG=self.Exp_data['QG_bpd'],
                        VISO_in=self.Exp_data['TargetVISL_cp']/1000, DENO_in=self.Exp_data['DENL_kgm3'], DENG_std=1.225, WC=self.Exp_data['TargetWC_%'], RPM=self.Exp_data['RPM'],
                        VISG_std=0.000018,O_W_ST=0.035,G_L_ST=0.073,P=self.Exp_data['Ptank_psi'],T=self.Exp_data['Tin_F'], 
                        curve_type=curve_type, remove_nective_hp = False)


        e1, e2, e3, e4, e5, e6 = self.stats_analysis(HP, self.Exp_data.DP_psi)
        print(e1, e2, e3, e4, e5, e6)
        
        bx.scatter(self.Exp_data.DP_psi, HP, marker='x', linewidths=0.75, s=3)
        x_max0 = max(HP.max(), self.Exp_data.DP_psi.max())
        dx_max0 = int(x_max0/4) if int(x_max0/4) > 0 else round(x_max0/4,2)
        x_max = np.arange(-x_max0, x_max0, dx_max0)

        bx.plot(x_max, x_max, color='black',linestyle='-', label='perfect match', linewidth=0.75)
        bx.plot(x_max, x_max*0.8, 'r--', label='-20%', linewidth=0.75)
        bx.plot(x_max, x_max*1.2, 'r-.', label='+20%', color='green',  linewidth=0.75)
        bx.set_xlim(0,x_max.max())
        bx.set_ylim(0,x_max.max())
        bx.set_xlabel(r'$P_{exp}$ (psi)', fontsize=8)
        bx.set_ylabel(r'$P_{sim}$ (psi)', fontsize=8)
        bx.xaxis.set_tick_params(labelsize=8)
        bx.yaxis.set_tick_params(labelsize=8)
        bx.set_title(r'Error analysis: e1: %.d, e2: %.d, e3: %.d' % (e1, e2, e3), fontsize=8)
        bx.legend(frameon=False, fontsize=5)

        return e1, e2, e3, e4, e5, e6 
    
    @staticmethod
    def stats_analysis(df_pre, df_exp):
        df_relative = (df_pre - df_exp) / df_exp * 100
        df_actual = df_pre - df_exp

        epsilon1 = df_relative.mean()
        epsilon2 = df_relative.abs().mean()
        epsilon3 = df_relative.std()
        epsilon4 = df_actual.mean()
        epsilon5 = df_actual.abs().mean()
        epsilon6 = df_actual.std()
        return epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6
 
class Turzo_2000(object):
    def __init__(self, QBEP, df_water, df_oil, ax=None):
        self.QBEP = QBEP
        self.H_WATER = df_water['DP_psi']
        self.Q_WATER = df_water['QL_bpd']
        self.RPM_WATER = df_water['RPM'].mean()

        self.df_oil = df_oil
        self.ax = ax

    def turzo_2000(self, VISL, DENL, RPM_vis):
        """
        :param pump: pump name in string
        :param vis: viscosity in cP
        :param den: density in kg/m3
        :param rpm: rotational speed in rpm
        :param Q: flow rate, input as BPD, calculation use 100 gpm, output BPD
        :param H: pump head, input as psi, calculation use ft, output psi
        :return: boosting pressure at four different flow rates, 0.6, 0.8, 1.0. 1.2 Qbep
        """
        # QBEP = {'TE2700': 2700, 'DN1750': 1750, 'GC6100': 6100, 'P100': 9000}

        bpd_to_100gpm = 0.02917

        QBEP = self.QBEP * bpd_to_100gpm * RPM_vis/self.RPM_WATER    # bpd to m3s to 100 gpm
        VISL = VISL / (DENL / 1000)                    # to cSt
        Q_WATER = self.Q_WATER * bpd_to_100gpm * RPM_vis/self.RPM_WATER
        H_WATER = self.H_WATER * psi_to_ft * (RPM_vis/self.RPM_WATER)**2
        
        polyfit = np.polynomial.Polynomial.fit(Q_WATER, H_WATER, 5)
        DPbep06 = polyfit(QBEP*0.6)
        DPbep08 = polyfit(QBEP*0.8)
        DPbep10 = polyfit(QBEP*1.0)
        DPbep12 = polyfit(QBEP*1.2)

        y = -7.5946 + 6.6504 * np.log(DPbep10) + 12.8429 * np.log(QBEP)
        Qstar = np.exp((39.5276 + 26.5605 * np.log(VISL) - y)/51.6565)

        CQ = 1.0 - 4.0327e-3 * Qstar - 1.724e-4 * Qstar**2

        CH06 = 1.0 - 3.68e-3 * Qstar - 4.36e-5 * Qstar**2
        CH08 = 1.0 - 4.4723e-3 * Qstar - 4.18e-5 * Qstar**2
        CH10 = 1.0 - 7.00763e-3 * Qstar - 1.41e-5 * Qstar**2
        CH12 = 1.0 - 9.01e-3 * Qstar + 1.31e-5 * Qstar**2

        Qvis = CQ * np.array([0.6 * QBEP, 0.8 * QBEP, QBEP, 1.2 * QBEP])        # to bpd
        DPvis = np.array([CH06, CH08, CH10, CH12]) * np.array([DPbep06, DPbep08, DPbep10, DPbep12])

        df = pd.DataFrame({'Qvis': Qvis, 'DPvis': DPvis})

        return Qvis/bpd_to_100gpm, DPvis/psi_to_ft

    def validation(self):

        if self.ax != None:
            ax = self.ax
        else:
            fig, (ax) = plt.subplots(dpi = dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
        VISL = self.df_oil['TargetVISL_cp'].unique()

        for visl in VISL:
            df_visl = self.df_oil[self.df_oil['TargetVISL_cp'] == visl].copy()
            H_oil = df_visl['DP_psi']
            Q_oil = df_visl['QL_bpd']
            RPM_vis = df_visl['RPM'].mean()
            DENL = df_visl['DENL_kgm3'].mean()
            Q_pri, H_pri = self.turzo_2000(visl, DENL, RPM_vis)
            ax.scatter(Q_oil, H_oil, s=2, label = ('Test %d cp' %int(visl)))
            ax.plot(Q_pri, H_pri, label = ('Turzo %d cp' %int(visl)))


        ax.set_xlabel('QL bpd', fontsize=8)
        ax.set_ylabel('Head psi', fontsize=8)
        ax.legend(frameon=False, fontsize=5)
        # if self.pump_name == 'Flex31':
        #     title='Oil performance MTESP at '+str(int(self.Exp_data['RPM'].mean()))+' RPM'
        # else:
        #     title='Oil performance '+self.pump_name+' at '+str(int(self.Exp_data['RPM'].mean()))+ ' RPM'
        title = 'Turzo validation with test data'
        ax.set_title(title, fontsize=8)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)

        return ax

if __name__ == "__main__":

    dpi = 300

    # test()

    '''GC6100 gas liquid'''
    # pump_name = 'GC6100'
    # ESP = ESP_default[pump_name]
    # QBEM = QBEM_default[pump_name]

    # TargetRPM = 1500        # 1500, 1800, 2400, 3000
    # TargetP_psi = 150

    # conn, c = connect_db('ESP.db')
    # Exp_data = pd.read_sql_query("SELECT * FROM All_pump "
    #                           + "ORDER BY TargetQG_bpd, QL_bpd"
    #                           + ";", conn)
          

    # # Exp_data = Exp_data[(Exp_data.Pump == pump_name) & (Exp_data.Test == 'Mapping') & 
    # #             (Exp_data.TargetRPM == TargetRPM) & (Exp_data.TargetP_psi == TargetP_psi)]
    # Exp_data = Exp_data[(Exp_data.Pump == pump_name) & (Exp_data.Test == 'Mapping') & 
    #             (Exp_data.TargetRPM == TargetRPM) & (Exp_data.TargetP_psi == TargetP_psi)
    #             & (Exp_data.TargetQG_bpd < 5)]
                
    # Exp_data = Exp_data[(Exp_data.DP_psi > 0)]
    # Exp_data=Exp_data.reset_index(drop=True)
    # disconnect_db(conn)

    # Input = Exp_data
    # Target = Exp_data.DP_psi

    # error = []
    
    # for Pump in Exp_data['Pump'].unique():
    #     df_1 = Exp_data[Exp_data['Pump']==Pump]
    #     for RPM in df_1['RPM'].unique():
    #         df_2 = df_1[df_1['RPM']==RPM]
    #         fig1, (ax, ax2) = plt.subplots(dpi = dpi, figsize = (8,3), nrows=1, ncols=2)
    #         fig2, (bx, bx2) = plt.subplots(dpi = dpi, figsize = (8,3), nrows=1, ncols=2)
    #         # ori pump
    #         ESP_validation(ESP, QBEM, df_2, pump_name, ax).GL_validation()
    #         # e1_ori, e2_ori, e3_ori, e4_ori, e5_ori, e6_ori = ESP_validation(ESP, QBEM, df_2, pump_name, bx).error_analysis()
    #         # error.append([Pump, RPM, 'ori', e1_ori, e2_ori, e3_ori, e4_ori, e5_ori, e6_ori ])
    #         fig1.tight_layout()
    #         fig2.tight_layout()

    '''water validation'''
    # pump_name = 'P100'
    # ESP = ESP_default[pump_name]
    # QBEM = QBEM_default[pump_name]
    # QBEM = 12000
    # conn, c = connect_db('ESP.db')
    # df_data = pd.read_sql_query("SELECT * FROM Catalog_All;", conn)
    # df_data = df_data[df_data.Pump == pump_name]
    # df_data = df_data[df_data.QL_bpd != 0]
    # df_data=df_data.reset_index(drop=True)
    # disconnect_db(conn)

    # fig1, (ax) = plt.subplots(dpi = dpi, figsize = (4,3), nrows=1, ncols=1)
    # ESP_validation(ESP, QBEM, df_data, pump_name, ax).water_validation()
        
    ''' viscosity '''
    # pump_name = 'DN1750'
    # ESP = ESP_default[pump_name]
    # QBEM = QBEM_default[pump_name]
    # QBEM = 4000

    # # ESP_Cal = ESP_validation(ESP_GEO=ESP, QBEM=QBEM, Exp_data=1)
    # # QL = np.arange(0.01, 1.1, 0.02) * 25000
    # # ql, hp = ESP_Cal.ESP_curve(QL=QL, VISO_in=0.001,WC=100, RPM=2400)

    # conn, c = connect_db('ESP.db')
    # Exp_data = pd.read_sql_query("SELECT * "
    #                             + "FROM df_Viscosity "
    #                         #   + "WHERE [TargetWC_%] = 0 "
    #                             + "ORDER BY Pump, RPM, TargetVISL_cp, QL_bpd"
    #                             + ";", conn)
    # disconnect_db(conn)
    
    # Exp_data.drop(Exp_data[Exp_data['Case']=='DN1750_Solano'].index, inplace=True)
    # Exp_data.drop(Exp_data[Exp_data['Case']=='DN1750_Banjar'].index, inplace=True)
    # Exp_data.drop(Exp_data[Exp_data['Case']=='DN1750_Solano_Ave'].index, inplace=True)
    # # Exp_data.drop(Exp_data[Exp_data['Case']=='P100'].index, inplace=True)
    # # Exp_data.drop(Exp_data[Exp_data['Case']=='DN1750_CFD'].index, inplace=True)
    # # Exp_data.reset_index(drop=True, inplace=True)
    # # Exp_data = Exp_data[(Exp_data.Pump.str.startswith(pump_name)) & (Exp_data.RPM == TargetRPM)]
    # # Exp_data = Exp_data[Exp_data['Case']==pump_name]
    # Exp_data = Exp_data[(Exp_data.Pump.str.startswith(pump_name))]
    # Exp_data = Exp_data[Exp_data.QL_bpd != 0]
    # Exp_data = Exp_data[Exp_data.TargetVISL_cp <  1100]
    # Exp_data = Exp_data.reset_index(drop=True)

    # # Test_data = Exp_data[(Exp_data.Pump.str.startswith(pump_name)) & (Exp_data.RPM == TargetRPM)]
    
    # # Exp_data = Exp_data[(Exp_data.Pump.str.startswith(pump_name)) & (Exp_data.RPM == TargetRPM) & ((Exp_data.TargetVISL_cp == 1000) | (Exp_data.TargetVISL_cp == 50))]
    # Exp_data = Exp_data.reset_index(drop=True)

    # Input = Exp_data
    # Target = Exp_data.DP_psi
    # print('Train data: ', Input.shape[0])

    # error = []
    # for Pump in Exp_data['Pump'].unique():
    #     df_1 = Exp_data[Exp_data['Pump']==Pump]
    #     for RPM in df_1['RPM'].unique():
    #         df_2 = df_1[df_1['RPM']==RPM]
    #         fig1, (ax, ax2) = plt.subplots(dpi = dpi, figsize = (8,3), nrows=1, ncols=2)
    #         fig2, (bx, bx2) = plt.subplots(dpi = dpi, figsize = (8,3), nrows=1, ncols=2)
    #         # ori pump
    #         ESP_validation(ESP, QBEM, df_2, pump_name, ax).Oil_validation()
    #         e1_ori, e2_ori, e3_ori, e4_ori, e5_ori, e6_ori = ESP_validation(ESP, QBEM, df_2, pump_name, bx).error_analysis()
    #         error.append([Pump, RPM, 'oil', e1_ori, e2_ori, e3_ori, e4_ori, e5_ori, e6_ori ])
    #         fig1.tight_layout()
    #         fig2.tight_layout()

    # error = pd.DataFrame(error)
    # print(error)
    

    '''turzo validation'''
    
    pump_name = 'DN1750'
    case_name = 'DN1750_CFD'
    QBEP = 1750
    conn, c = connect_db('C:/Users/haz328/Desktop/Research/003 ESP model/SPSA/ESP.db')
    df_water = pd.read_sql_query("SELECT * FROM Catalog_All;", conn)
    df_water = df_water[df_water.Pump == pump_name]
    df_water = df_water[df_water.QL_bpd != 0]
    df_water=df_water.reset_index(drop=True)
    disconnect_db(conn)


    conn, c = connect_db('C:/Users/haz328/Desktop/Research/003 ESP model/SPSA/ESP.db')
    df_oil = pd.read_sql_query("SELECT * "
                                + "FROM df_Viscosity "
                                + "ORDER BY Pump, RPM, TargetVISL_cp, QL_bpd"
                                + ";", conn)
    disconnect_db(conn)
    df_oil = df_oil[(df_oil['Case']==case_name) & (df_oil.RPM == 3500) & (df_oil.TargetVISL_cp < 900)]
    df_oil = df_oil.reset_index(drop=True)


    fig1, (ax) = plt.subplots(dpi = dpi, figsize = (4,3), nrows=1, ncols=1)

    ESP_turzo = Turzo_2000(QBEP, df_water, df_oil, ax)
    ESP_turzo.validation()
    fig1.tight_layout()
    fig1.savefig('turzo')
    plt.show()