# -*- coding: utf-8 -*-
"""
Reference:  Unified Model for Gas-Liquid Pipe Flow Via Slug Dynamics
            #TUALP 3P 2012v1 * May 24, 2012
Developer: Jianjun Zhu
Date: Jan 12, 2019
"""

import numpy as np
import pandas as pd

# global variables
FEC = 0.90      # maximum entrainment fraction in gas core
HLSC = 0.24     # maximum liquid holdup in slug body
G = 9.81        # gravitational acceleration (m/s2)
PI = np.pi      # ratio of the circumference of a circle to its diameter
STAW = 0.0731   # water surface tension against air (N/m)
DENA = 1.2      # density of air at atmospheric pressure (kg/m3)
E1 = 0.00001     # tolerance for iterations
FIC = 0.0142    # critical interfacial friction factor
AXP = 0.        # cross sectional area of pipe (m2)
R = 8.314                   # J/mol/K
M_air = 28.97e-3            # air molecular weight in kg/mol
UiMPa = 10e6                # MPa to Pa

"""
This global is set anywhere when the input data has triggered a possible error. A non-zero value triggers a write of the
input data to the dump file for later analysis of the error
"""
Ierr = 0
IFFM = 0        # interfacial friction indicator

# get_fff - get fanning friction factor
def get_fff(re, ed):
    """
    :param re:  Reynolds number
    :param ed:  relative wall roughness
    :return:    Colebrook frictin factor correlation
    """
    lo = 800.
    hi = 1200.
    Fl = 0.
    Fh = 0.

    if re < hi: Fl = 16.0 / re
    if re > lo or re == lo:
        AA = (2.457 * np.log(1.0 / ((7.0 / re) ** 0.9 + 0.27 * ed))) ** 16.0
        BB = (37530.0 / re) ** 16.0
        Fh = 2.0 * ((8.0 / re) ** 12.0 + 1.0 / (AA + BB) ** 1.5) ** (1.0 / 12.0)

    if re < lo:
        return Fl
    elif re > hi:
        return Fh
    else:
        return (Fh * (re - lo) + Fh * (hi - re)) / (hi - lo)

def get_fff_churchill(re, ed, d=None):
    # friction factor based on Churchill correlation
    if d is None:
        REM = re
        AA = (2.457 * log(1.0 / ((7.0 / REM) ** 0.9 + 0.27 * ed))) ** 16.0
        BB = (37530.0 / REM) ** 16.0
        fff = 2.0 * ((8.0 / REM) ** 12.0 + 1.0 / (AA + BB) ** 1.5) ** (1.0 / 12.0) / 4
        return fff
    else:
        f = (-4 * log10(0.27 * ed / d + (7 / re) ** 0.9)) ** 2
        return 2 / f

def wet_fra_biberg(hx):
    # wet_frc_biberg - Calculate wetted wall fraction assuming flat interface based on Biberg (1999) approximate method
    third = 1.0 / 3.0
    eps = 1.E-06
    a = 0.
    h = 0.
    if (abs(hx) >= 0) and (abs(hx) <= eps):                 a = 0
    if (abs(hx) > eps) and (abs(hx) < (1 - eps)):           h = abs(hx)
    if (abs(hx) > (1. - eps)) and (abs(hx) <= (1. + eps)):  a = 1.
    if abs(hx) > (1. + eps):                                h = 1. / abs(hx)

    if (h > 0) and (h < 1):
        a = h + 0.533659 * (1.0 - 2.0 * h + h ** third - (1.0 - h) ** third) - h * (1.0 - h) * (1.0 - 2.0 * h) \
            * (1.0 + 4.0 * (h * h + (1.0 - h) ** 2.0)) / 200.0 / PI

    if a > 1.:                  a = 1.
    if (a < 0.01) and (a < hx): a = hx

    return a

def wet_fra_zhang_sarica(ang, deng, denl, vc, vf, d, hlf, th0):
    # Calculate Wetted wall fraction according to Zhang and Sarica, SPEJ (2009)
    th = 1.0

    if abs(ang) >= 85. * PI / 180.:
        return th
    else:
        FRGF = deng * (vc - vf) ** 2 / (denl - deng) / G / d / np.cos(ang)
        FROT = 0.7 * np.sqrt((1.0 - hlf) / hlf)
        RFR = FRGF / FROT

        if RFR > 20: return th

        YO = -d * np.sin(PI * th0) ** 3 / (3.0 * PI * hlf)
        YI = d * 0.25 * (hlf ** 1.2 - 1.0)
        RAY = YO / YI

        if RAY < 0.000001: RAY = 0.000001

        COO = np.log(abs(RAY))
        RYD = abs(YI - YO) / d

        if RYD < 0.000001: return th

        xx = COO * RFR ** 1.4

        if abs(xx) > 600: return th

        YY = YO / np.exp(xx)

        if (RFR > 20) or (RYD < 0.000001): return th

        th = (1.0 + th0) / 2.0 + (1.0 - th0) * np.tan(PI * (2.0 * YY - YI - YO) / (3.0 * (YI - YO))) / 3.464

        if (th > 1.) or (YY > YI): th = 1.0
    return th

def SGL(D, ED, ANG, P, DEN, V, VIS):
    # Single Phase Flow Calculation
    # SGL  =  calculates pressure gradient for single phase flow of liquid or gas
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param P: pressure (Pa)
    :param DEN: density (kg/m3)
    :param V: velocity (m/s)
    :param VIS: viscosity (Pa-s)
    :param PGT: total pressure gradient
    :param PGF: friction pressure gradient
    :param PGG: gravity pressure gradient
    :return: FF, PGT, PGF, PGG, PGA
    """
    # Calculate elevation pressure gradient
    PGG = -DEN * np.sin(ANG) * G

    # Calculate frictional pressure gradient
    RE = abs(D * DEN * V / VIS)
    FF = get_fff(RE, ED)
    PGF = -2.0 * FF * DEN * V * V / D

    # Calculate acceleration pressure gradient
    if DEN <= 400:
        EKK = DEN * V * V / P
        ICRIT = 0
        if EKK > 0.95:
            ICRIT = 1
        if ICRIT == 1:
            EKK = 0.95
        PGT = (PGG + PGF) / (1.0 - EKK)
        PGA = PGT * EKK
    else:
        PGA = 0.0

    PGT = (PGG + PGF + PGA)

    return FF, PGT, PGF, PGG, PGA

def DISLUG(D, ED, ANG, VSG, DENL, DENG, VISL, STGL):
    # DISLUG - calculates the superficial liquid velocity on the boundary between dispersed bubble flow and slug flow
    #          with a given superficial gas velocity
    #       -- slug/bubbly transition, constant vsg
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param STGL: gas viscosity (Pa-s)
    :return: VDB, Ierr
    """
    global Ierr, AXP, IFFM
    Ierr = 0
    CC = 1.25 - 0.5 * abs(np.sin(ANG))
    VMC = VSG / (1.0 - HLSC)

    # guess a VDB and VM
    VDB1 = 2.0
    VM = 2 * VSG

    for icon in np.arange(1, 501):
        VM = VDB1 + VSG
        HLB = VDB1 / VM
        DENM = (1.0 - HLB) * DENG + HLB * DENL
        REM = abs(DENM * D * VM / VISL)
        FM = get_fff(REM, ED)

        if REM > 5000.0:
            VMN = np.sqrt(abs(((1.0 / HLB - 1.0) * 6.32 * CC * np.sqrt(abs(DENL - DENG) * G * STGL)) / (FM * DENM)))
        else:
            VMN = np.sqrt(abs(((1.0 / HLB - 1.0) * 6.32 * CC * np.sqrt(abs(DENL - DENG) * G * STGL))
                              * np.sqrt(5000.0 / REM) / (FM * DENM)))

        if VMN < VMC: VMN = VMC

        ABM = abs((VMN - VM) / VM)

        if ABM < E1: break

        VM = (VMN + 4.0 * VM) / 5.0
        VDB1 = VM - VSG

        if icon > 500: Ierr = 4

    VDB = VM - VSG

    return VDB, Ierr

def BUSLUG(D, ANG, VSL):
    # BUSLUG - calculates the superficial gas velocity on the boundary between slug flow and bubbly flow
    #          with a given superficial liquid velocity (for near vertical upward flow, >60 deg, and large D)
    #       -- slug/bubbly transition, constant vsl (for near vertical upward flow, >60 deg, and large D)
    """
    :param D: pipe diameter (m)
    :param ANG: angle of pipe from hor (rad)
    :param VSL: liquid superficial velocity (m/s)
    :return: VBU
    """
    VO = (0.54 * np.cos(ANG) + 0.35 * np.sin(ANG)) * np.sqrt(G * D)
    HGC = 0.25
    VBU = VSL * HGC / (1.0 - HGC) + VO * HGC
    return VBU

def STSLUG(D, ED, ANG, VSG, DENL, DENG, VISL, VISG, STGL, IFFM):
    # STSLUG  =  calculates the superficial liquid velocity on the boundary between slug flow and stratified
    #            (or annular)  flow with a given superficial gas velocity (for horizontal and downward flow)
    #       -- slug/stratified-annular transition, constant vsg (for horizontal and downward flow)
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param VISG: gas viscosity (Pa-s)
    :param IFFM: interfacial friction indicator
    :return: VST, Ierr
    """
    global Ierr, AXP
    CS = (32.0 * np.cos(ANG) ** 2 + 16.0 * np.sin(ANG) ** 2) * D
    CC = 1.25 - 0.5 * abs(np.sin(ANG))
    VDB, Ierr = DISLUG(D, ED, ANG, VSG, DENL, DENG, VISL, STGL)

    # Guess a VST
    VST = 0.5
    VM = VST + VSG
    HLS = 1.0 / (1.0 + abs(VM / 8.66) ** 1.39)

    if HLS < HLSC: HLS = HLSC

    FE = 0.0
    HLF = VST / VM
    VF = VM
    VC = VM
    REMX = 5000.0
    RESG = abs(DENG * VSG * D / VISG)
    WEB = abs(DENG * VSG * VSG * D / STGL)
    FRO = abs(np.sqrt(G * D) / VSG)
    VSGT = 5.0 * np.sqrt(DENA / DENG)

    # Initial interfacial absolute roughness
    EAI = D / 7.0
    FI = FIC
    FI1, FI2 = 0, 0

    for icon in np.arange(501):
        if VST > VDB:
            break

        # Entrainment fraction according to Oliemans et al.'s (1986) correlation
        RESL = abs(DENL * VST * D / VISL)
        CCC = 0.003 * WEB ** 1.8 * FRO ** 0.92 * RESL ** 0.7 * (DENL / DENG) ** 0.38 * \
              (VISL / VISG) ** 0.97 / RESG ** 1.24
        FEN = CCC / (1.0 + CCC)

        if FEN > FEC: FEN = FEC
        FE = (FEN + 9.0 * FE) / 10.0

        # Translational velocity according to Nicklin (1962), Bendiksen (1984) and Zhang et al. (2000)
        if REMX < 2000.0:
            VAV = 2.0 * VM
        elif REMX > 4000.0:
            VAV = 1.3 * VM
        else:
            VAV = (2.0 - 0.7 * (REMX - 2000.0) / 2000.0) * VM

        VT = VAV + (0.54 * np.cos(ANG) + 0.35 * np.sin(ANG)) * np.sqrt(G * D * abs(DENL - DENG) / DENL)
        HLFN = ((HLS * (VT - VM) + VST) * (VSG + VST * FE) - VT * VST * FE) / (VT * VSG)

        if HLFN <= 0.0: HLFN = abs(HLFN)
        if HLFN >= 1.0: HLFN = 1.0 / HLFN

        HLF = (HLFN + 9.0 * HLF) / 10.0
        HLC = (1.0 - HLF) * VST * FE / (VM - VST * (1.0 - FE))

        if HLC < 0.0: HLC = 0.0
        AF = HLF * AXP
        AC = (1.0 - HLF) * AXP

        # Calculate wetted wall fraction
        TH0 = wet_fra_biberg(HLF)

        # Wetted wall fraction according to Zhang and Sarica, SPEJ (2011)
        TH = wet_fra_zhang_sarica(ANG, DENG, DENL, VC, VF, D, HLF, TH0)

        # Wetted perimeters
        SF = PI * D * TH
        SC = PI * D * (1.0 - TH)
        AB = D * D * (PI * TH - np.sin(2.0 * TH * PI) / 2.0) / 4.0
        SI = (SF * (AB - AF) + D * np.sin(PI * TH) * AF) / AB

        # The hydraulic diameters
        DF = 4.0 * AF / (SF + SI)
        THF = 2.0 * AF / (SF + SI)
        DC = 4.0 * AC / (SC + SI)
        VC = (VM - VST * (1.0 - FE)) / (1.0 - HLF)

        # Reynolds numbers
        DENC = (DENL * HLC + DENG * (1.0 - HLF - HLC)) / (1.0 - HLF)
        REF = abs(DENL * VF * DF / VISL)
        REC = abs(DENG * VC * DC / VISG)

        # Friction factors
        FF = get_fff(REF, ED)
        FC = get_fff(REC, ED)

        # Interfacial friction factor:
        # Stratified flow interfacial friction factor
        if D <= 0.127:
            if IFFM == 1:
                FI1 = FC * (1.0 + 15.0 * abs(2.0 * THF / D) ** 0.5 * (VSG / VSGT - 1.0))
            else:
                FI1 = FC * (1.0 + 21.0 * abs(THF / D) ** 0.72 * abs(VSG / VSGT - 1.0) ** 0.8)
        else:
            # Interfacial friction factor according to Baker et al. (1988)
            WEE = DENG * VF * VF * EAI / STGL
            VIS = VISL * VISL / (DENL * STGL * EAI)
            WVM = WEE * VIS

            if WVM <= 0.005:
                EAI = 34.0 * STGL / (DENG * VF * VF)
            else:
                EAI = 170.0 * STGL * WVM ** 0.3 / (DENG * VF * VF)

            if EAI > THF / 4.0: EAI = THF / 4.0

            EDI = EAI / D
            FI2 = get_fff(REC, EDI)

        FIN = (0.127 * FI1 / D + FI2) / (1.0 + 0.127 / D)
        if FIN < FC: FIN=FC
        if FIN > 0.1: FIN = 0.1
        FI = (FIN + 9.0 * FI) / 10.0

        ABCD = (SC * FC * DENG * VC * abs(VC) / (2.0 * AC) + SI * FI * DENG * (VC - VF) * abs(VC - VF)
                * (1.0 / AF + 1.0 / AC) / 2.0 - (DENL - DENC) * G * np.sin(ANG)) * AF * 2.0 / (SF * FF * DENL)
        if ABCD < 0:
            VFN = VF * 0.9
        else:
            VFN = np.sqrt(ABCD)

        ABU = abs((VFN - VF) / VF)

        if ABU <= E1:
            VST = VFN * HLF / (1.0 - FE)
            break

        VF = (VFN + 9.0 * VF) / 10.0
        VST = VF * HLF / (1.0 - FE)
        VM = VST + VSG
        DPEX = (DENL * (VM - VF) * (VT - VF) * HLF + DENC * (VM - VC) * (VT - VC) * (1.0 - HLF)) * D / CS / 4.0
        REMX = abs(D * VM * DENL / VISL)
        FM = get_fff(REMX, ED)
        DPSL = FM * DENL * VM * VM / 2.0
        DPAL = DPSL + DPEX

        if REMX < 5000.: DPAL = DPAL * REMX / 5000.0

        AD = DPAL / (3.16 * CC * np.sqrt(STGL * abs(DENL - DENG) * G))
        HLSN = 1.0 / (1.0 + AD)

        if HLSN < HLSC: HLSN = HLSC

        HLS = (HLSN + 4.0 * HLS) / 5.0

        if icon >= 500: Ierr = 2

    if VST > VDB: VST = VDB

    return VST, Ierr

def ANSLUG(D, ED, ANG, VSL, DENL, DENG, VISL, VISG, STGL, IFFM):
    # ANSLUG - calculates the superficial gas velocity on the boundary between slug flow and annular (or stratified)
    #          flow with a given superficial liquid velocity (for upward flow)
    #       -- slug/stratified-annular transition, constant vsl (for upward flow)
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSL: liquid superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param VISG: gas viscosity (Pa-s)
    :param IFFM: interfacial friction indicator
    :return: VAN, Ierr
    """
    global Ierr, AXP

    Ierr = 0
    CS = (32.0 * np.cos(ANG) ** 2 + 16.0 * np.sin(ANG) ** 2) * D
    CC = 1.25 - 0.5 * abs(np.sin(ANG))
    V24 = 19.0 * VSL / 6.0

    # guess a VAN
    VAN = 10.
    VM = VSL + VAN
    HLS = 1.0 / (1.0 + abs(VM / 8.66) ** 1.39)

    if HLS < HLSC: HLS = HLSC

    FE = 0.0
    HLF = VSL / VM
    VC = VAN / (1.0 - HLF)
    VF = VSL / HLF
    FI = FIC
    REMX = 5000.0
    RESL = DENL * VSL * D / VISL
    VSGT = 5.0 * np.sqrt(DENA / DENG)
    ABCD = V24 ** 2

    # Initial interfacial absolute roughness
    EAI = D / 7.0
    FI1, FI2 = 0, 0

    for icon in np.arange(1001):
        # Entrainment fraction according to Oliemans et al's (1986) correlation
        WEB = DENG * VAN * VAN * D / STGL
        FRO = np.sqrt(G * D) / VAN
        RESG = DENG * VAN * D / VISG
        CCC = 0.003 * WEB ** 1.8 * FRO ** 0.92 * RESL ** 0.7 * (DENL / DENG) ** 0.38 * \
              (VISL / VISG) ** 0.97 / RESG ** 1.24
        FEN = CCC / (1.0 + CCC)

        if FEN > 0.75: FEN = 0.75

        FE = FEN

        # Translational velocity according to Nicklin (1962), Bendiksen (1984) and Zhang et al. (2000)
        if REMX < 2000.0:
            VAV = 2.0 * VM
        elif REMX > 4000.0:
            VAV = 1.3 * VM
        else:
            VAV = (2.0 - 0.7 * (REMX - 2000.0) / 2000.0) * VM

        VT = VAV + (0.54 * np.cos(ANG) + 0.35 * np.sin(ANG)) * np.sqrt(G * D * abs(DENL - DENG) / DENL)
        HLFN = ((HLS * (VT - VM) + VSL) * (VAN + VSL * FE) - VT * VSL * FE) / (VT * VAN)

        if HLFN <= 0.0: HLFN = abs(HLFN)
        if HLFN >= 1.0: HLFN = 1.0 / HLFN

        HLF = HLFN
        HLC = (1.0 - HLF) * VSL * FE / (VM - VSL * (1.0 - FE))

        if HLC < 0.0: HLC = 0.0

        AF = HLF * AXP
        AC = (1.0 - HLF) * AXP

        # Calculate wet wall fraction
        TH0 = wet_fra_biberg(HLF)

        # Wetted wall fraction according to Zhang and Sarica, SPEJ (2011)
        TH = wet_fra_zhang_sarica(ANG, DENG, DENL, VC, VF, D, HLF, TH0)

        # Wet perimeters
        SF = PI * D * TH
        SC = PI * D * (1.0 - TH)
        AB = D * D * (PI * TH - np.sin(2.0 * TH * PI) / 2.0) / 4.0
        SI = (SF * (AB - AF) + D * np.sin(PI * TH) * AF) / AB

        # The hydraulic diameters
        DF = 4.0 * AF / (SF + SI)
        THF = 2.0 * AF / (SF + SI)
        DC = 4.0 * AC / (SC + SI)
        VFN = VSL * (1.0 - FE) / HLF
        VF = (VFN + 9.0 * VF) / 10.0

        # Reynolds numbers
        DENC = (DENL * HLC + DENG * (1.0 - HLF - HLC)) / (1.0 - HLF)
        REF = abs(DENL * VF * DF / VISL)
        REC = abs(DENG * VC * DC / VISG)

        # Frictional factors
        FF = get_fff(REF, ED)
        FC = get_fff(REC, ED)

        if D <= 0.127:
            # Interfacial friction factor (stratified) according to Andritsos et al. (1987)
            # Modified by Zhang (2001)
            if IFFM == 1:
                FI1 = FC * (1.0 + 15.0 * abs(2.0 * THF / D) ** 0.5 * (VAN / VSGT - 1.0))
            else:
                # Use Fan's correlation (2005)
                FI1 = FC * (1.0 + 21.0 * abs(THF / D) ** 0.72 * abs(VAN / VSGT - 1.0) ** 0.8)

        else:
            # Interfacial friction factor according to Baker et al. (1988)
            WEE = DENG * VF * VF * EAI / STGL
            VIS = VISL * VISL / (DENL * STGL * EAI)
            WVM = WEE * VIS

            if WVM <= 0.005:
                EAI = 34.0 * STGL / (DENG * VF * VF)
            else:
                EAI = 170.0 * STGL * WVM ** 0.3 / (DENG * VF * VF)

            if EAI > THF / 4.0: EAI = THF / 4.0

            EDI = EAI / D
            FI2 = get_fff(REC, EDI)

        FIN = (0.127 * FI1 / D + FI2) / (1.0 + 0.127 / D)

        if FIN < FC: FIN = FC
        if FIN > 0.1: FIN = 0.1
        FI = (FIN + 9.0 * FI) / 10.0

        ABCDN = (SF * FF * DENL * VF * VF / (2.0 * AF) - SC * FC * DENG * VC * VC / (2.0 * AC)
                 + (DENL - DENC) * G * np.sin(ANG)) * 2.0 / (SI * FI * DENG * (1.0 / AF + 1.0 / AC))

        ABCD = (ABCDN + 9.0 * ABCD) / 10.0

        if ABCD < 0.0:
            VCN = VC * 0.9
        else:
            VCN = np.sqrt(ABCD) + VF

        if VCN < V24: VCN = V24

        ABU = abs((VCN - VC) / VC)
        VC = VCN

        if ABU < E1: break

        VAN = VC * (1.0 - HLF) - VSL * FE

        if VAN < 0: VAN = -VAN

        VM = VSL + VAN
        DPEX = (DENL * (VM - VF) * (VT - VF) * HLF + DENC * (VM - VC) * (VT - VC) * (1.0 - HLF)) * D / CS / 4.0
        REMX = abs(D * VM * DENL / VISL)
        FM = get_fff(REMX, ED)
        DPSL = FM * DENL * VM * VM / 2.0
        DPAL = DPSL + DPEX

        if REMX < 5000: DPAL = DPAL * np.sqrt(REMX / 5000.0)

        AD = DPAL / (3.16 * CC * np.sqrt(STGL * abs(DENL - DENG) * G))
        HLSN = 1.0 / (1.0 + AD)

        if HLSN < HLSC: HLSN = HLSC

        HLS = HLSN

        if icon >= 1000: Ierr = 3

    VAN = VC * (1.0 - HLF) - VSL * FE

    return VAN, Ierr

def DBFLOW(D, ED, ANG, VSG, VSL, DENL, DENG, VISL):
    # DBFLOW - calculates pressure gradient and liquid holdup for dispersed bubble flow (without bubble rise velocity)
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param VISG: gas viscosity (Pa-s)
    :return: FM, HL, PGT, PGA, PGF, PGG
    """
    VM = VSG + VSL

    # Calculate liquid holdup
    HL = VSL / (VSG + VSL)
    DENM = (1.0 - HL) * DENG + HL * DENL
    DENS = DENM + HL * (DENL - DENM) / 3.0
    REM = abs(DENS * D * VM / VISL)
    FM = get_fff(REM, ED)
    PGF = -2.0 * FM * DENS * VM ** 2 / D
    PGG = -G * DENM * np.sin(ANG)
    PGA = 0.0
    PGT = (PGF + PGG + PGA)
    return FM, HL, PGT, PGA, PGF, PGG

def BUFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, STGL):
    # BUFLOW - calculates pressure gradient and liquid holdup for bubbly flow (with bubble rise velocity Vo)
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param STGL: gas-liquid surface tension (N/m)
    :return: FM, HL, PGT, PGA, PGF, PGG
    """
    VM = VSG + VSL
    VO = 1.53 * abs(G * (DENL - DENG) * STGL / DENL / DENL) ** 0.25 * np.sin(ANG)

    # Calculate liquid holdup
    if abs(ANG) < 10. * PI / 180.:
        HL = VSL / (VSG + VSL)
    else:
        HL = (np.sqrt(abs((VM - VO) ** 2 + 4.0 * VSL * VO)) - VSG - VSL + VO) / (2.0 * VO)

    # CALCULATE PRESSURE GRADIENTS
    DENM = (1.0 - HL) * DENG + HL * DENL
    DENS = DENM + HL * (DENL - DENM) / 3.0
    REM = abs(DENS * D * VM / VISL)

    FM = get_fff(REM, ED)
    PGF = -2.0 * FM * DENS * VM ** 2 / D
    PGG = -G * DENM * np.sin(ANG)
    PGA = 0.
    PGT = (PGF + PGG + PGA)
    return FM, HL, PGT, PGA, PGF, PGG

def ITFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL):
    # ITFLOW - calculates pressure gradient, liquid holdup and slug characteristics for intermittent flow
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param VISG: gas viscosity (Pa-s)
    :param STGL: gas-liquid surface tension (N/m)
    :return: HL, FF, PGT, PGA, PGF, PGG, FGL, HLF, CU, CS, CF, VF, VC, FQN, RSU, HLS, HLS, ICON, IFGL
    """

    global Ierr
    Ierr = 0
    VM = VSL + VSG
    HLS = 1.0 / (1.0 + abs(VM / 8.66) ** 1.39)

    if HLS < HLSC: HLS = HLSC

    # Translational velocity according to Nicklin (1962), Bendiksen (1984) and Zhang et al. (2000)
    REMM = abs(DENL * D * VM / VISL)

    if REMM < 2000:
        VAV = 2.0 * VM
    elif REMM > 4000:
        VAV = 1.3 * VM
    else:
        VAV = (2.0 - 0.7 * (REMM - 2000.0) / 2000.0) * VM

    VT = VAV + (0.54 * np.cos(ANG) + 0.35 * np.sin(ANG)) * np.sqrt(G * D * abs(DENL - DENG) / DENL)

    # slug length
    CS = (32.0 * np.cos(ANG) ** 2 + 16.0 * np.sin(ANG) ** 2) * D
    CE = 1.25 - 0.5 * abs(np.sin(ANG))
    VSGT = 5.0 * np.sqrt(DENA / DENG)
    VMT = (VM + VT) / 2.0

    # Guess CU and CF
    CU = CS * VM / VSL
    CF = CU - CS
    HLF = (VT - VM) / VT
    VF = VSL / 2.0
    VFN1 = VSL / 2.0
    VC = VM
    FF = 0.1
    FI = FIC

    # Initial interfacial absolute roughness
    EAI = D / 7.0
    FGL = 'INT'
    IFGL = 4
    FI1, FI2 = 0, 0
    icon = 1

    for icon in np.arange(1, 2001):
        # Overall liquid holdup
        HL = (HLS * (VT - VM) + VSL) / VT
        HLF = HLS * (VT - VM) / (VT - VF)

        if HLF > HL:  HLF = 0.99 * HL

        VCN = (VM - HLF * VF) / (1.0 - HLF)

        if VCN < 0.0:   VCN = -VCN
        if VCN > VT:    VCN = VT

        VC = VCN
        CFN = (HLS - HL) * CS / (HL - HLF)
        ABU = abs((CFN - CF) / CF)
        CF = (CFN + 19.0 * CF) / 20.0
        AF = HLF * AXP
        AC = (1.0 - HLF) * AXP

        # Slug liquid holdup
        DPEX = (DENL * (VM - VF) * (VT - VF) * HLF + DENG * (VM - VC) * (VT - VC) * (1.0 - HLF)) * D / CS / 4.0
        REM = abs(DENL * VM * D / VISL)
        FM = get_fff(REM, ED)
        DPSL = FM * DENL * VM * VM / 2.0
        DPAL = DPSL + DPEX

        if REM < 5000:  DPAL = DPAL * (REM / 5000.0)

        AD = DPAL / (3.16 * CE * np.sqrt(STGL * abs(DENL - DENG) * G))
        HLSN = 1.0 / (1.0 + AD)

        if HLSN < HLSC: HLSN = HLSC

        HLS = HLSN

        if (REM < 1500) and (HLS < 0.6):   HLS = 0.6

        if (VSL / VM > HLS) or (abs(CF) < D):
            FGL = "D-B"
            IFGL = 2
            exit

        # Wetted wall fraction assuming flat film surface
        TH0 = wet_fra_biberg(HLF)

        # Wetted wall fraction according to Zhang and Sarica, SPEJ (2011)
        TH = wet_fra_zhang_sarica(ANG, DENG, DENL, VC, VF, D, HLF, TH0)

        # Wetted perimeters
        SF = PI * D * TH
        SC = PI * D * (1.0 - TH)
        AB = D * D * (PI * TH - np.sin(2.0 * TH * PI) / 2.0) / 4.0
        SI = (SF * (AB - AF) + D * np.sin(PI * TH) * AF) / AB

        # The hydraulic diameters
        DF = 4.0 * AF / (SF + SI)
        THF = 2.0 * AF / (SF + SI)
        DC = 4.0 * AC / (SC + SI)

        # Frictional factors
        REF = abs(DENL * VF * DF / VISL)
        REC = abs(DENG * VC * DC / VISG)
        FFN = get_fff(REF, ED)
        FC = get_fff(REC, ED)
        FF = (FFN + 9.0 * FF) / 10.0
        VSGF = VC * (1.0 - HLF)

        if D <= 0.127:
            if IFFM == 1:
                # Interfacial friction factor (stratified) according to Andritsos et al. (1987) Modified by Zhang (2001)
                FI1 = FC * (1.0 + 15.0 * abs(2.0 * THF / D) ** 0.5 * (VSGF / VSGT - 1.0))
            else:
                # Use Fan's correlation (2005)
                FI1 = FC * (1.0 + 21.0 * abs(THF / D) ** 0.72 * abs(VSGF / VSGT - 1.0) ** 0.8)
        else:

            # Interfacial friction factor according to Baker et al. (1988)
            WEE = DENG * VF * VF * EAI / STGL
            VIS = VISL * VISL / (DENL * STGL * EAI)
            WVM = WEE * VIS

            if WVM <= 0.005:
                EAI = 34.0 * STGL / (DENG * VF * VF)
            else:
                EAI = 170.0 * STGL * WVM ** 0.3 / (DENG * VF * VF)

            if EAI > THF / 4.0: EAI = THF / 4.0

            EDI = EAI / D
            FI2 = get_fff(REC, EDI)

        FRS = (0.127 * FI1 / D + FI2) / (1.0 + 0.127 / D)

        # Interfacial friction factor (annular) according to Ambrosini et al. (1991)
        REG = abs(VC * DENG * D / VISG)
        WED = DENG * VC * VC * D / STGL
        FIF = 0.046 / REG ** 0.2
        SHI = abs(FI * DENG * (VC - VF) ** 2 / 2.0)
        THFO = THF * np.sqrt(abs(SHI * DENG)) / VISG
        FRA = FIF * (1.0 + 13.8 * (THFO - 200.0 * np.sqrt(DENG / DENL)) * WED ** 0.2 / REG ** 0.6)
        FRA1 = get_fff(REC, THF / D)

        if FRA > FRA1:  FRA = FRA1

        RTH = SF / THF
        FIN = (50.0 * FRS / RTH + FRA) / (1.0 + 50.0 / RTH)

        if FIN < FC:    FIN = FC
        if FIN > 0.1:   FIN = 0.1

        FI = (FIN + 9.0 * FI) / 10.0

        # Calculate film length CF using the combined momentum equation
        FSL = (DENL * (VM - VF) * (VT - VF) - DENG * (VM - VC) * (VT - VC)) / CF
        ABCD = (FSL + SC * FC * DENG * VC * abs(VC) / 2.0 / AC +
                SI * FI * DENG * (VC - VF) * abs(VC - VF) / 2.0 * (1.0 / AF + 1.0 / AC) -
                (DENL - DENG) * G * np.sin(ANG)) * 2.0 * AF / (SF * FF * DENL)

        if ABCD > 0:
            VFN = np.sqrt(ABCD)
            if VFN > VM:    VFN = VM
        else:
            VFN = -np.sqrt(-ABCD)
            if VFN < -VM:   VFN = -VM

        ABV = abs((VFN - VF) / VF)
        VF = (VFN + 19.0 * VF) / 20.0

        if (ABU < E1) or (ABV < E1): break
        if icon >= 2000:
            Ierr = 1
            break
    if FGL == "D-B":
        IFGL = 2
        SL = D * PI
        FF, HL, PGT, PGA, PGF, PGG = DBFLOW(D, ED, ANG, VSG, VSL, DENL, DENG, VISL)
        TH = 1.0
        THF = D / 2.0
        RSU = 0.0
        HLF = HL
        VF = VM
        VC = 0.0
        CF = 0.0
        CU = 0.0
        FE = 0.0
        FQN = 0.0
        SI = 0.0
        FI = 0.0
    else:
        # Slug unit length
        CU = CF + CS
        DENM = DENL * HLS + DENG * (1.0 - HLS)
        DENS = DENM + HLS * (DENL - DENM) / 3.0
        RES = abs(DENS * VM * D / VISL)
        FS = get_fff(RES, ED)
        FQN = VT / CU  # slug frequency
        RSU = CS / CU  # slug to slug unit length ratio

        # Pressure gradient in slug
        FOS = RSU * (DENL * (VM - VF) * (VT - VF) * HLF + DENG * (VM - VC) * (VT - VC) * (1.0 - HLF)) / CS
        DPS = -FS * DENS * VM * VM * 2.0 / D - DENM * G * np.sin(ANG) - FOS

        # Pressure gradient in film
        FOF = FOS * CS / CF
        DPF = FOF - SF * FF * DENL * VF * abs(VF) / (2.0 * AXP) - DENG * FC * SC * \
            VC * abs(VC) / (2.0 * AXP) - (DENL * HLF + DENG * (1.0 - HLF)) * G * np.sin(ANG)

        # Gravitational pressure gradient
        PGG = -(DENL * HL + DENG * (1.0 - HL)) * G * np.sin(ANG)

        # Frictional pressure gradient
        PGF = -((FS * DENS * (VF + HLS * (VM - VF)) ** 2.0 * 2.0 / D) * CS +
                    (SF * FF * DENL * VF * abs(VF) / (2.0 * AXP) +
                    SC * FC * DENG * VC * abs(VC) / (2.0 * AXP)) * CF) / CU

        # Acceleration pressure gradient
        PGA = 0.0

        # Total pressure gradient
        PGT = PGG + PGF + PGA

    return HL, FF, PGT, PGA, PGF, PGG, FGL, HLF, CU, CS, CF, VF, VC, FQN, RSU, HLS, icon, IFGL

def SAFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P):
    # SAFLOW  =  calculates pressure gradient and liquid holdup for stratified or annular flow
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (rad)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param VISG: gas viscosity (Pa-s)
    :param STGL: gas-liquid surface tension (N/m)
    :param P: pressure (P)
    :return: HL, FE, FF, PGT, PGA, PGF, PGG, FGL, HLF, VF, SF, THF, ICON, IFGL
    """
    global Ierr
    Ierr = 0
    FI = FIC
    # Initial interfacial absolute roughness
    EAI = D / 7.0
    FGL = 'STR'
    IFGL = 5
    FI1, FI2 = 0, 0
    ICON = 1
    FF = 0

    # Entrainment fraction according to Oliemans et al's (1986) correlation
    RESG = abs(DENG * VSG * D / VISG)
    WEB = abs(DENG * VSG * VSG * D / STGL)
    FRO = abs(VSG / np.sqrt(G * D))
    RESL = abs(DENL * VSL * D / VISL)
    CCC = 0.003 * WEB ** 1.8 * FRO ** (-0.92) * RESL ** 0.7 * (DENL / DENG) ** 0.38 \
          * (VISL / VISG) ** 0.97 / RESG ** 1.24
    FE = CCC / (1.0 + CCC)

    if FE > FEC: FE = FEC

    VSGT = 5.0 * np.sqrt(DENA / DENG)

    # Guess a film velocity
    VF = VSL
    VC = VSL + VSG
    HLF = VSL / (VSL + VSG)
    ABCD = 5.0
    TH = 0.5

    for ICON in np.arange(1, 2001):
        if VF > 0:
            HLFN = VSL * (1.0 - FE) / VF
        else:
            HLFN = 1.2 * HLF

        if HLFN >= 1.:
            HLFN = 1.0 - 1.0 / HLFN

        HLF = (HLFN + 19.0 * HLF) / 20.0

        if VF > 0:
            VCN = (VSG + FE * VSL) / (1.0 - HLF)
        else:
            VCN = (VSG + VSL - VF * HLF) / (1.0 - HLF)

        VC = (VCN + 9.0 * VC) / 10.0
        AF = HLF * AXP
        AC = (1.0 - HLF) * AXP

        if VF > 0:
            HLC = VSL * FE / VC
        else:
            HLC = (VSL - VF * HLF) / VC

        if HLC < 0: HLC = 0

        DENC = (DENL * HLC + DENG * (1.0 - HLF - HLC)) / (1.0 - HLF)

        # Wetted wall fraction assuming flat film surface
        TH0 = wet_fra_biberg(HLF)

        # Wetted wall fraction according to Zhang and Sarica, SPEJ (2011)
        TH = wet_fra_zhang_sarica(ANG, DENG, DENL, VC, VF, D, HLF, TH0)

        # Wetted perimeters
        SF = PI * D * TH
        SC = PI * D * (1.0 - TH)
        AB = D * D * (PI * TH - np.sin(2.0 * TH * PI) / 2.0) / 4.0
        SI = (SF * (AB - AF) + D * np.sin(PI * TH) * AF) / AB

        # The hydraulic diameters
        DF = 4.0 * AF / (SF + SI)
        THF = 2.0 * AF / (SF + SI)
        DC = 4.0 * AC / (SC + SI)

        # Frictional factors
        REF = abs(DENL * VF * DF / VISL)
        REC = abs(DENG * VC * DC / VISG)
        FFN = get_fff(REF, ED)
        FC = get_fff(REC, ED)
        FF = (FFN + 9.0 * FF) / 10.0
        VSGF = VC * (1.0 - HLF)

        if D <= 0.127:
            if IFFM == 1:
                # Interfacial friction factor (stratified) according to Andritsos et al. (1987) Modified by Zhang (2001)
                FI1 = FC * (1.0 + 15.0 * abs(2.0 * THF / D) ** 0.5 * (VSGF / VSGT - 1.0))
            else:
                # Use Fan's correlation (2005)
                FI1 = FC * (1.0 + 21.0 * abs(THF / D) ** 0.72 * abs(VSGF / VSGT - 1.0) ** 0.8)
        else:

            # Interfacial friction factor according to Baker et al. (1988)
            WEE = DENG * VF * VF * EAI / STGL
            VIS = VISL * VISL / (DENL * STGL * EAI)
            WVM = WEE * VIS

            if WVM <= 0.005:
                EAI = 34.0 * STGL / (DENG * VF * VF)
            else:
                EAI = 170.0 * STGL * WVM ** 0.3 / (DENG * VF * VF)

            if EAI > THF / 4.0: EAI = THF / 4.0

            EDI = EAI / D
            FI2 = get_fff(REC, EDI)

        FRS = (0.127 * FI1 / D + FI2) / (1.0 + 0.127 / D)

        # Interfacial friction factor (annular) according to Ambrosini et al. (1991)
        REG = abs(VC * DENG * D / VISG)
        WED = DENG * VC * VC * D / STGL
        FIF = 0.046 / REG ** 0.2
        SHI = abs(FI * DENG * (VC - VF) ** 2 / 2.0)
        THFO = THF * np.sqrt(abs(SHI * DENG)) / VISG
        FRA = FIF * (1.0 + 13.8 * (THFO - 200.0 * np.sqrt(DENG / DENL)) * WED ** 0.2 / REG ** 0.6)
        FRA1 = get_fff(REC, THF / D)

        if FRA > FRA1:  FRA = FRA1

        RTH = SF / THF
        FIN = (50.0 * FRS / RTH + FRA) / (1.0 + 50.0 / RTH)

        if FIN < FC:    FIN = FC
        if FIN > 0.1:   FIN = 0.1

        FI = (FIN + 9.0 * FI) / 10.0

        ABCD = (SC * FC * DENG * VC * abs(VC) / 2.0 / AC +
                SI * FI * DENG * (VC - VF) * abs(VC - VF) / 2.0 * (1.0 / AF + 1.0 / AC) -
                (DENL - DENG) * G * np.sin(ANG)) * 2.0 * AF / (SF * FF * DENL)

        if ABCD > 0:
            VFN = np.sqrt(ABCD)
        else:
            VFN = 0.95 * VF

        ABU = abs((VFN - VF) / VF)
        VF = (VFN + 9.0 * VF) / 10.0
        # print(VF, FI, ABU)

        if ABU < E1: break

        if ICON >= 2000:
            Ierr = 6
            break

    # Total pressure gradient due to friction
    PGF = -SF * FF * DENL * VF * abs(VF) / (2.0 * AXP)-SC * FC * DENG * VC * abs(VC) / (2.0 * AXP)
    PGG = -(DENL * HLF + DENC * (1.0 - HLF)) * G * np.sin(ANG)

    # total pressure gradient
    PGT = (PGF + PGG) / (1.0 - DENG * VC * VSG / (P * (1.0 - HLF)))

    # Total pressure gradient due to acceleration
    PGA = PGT - PGF - PGG

    # liquid holdup
    HL = HLF + HLC

    return HL, FE, FF, PGT, PGA, PGF, PGG, FGL, HLF, VF, SF, THF, ICON, IFGL

def ITFLOW_ETC(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, PGT, PGF, PGG, PGA, FGL, IFGL, VF,
               VC, VT, HLF, SL, SI, THF, TH, TH0, FE, FF, FI, HLS, CU, CF, FQN, RSU, ICON, extra_in):

    # ----Initialize
    VM = VSL + VSG  # mixture velocity
    FGL = 'INT'
    IFGL = 4
    SL = D * PI

    HL, FF, PGT, PGA, PGF, PGG, FGL, HLF, CU, CS, CF, VF, VC, FQN, RSU, HLS, ICON, IFGL = \
        ITFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL)

    if FGL == 'D-B':
        SL = D * PI
        FF, HL, PGT, PGA, PGF, PGG = DBFLOW(D, ED, ANG, VSG, VSL, DENL, DENG, VISL)
        THF = D / 2.0
        TH = 1.0
        RSU = 0.0
        HLF = HL
        VF = VM
        VC = 0.0
        CF = 0.0
        CU = 0.0
        FE = 0.0
        FQN = 0.0
        SI = 0.0
        FI = 0.0
    elif FGL == 'STR':
        HL, FE, FF, PGT, PGA, PGF, PGG, FGL, HLF, VF, SF, THF, ICON, IFGL = \
            SAFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P)
        HLS = HL
        SL = SF

    return CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, ICON

def GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, extra_in):
    """
    :param D: pipe diameter (m)
    :param ED: relative pipe wall roughness
    :param ANG: angle of pipe from hor (deg)
    :param VSL: liquid superficial velocity (m/s)
    :param VSG: gas superficial velocity (m/s)
    :param DENL: liquid density (kg/m3)
    :param DENG: gas density (kg/m3)
    :param VISL: liquid viscosity (Pa-s)
    :param VISG: gas viscosity (Pa-s)
    :param STGL: liquid surface tension (N/m)
    :param P: pressure (Pa)
    :param extra_in: additional input vector/list
    :return: FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, ICON
    """
    global Ierr, AXP
    ANG = ANG * np.pi / 180.    # convert deg to rad

    FE, FI, FQN, HLF, HLS, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, \
    IFGL, ICON, THF, FF, CU, CF, HL = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', 0, 0, 0, 0, 0, 0, 0

    # ----Initialize
    IFFM = extra_in[0]      # set the interfacial friction factor method
    Ierr = 0
    VM = VSL + VSG          # mixture velocity
    AXP = PI * D * D / 4.0  # cross sectional area of the pipe
    ICON = 0

    # - --------------------------
    # Check for single phase flow
    # - --------------------------

    ENS = VSL / VM
    HLS = ENS

    if ENS >= 0.99999:          # liquid
        FGL = 'LIQ'
        IFGL = 1
        HL = 1.0
        SL = D * PI
        FF, PGT, PGF, PGG, PGA = SGL(D, ED, ANG, P, DENL, VSL, VISL)
        THF = D / 2.0
        TH = 1.0

    elif ENS <= 0.0000001:      # gas
        FGL = 'GAS'
        IFGL = 7
        HL = 0.0
        SL = 0.0
        FF, PGT, PGF, PGG, PGA = SGL(D, ED, ANG, P, DENL, VSL, VISL)
        TH = 0.0
        THF = 0.0

    else:
        # - --------------------------
        # Check INT - D-B transition boundary
        # - --------------------------
        FGL = 'N-A'
        IFGL = 0
        if ENS > 0.36:
            VDB, Ierr = DISLUG(D, ED, ANG, VSG, DENL, DENG, VISL, STGL)
            if VSL > VDB:
                FGL = 'D-B'
                IFGL = 2
                SL = D * PI
                FF, HL, PGT, PGA, PGF, PGG = DBFLOW(D, ED, ANG, VSG, VSL, DENL, DENG, VISL)
                TH = 1.0
                THF = D / 2.0
                RSU = 0.0
                HLF = HL
                VF = VM
                VC = 0.0
                CF = 0.0
                CU = 0.0
                FE = 0.0
                FQN = 0.0
                SI = 0.0
                FI = 0.0

        if FGL != 'D-B':
            if ANG <= 0:        # downhill or horizontal
                # - --------------------------
                # Check I-SA transition boundary for downward flow (mostly I-S)
                # - --------------------------
                VST, Ierr = STSLUG(D, ED, ANG, VSG, DENL, DENG, VISL, VISG, STGL, IFFM)
                if VSL < VST:
                    HL, FE, FF, PGT, PGA, PGF, PGG, FGL, HLF, VF, SF, THF, ICON, IFGL = \
                        SAFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P)
                else:
                    CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, \
                    TH, TH0, VF, VC, VT, FGL, IFGL, ICON = \
                    ITFLOW_ETC(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, PGT, PGF, PGG, PGA, FGL, IFGL, VF,
                               VC, VT, HLF, SL, SI, THF, TH, TH0, FE, FF, FI, HLS, CU, CF, FQN, RSU, ICON, extra_in)

            else:               # uphill
                VAN, Ierr = ANSLUG(D, ED, ANG, VSL, DENL, DENG, VISL, VISG, STGL, IFFM)
                if VSG > VAN:
                    FGL = 'ANN'
                    IFGL = 6
                    SL = D * PI
                    # print(FGL)
                    HL, FE, FF, PGT, PGA, PGF, PGG, FGL, HLF, VF, SF, THF, ICON, IFGL = \
                        SAFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P)
                    SL = SF
                    HLS = HL
                else:
                    # - --------------------------
                    # Check I-BU transition boundary
                    # - --------------------------
                    CKD = (DENL * DENL * G * D * D / (abs(DENL - DENG) * STGL)) ** 0.25
                    if CKD <= 4.36:
                        CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, \
                        TH, TH0, VF, VC, VT, FGL, IFGL, ICON = \
                            ITFLOW_ETC(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, PGT, PGF, PGG, PGA, FGL,
                                       IFGL, VF, VC, VT, HLF, SL, SI, THF, TH, TH0, FE, FF,
                                       FI, HLS, CU, CF, FQN, RSU, ICON, extra_in)
                    else:
                        VBU = BUSLUG(D, ANG, VSL)
                        if (VSG < VBU) and (ANG > 60 * PI / 180):
                            FGL = 'BUB'
                            IFGL = 3
                            SL = D * PI
                            FM, HL, PGT, PGA, PGF, PGG = BUFLOW(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, STGL)
                            TH = 1.0
                            THF = D / 2.0
                            RSU = 0.0
                            HLF = HL
                            VF = VM
                            VC = 0.0
                            CF = 0.0
                            CU = 0.0
                            FE = 0.0
                            FQN = 0.0
                            SI = 0.0
                            FI = 0.0
                        else:
                            CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, \
                            TH, TH0, VF, VC, VT, FGL, IFGL, ICON = \
                                ITFLOW_ETC(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P,
                                           PGT, PGF, PGG, PGA, FGL, IFGL, VF, VC, VT, HLF, SL, SI,
                                           THF, TH, TH0, FE, FF, FI, HLS, CU, CF, FQN, RSU, ICON, extra_in)

    return CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, ICON

def fgreservoir(P_res, pwf, C_res, n, GLR):
    """
    Rawlins and Schellhardt (1935) IPR for a gas well
    :param P_res: reservoir pressure in pa
    :param pwf: bottom hole pressure in pa
    :param den_gas_relative: 
    :param C_res: 
    :param n: 
    :param GLR: Gas liquid ratio
    :return: reservoir gas and liquid flow rates in m3/s
    """
    t_std = 273.15
    p_std = 1.013e5
    fg_res = C_res*(P_res**2 - pwf**2)**n
    fl_res = fg_res/GLR
    return fg_res, fl_res

def gas_density(den_gas_relative, t, p):
    """
    Get real gas density in Kg/m3
    :param den_gas_relative: relative gas density to air density (-)
    :param t: temperature in K
    :param p: pressure in Pa
    :return: real gas density in Kg/m3
    """
    z = z_factor_Starling(t, p, den_gas_relative)
    if z<0.3:
        z=0.3
    elif z>1:
        z=1
    # print('p:', p,  '\t',   'z:',  round(z, 2),   '\t',   't:', round(t, 2),  '\t')
    return 3488 * p * den_gas_relative / z / t * 1e-6

def gas_viscosity(den_gas_relative, t, p):
    """
    Based on Lee & Wattenbarger correlation
    :param den_gas_relative: relative gas density to air density (-)
    :param t: temperature in K
    :param p: pressure in Pa
    :return: gas viscosity in Pa.s
    """
    deng = gas_density(den_gas_relative, t, p) / 1000   # g/cm3
    Mg = den_gas_relative * M_air * 1000                # g/mol

    x = 3.448 + 986.4 / (1.8 * t) + 0.01 * Mg
    y = 2.447 - 0.2224 * x
    k = (9.379 + 0.01607 * Mg) * (1.8 * t) ** 1.5 / (209.2 + 19.26 * Mg + 1.8 * t)
    gas_vis = k * 1e-4 * np.exp(x * deng ** y)          # cp
    # print('deng:', deng,  '\t',   'k:',  round(k, 2),   '\t',   'x:', round(x, 2),  '\t',   'y:', round(y, 2),  '\t')
    return gas_vis / 1000                               # Pa.s

def liquid_viscosity(den_liquid_relative, t):
    """
    Based on Beggs and Robinson
    :param den_liquid_relative: specific gravity of oil
    :param t: temperature in K
    :return: viscosity in Pa.s
    """
    if abs(den_liquid_relative - 1.) <= 0.1:
        t = t-273.15
        return np.exp(1.003 - 1.479 * 1e-2 * (1.8 * t + 32) + 1.982 * 1e-5 * (1.8 * t + 32)**2) * 1e-3
    else:
        tf = (t - 273.15) * 1.8 + 32
        API = 141.5 / den_liquid_relative - 131.5
        x = 1.8653 - 0.025086*API - 0.5644*np.log10(tf)
        oil_vis = 10**(10**x) - 1
        return oil_vis/1000

def surface_tension(den_liquid_relative, den_gas_relative, t):
    """
    calculate the surface tension of oil or water
    For water-gas: Firoozabadi & Ramey, 1988
    For oil-gas: Abdul-Majeed & Abu Al-Soof, 2000
    :param den_liquid_relative: specific gravity of liquid
    :param den_gas_relative: relative gas density to air density (-)
    :param t: temperature in K
    :return: surface tension in N/m
    """
    if abs(den_liquid_relative - 1) > 1e-2:
        API = 141.5 / den_liquid_relative - 131.5
        F = (t - 273.15) * 1.8 + 32.0
        A = 1.11591 - 0.00305 * F
        return A * (38.085 - 0.259 * API) / 1000.
    else:
        # water-gas
        Kw = 4.5579 * 18 ** 0.15178 * (1.0)**(-0.84573)         # Watson characterization factor, °R1/3
        Tc = 24.2787 * Kw ** 1.76544 * (1.0)**2.12504           # Riazi, M.R. and Daubert, T.E. 1980.
        T = (t - 273.15) * 1.8 + 491.67                         # Celsius to Rankine
        Tr = T / Tc                                             # reduced temperature
        sigma = ((1.58*(den_liquid_relative - den_gas_relative * 1.25e-3) + 1.76)/Tr**0.3125)**4 / 1000
        return 0.072

def z_factor_Starling(t, p, den_gas_relative):
    # '''
    #     Hall-Yarborough z factor model, programmed by Haiwen Zhu
    # '''
    # p = abs(p) / 6894.76  # Pa to psi
    # Ppc = 756.8 - 131.07 * den_gas_relative - 3.6 * den_gas_relative ** 2
    # Tpc = 169.2 + 349.5 * den_gas_relative - 74.0 * den_gas_relative ** 2
    # Pr = p / Ppc
    # Tr = abs(t) / Tpc

    # xi_1 = 0.06125 * Pr / Tr * np.exp(-1.2 * (1 - 1 / Tr) ** 2)
    # z=1
    
    # for i in range (1, 1000, 1):
    #     t = 1 / Tr
    #     Y = xi_1
    #     A = ((((Y - 4) * Y + 4) * Y + 4) * Y + 1) / (1 - Y) ** 4
    #     b = -((9.16 * t - 19.52) * t + 29.52) * t * Y
    #     c = (2.18 + 2.82 * t) * ((42.4 * t - 242.2) * t + 90.7) * t * Y ** (1.182 + 2.82 * t)
    #     dfFnc = A + b + c

    #     A = -0.06125 * Pr * t * np.exp(-1.2 * (1 - t) ** 2)
    #     b = (((1 - Y) * Y + 1) * Y + 1) * Y / (1 - Y) ** 3
    #     c = -((4.58 * t - 9.76) * t + 14.76) * t * Y * Y
    #     d = ((42.4 * t - 242.2) * t + 90.7) * t * Y ** (2.18 + 2.82 * t)
    #     fFnc = A + b + c + d

    #     xi = xi_1 - fFnc / dfFnc
    #     if abs((xi - xi_1) / xi_1) < 0.01 or abs(fFnc) < 0.01:
    #         # Method converge
    #         z = 0.06125 * Pr / Tr * np.exp(-1.2 * (1 - 1 / Tr) ** 2) / xi
    #         break
    #     else:
    #         xi_1 = 0.99*xi_1+0.01*xi
    
    # if i == 100 or z<0:
    #     z=1
    # return z
    

    # Following method have converge problem
    """
    The Dranchuk and Abou-Kassem (1973) equation of state is based on the generalized Starling equation of state
    :param t: Temperature in K
    :param p: Pressure in Pa
    :param den_gas_relative: specific gravity of gas
    :return: compressibility factor of real gas
    """
    # not converge for some case

    i = 0
    if p < 35e6:
        p = abs(p) / 6894.76  # Pa to psi
        Ppc = 756.8 - 131.07 * den_gas_relative - 3.6 * den_gas_relative ** 2
        Tpc = 169.2 + 349.5 * den_gas_relative - 74.0 * den_gas_relative ** 2
        Pr = p / Ppc
        Tr = abs(t) / Tpc
        A1 = 0.3265
        A2 = -1.0700
        A3 = -0.5339
        A4 = 0.01569
        A5 = -0.05165
        A6 = 0.5475
        A7 = -0.7361
        A8 = 0.1844
        A9 = 0.1056
        A10 = 0.6134
        A11 = 0.7210
        z = 1.
        
        while True:
            i += 1
            rour = 0.27 * Pr / (z * Tr)
            z_n = 1 + (A1 + A2 / Tr + A3 / Tr ** 3 + A4 / Tr ** 4 + A5 / Tr ** 5) * rour \
                  + (A6 + A7 / Tr + A8 / Tr ** 2) * rour ** 2 - A9 * (A7 / Tr + A8 / Tr ** 2) * rour ** 5 \
                  + A10 * (1 + A11 * rour ** 2) * (rour ** 2 / Tr ** 3) * np.exp(-A11 * rour ** 2)
            if z_n>1.5:
                z_n=1.5
            elif z_n<0.1:
                z_n=0.1
            
            if abs(z_n - z) / abs(z) > 1e-3:
                z = 0.05 * z_n + 0.95 * z

                if i > 1000:
                    break
            else:
                break
        
        # if z < 0:
        #     z = 1.

        # Newton method
        # for i in range (1, 100, 1):
        #     DenR = 0.27 * Pr / (z * Tr)
        #     dpdz = -DenR / z
        #     fz = (A1 + A2 / Tr + A3 / Tr ** 3 + A4 / Tr ** 4 + A5 / Tr ** 5) * DenR + \
        #             (A6 + A7 / Tr + A8 / Tr ** 2) * DenR ** 2 - A9 * (A7 / Tr + A8 / Tr ** 2) \
        #                 * DenR ** 5 + A10 * (1 + A11 * DenR ** 2) * (DenR ** 2) / (Tr ** 3) * \
        #                     np.exp(-A11 * DenR ** 2) + 1 - z
        #     dz = (2 * A5 * A6 * DenR / Tr ** 5 + A4 / Pr ** 4 + A3 / Tr ** 3 + A8 * \
        #         (2 - 5 * A9 * DenR ** 3) * DenR / Tr ** 2 + A7 * ((2 - 5 * A9 * DenR ** 3) \
        #             * DenR + A2) / Tr + A1 + 2 * A10 * A11 * (DenR ** 2 - A11 * DenR ** 4 + 1) * \
        #                 DenR * np.exp(-A11 * DenR ** 2) / Tr ** 3) * dpdz - 1
    

        #     zi = z - fz / dz
        #     if abs((zi - z) / z) < 0.001 or abs(fz) < 0.001:
        #         # Method converge
        #         break
        #     else:
        #         z = zi
        
        # Hall_Yarborough Z factor, programed by Haiwen Zhu


        return z

    else:
        z1 = 1.
        e = 1.
        tc = 191.05
        pc = 4.6407
        P = p / 1e6
        a = 0.4278 * tc ** 2.5 / (pc * t ** 2.5)
        b = 0.0867 * tc / (pc * t)
        while e > 1e-3 and i < 20:
            z2 = (z1 ** 2 + a ** 2 * b * P ** 2 - z1 * (a - b ** 2 * P - b)) ** (1. / 3)
            e = abs(z2 - z1)
            z1 = z2
            i += 1

        if z1 < 0 or z1 > 1:
            z1 = 1.

        return z1

if __name__ == "__main__":
    D = 0.1
    ED = 0
    ANG = 0     # Horizontal
    VSL = 0.48
    VSG = 0.021
    DENL = 850
    DENG = 18
    VISL = 0.001
    VISG = 0.000018
    STGL = 0.025
    P = 2075502
    extra_in = [1]

    CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, ICON = \
        GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, extra_in)

    print(PGT, PGA, PGF, PGG, HL, FGL)

#     """
#     Nodal Test
    
#     """
#     inputValues = {"Surface_line_pressure": 101325, "Pipe_diameter":0.1,        "Pipe_Length":5000,    
#                     "Roughness":0,                   "Inclination_angle":0,      "Liquid_viscosity":0.001,
#                     "Liquid_relative_density":0.85,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
#                     "Surface_tension_GL":0.0254,     "Reservoir_C":2.58e-15,     "Reservoir_n":1, 
#                     "Reservoir_P":6e6,               "GLR":5,                   "Geothermal_gradient":0.03, 
#                     "Gas_solubility":0.1,            "dL": 100,                  "Sand_diameter":0.0001, 
#                     "Sand_sphericity":1,             "Surface_T":288 }
#     n = 100                          # maximum run times
#     DF_HL=[]
#     DF_Ppipe=[]
#     DF_Tpipe=[]
#     DF_Qpipe=[]
#     DF_VL=[]
#     DF_VG=[]
#     DF_Qipr=[]
#     DF_Pipr=[]
#     DF_Qopr=[]
#     DF_Popr=[]

#     D = inputValues['Pipe_diameter']
#     ED = inputValues['Roughness']
#     ANG = inputValues['Inclination_angle']
#     DENL = inputValues['Liquid_relative_density'] * 1000
#     DENG_stg = gas_density(inputValues['Gas_relative_density'], 273.15, 1.013e5)
#     A_pipe = 0.25*PI*inputValues['Pipe_diameter']**2
#     pwf=0       # Used to obtain the max flow from reservoir
#     Qg_res, Ql_res = fgreservoir(inputValues['Reservoir_P'], pwf,
#                                         inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])
#     Qmax_res=Qg_res+Ql_res
    
#     i = 0
#     j = 0
#     jmax = inputValues['Pipe_Length']/inputValues['dL']
#     # while i <= n:
               
#     #     # Calculate bottom hole pressure from surface and flow rate
#     #     P=inputValues['Surface_line_pressure']
#     #     T=inputValues['Surface_T']
#     #     for j in range(0, int(jmax), 1):
#     #         VSL = Ql_res/A_pipe
#     #         DENG = gas_density(inputValues['Gas_relative_density'], T, P)
#     #         VSG = Qg_res/A_pipe/DENG*DENG_stg
#     #         VISL = liquid_viscosity(inputValues['Liquid_relative_density'], T)
#     #         VISG = gas_viscosity(inputValues['Gas_relative_density'], T, P)
#     #         STGL = surface_tension(inputValues['Liquid_relative_density'], inputValues['Gas_relative_density'], T)
#     #         # CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, \
#     #         #         ICON = GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
#     #         _,_,_,_,_,HLF, HLS, HL, PGT, PGA, PGF, PGG,_,_,_,_,_,_,_,_,_,_,_= GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
#     #         P=P-PGT*inputValues['dL']
#     #         T=T+inputValues['Geothermal_gradient']*inputValues['dL']
#     #         # print('P:', round(P, 2),  '\t',   'deng:', round(DENG, 2),  '\t',   'PGT:', round(PGT, 2),  '\t',   'HL:', round(HL, 2),  '\t' ,   'Qg_res:', round(Qg_res, 2),  '\t'  )
                        
#     #     # check if converge
#     #     if abs((pwf-P)/P)>1:
#     #         # update near well pressure and flow from reservoir
#     #         # print('error:', (pwf-P)/P,  '\t',   'pwf:',  round(pwf, 2),   '\t',   'P:', round(P, 2),  '\t',   'deng:', round(DENG, 2),  '\t',   'PGT:', round(PGT, 2),  '\t' )
#     #         pwf=(pwf+P)/2
#     #         Qg_res, Ql_res = fgreservoir(inputValues['Reservoir_P'], pwf,
#     #                                             inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])
#     #         i=i+1
#     #         print(i)

#     #     else:
#             # # Nodal analysis
#             # for j in range (0, 10, 1):
#             #     pwf=j/10*inputValues['Reservoir_P']
#             #     if pwf == inputValues['Reservoir_P']:
#             #         pwf=0.99*pwf
#             #     Qg_res, Ql_res = fgreservoir(inputValues['Reservoir_P'], pwf,
#             #                                                 inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])
#             #     P=inputValues['Surface_line_pressure']
#             #     T=inputValues['Surface_T']
#             #     for j in range(0, int(jmax), 1):
#             #         VSL = Ql_res/A_pipe
#             #         DENG = gas_density(inputValues['Gas_relative_density'], T, P)
#             #         VSG = Qg_res/A_pipe/DENG*DENG_stg
#             #         VISL = liquid_viscosity(inputValues['Liquid_relative_density'], T)
#             #         VISG = gas_viscosity(inputValues['Gas_relative_density'], T, P)
#             #         STGL = surface_tension(inputValues['Liquid_relative_density'], inputValues['Gas_relative_density'], T)
#             #         # CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, \
#             #         #         ICON = GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
#             #         _,_,_,_,_,_,_,_,PGT,_,_,_,_,_,_,_,_,_,_,_,_,_,_= GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
#             #         P=P-PGT*inputValues['dL']
#             #         T=T+inputValues['Geothermal_gradient']*inputValues['dL']

#             #     DF_Qipr.append(Qg_res+Ql_res)
#             #     DF_Pipr.append(pwf)
#             #     DF_Qopr.append(Qg_res+Ql_res)
#             #     DF_Popr.append(P)

#             # Pipe flow information 
#     for i in range (0, 10, 1):
#         print(i)
#         pwf=i/10*inputValues['Reservoir_P']
#         if pwf == inputValues['Reservoir_P']:
#             pwf=0.99*pwf
#         Qg_res, Ql_res = fgreservoir(inputValues['Reservoir_P'], pwf,
#                                                     inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])
#         P=inputValues['Surface_line_pressure']
#         T=inputValues['Surface_T']             
#         for j in range(0, int(jmax), 1):
#             VSL = Ql_res/A_pipe
#             DENG = gas_density(inputValues['Gas_relative_density'], T, P)
#             VSG = Qg_res/A_pipe/DENG*DENG_stg
#             VISL = liquid_viscosity(inputValues['Liquid_relative_density'], T)
#             VISG = gas_viscosity(inputValues['Gas_relative_density'], T, P)
#             STGL = surface_tension(inputValues['Liquid_relative_density'], inputValues['Gas_relative_density'], T)
#             # CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, \
#             #         ICON = GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
#             _,_,_,_,_,HLF, HLS, HL, PGT, PGA, PGF, PGG,_,_,_,_,_,_,_,_,_,_,_= GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
#             P=P-PGT*inputValues['dL']
#             T=T+inputValues['Geothermal_gradient']*inputValues['dL']
#             DF_HL.append(HL)
#             DF_Ppipe.append(P)
#             DF_Tpipe.append(T)
#             DF_Qpipe.append(Ql_res+Qg_res)
#             DF_VL.append(VSL/HL)
#             DF_VG.append(VSG/(1-HL))

#             print('P:', round(P, 2),  '\t',   'deng:', round(DENG, 2),  '\t',   'PGT:', round(PGT, 2),  '\t',   'HL:', round(HL, 2),  '\t' ,   'Qg_res:', round(Qg_res, 2),  '\t'  )
#             data = pd.DataFrame({"HL": DF_HL,  "Ppipe": DF_Ppipe,       "Tpipe": DF_Tpipe, "Qpipe": DF_Qpipe,
#                                 "VL": DF_VL,      "VG": DF_VG})

# data.to_excel("output.xlsx")