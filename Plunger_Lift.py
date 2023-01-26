# -*- coding: utf-8 -*-
"""
Created on Thu Oct 1 16:31:06 2018 based on A. Gupta, 2017 (Computers and Chemical Engineering)
Include the functions for calculations, assumptions:
1) liquid downward flow is neglected
2) a simple choke valve flow rate for gas
3) constant plunger drop velocity in gas/liquid columnsvp
4) average temperature and pressure for gas column pressure
5) others
@author: jiz172

Modification History:
Jianjun Zhu     Jun 04 2019     Added unified_model() to consider the horizontal section of well
Jianjun Zhu     Feb 18 2019     Modified real gas compressibility function with Dranchuk and Abou-Kassem equation
Jianjun Zhu     Feb 16 2019     Replace bhp_balance_downward_liquid() and bhp_balance_downward_gas() with a combined
                                function bhp_balance_downward() to calculate the net inflow flux to tubing and annulus
Jianjun Zhu     Feb 15 2019     Modified upward() add plunger sink in gas/liquid columns due to Vp < 0
Jianjun Zhu     Feb 14 2019     Modified friction factor calculation by standard form of Churchill equation
Jianjun Zhu     Feb 05 2019     Modified main() to enable multiple plunger lift cycle calculation
Jianjun Zhu     Jan 31 2019     Modified upward() to eliminate discontinuity when liquid slug reaches the top
Jianjun Zhu     Jan 25 2019     Added a print_io() function to output intermediate results on console
Haiwen Zhu      Mar 18 2020     Change dt, cycles, Plunger lift period (original 1600s), surface valve open time (700s) to public variables
Haiwen Zhu      Mar 22 2020     Added "updateUI" function for input & output for GUI
                                #add inputs in the "initialize" function to update global variables between GUI.py and current .py
"""

import sys
import os
path_abs = os.path.dirname(os.path.abspath(__file__))   # current path
path = os.path.join(path_abs, 'C:\\Users\\haz328\\Desktop\\Github') # join path
sys.path.append(path)  # add to current py
import time
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from TUALP_Models import GAL
#from plot_data import plot_data

"""global constants"""
PI = pi                  # 3.1415926
G = 9.81                    # gravitational acceleration m/s/s
R = 8.314                   # J/mol/K
M_air = 28.97e-3            # air molecular weight in kg/mol
UiMPa = 10e6                # MPa to Pa

"""global variables"""
# plunger parameters
m_plunger = 5               # plunger mass in kg
L_plunger = 0.4             # plunger length in m
d_plunger = 1.9 * 0.0254    # plunger diameter in
Cd_plunger = 0.1019         # plunger fall drag coefficient
Epsilon_plunger = 0.0457    # plunger rise

A_plunger = 0.25*PI*d_plunger**2   # plunger cross sectional area in m2

# well parameters
H_well = 10000 * 0.3048     # well depth in m
ID_tubing = 1.995 * 0.0254  # tubing ID in m
OD_tubing = 2.125 * 0.0254  # tubing OD in m
ID_casing = 4.85 * 0.0254   # casing ID in m
ED = 2.5e-4                 # tubing roughness

A_tubing = 0.25*PI*(ID_tubing**2)                   # tubing cross sectional area in m2
A_casing = 0.25*PI*(ID_casing**2 - OD_tubing**2)    # annulus cross section area in m2
AB = 0.25*PI*ID_casing**2                           # cross-sectional area of casing
Ann = 0.25*PI*(ID_tubing**2 - d_plunger**2)         # cross section area between plunger and tubing in m2

# reservoir IPR
P_res = 6e6                 # reservoir pressure in Pa
C_res = 2.58e-15            # reservoir coefficient for gas flow rate in Vogel model
n = 1                       # reservoir index for gas flow rate in Vogel model
GLR = 10                    # gas-liquid-ratio 
Rsl = 0.1                   # average gas solubility in liquid at average T and P, m3/m3
Tgrd = 0.03                 # formation temperature gradient 30K/km

# fluid properties
den_liquid_relative = 0.85  # relative density of liquid
den_gas_relative = 0.7      # relative density of gas
vis_liquid = 4e-4           # liquid viscosity in Pa.s
denl = den_liquid_relative * 1000
M_gas = M_air * den_gas_relative
fluid_type = 2                   # 1: black oil model, 2 compositional model
# surface parameters
Cv = 0.5e-7                 # coefficient of motor valve
T_wellhead = 288            # temperature at well head in K

# wellbore section
LB = 1500 * 0.3048          # deviated section of the wellbore
VB = AB * LB                # horizontal section volume in m3
ANG = 1                     # inclination angle of the deviated section
DB = ID_casing              # assume inner diameter the same as casing ID

"""initial variables"""
Pc = 1.6e6                  # casing pressure
Pt = 1.57e6                 # tubing pressure
Pl = 1.1e6                  # production line pressure - fixed if no surface line or separator considered
Ltt = 5.                    # initial liquid column length above plunger (m)
Ltb = 0.                    # initial liquid column length below plunger (m)
dt_H=0.5                    # time step for horizontal section
dt_U=5.                     # time step for plunger upward section
dt_D=10.0                   # time step for plunger downward section
cycles=4.                   # Plunger lift cycles to be computed
Period=30.*60.              # Plunger lift period (s to min)
T_open=12.*60.              # Surface valve open time (s to min)

# define state parameters
mga = 0.                    # mass of gas in annulus (kg)
mla = 0.                    # mass of liquid in annulus (kg)
mgtt = 0.                   # mass of gas in tubing top (kg)
mltt = 0.                   # mass of liquid in tubing top (kg)
mgtb = 0.                   # mass of gas in tubing bottom (kg)
mltb = 0.                   # mass of liquid in tubing bottom (kg)
Xp = 0.                     # plunger position from tubing shoe (m), initially at the bottom
Vp = 0.                     # plunger velocity (m/s)
Ar = 0.                     # plunger arrival time (s)
v = 0                       # motor valve open or close: 0 - close, 1 - open

# intermediate variables
La = 0.                     # ASSUME initial liquid column length in annulus (m)
Pcb = Pc                    # annular pressure right above liquid level
Ptb = Pt                    # tubing gas pressure right above plunger
Pwf = 0                     # bottom pressure at tubing shoe
PwfB = 0.                   # bottom pressure at wellbore
Fgout = 0.                  # surface gas mass flow rate (kg/s)
Flout = 0.                  # surface liquid mass flow rate (kg/s)
t = 0.                      # calculation time
dt = 0.01                   # delta t in s
PGT = 0.                    # total pressure gradient in horizontal well (pa/m)
Ppb = 0.                    # pressure on the bottom of plunger
Ppt = 0.                    # pressure on top of the plunger
Acc = -1                    # plunger acceleration in m/s2
Fgres = 0.                  # gas flow from horizontal well kg/s
Flres = 0.                  # liquid flow from horizontal well kg/s
FgresB = 0.                 # gas flow from reservoir kg/s
FlresB = 0.                 # liquid flow from reservoir kg/s
Fgtub = 0.                  # gas flow into tubing kg/s
Fltub = 0.                  # liquid flow into tubing kg/s
Fgann = 0.                  # gas flow into annulus kg/s
Flann = 0.                  # liquid flow into annulus kg/s
Ltt0 = Ltt                  # initial liquid height in tubing for each plunger cycle
HLB = 1                     # liquid holdup in the inclined or horizontal section of wellbore


#PR flash model
z_f = [0.13,0.18,61.92,14.08,8.35,0.97,3.41,0.84,1.48,1.79,6.85]
Pc_f = [493,1070.6,667.8,707.8,616.3,529.1,550.7,490.4,488.6,436.9,350]
w_f = [0.04,0.23,0.01,0.1,0.15,0.18,0.19,0.23,0.25,0.3,0.38]
Tc_f = [-232.4,87.9,-116.6,90.1,206,275,305.7,369.1,385.7,453.7,650]
M_f = [28.02,44.01,16.04,30.07,44.1,58.12,58.12,72.15,72.15,86.18,143]

def calPR(P,t,z_f=[0.13,0.18,61.92,14.08,8.35,0.97,3.41,0.84,1.48,1.79,6.85],Pc_f=[493,1070.6,667.8,707.8,616.3,529.1,550.7,490.4,488.6,436.9,350],w_f=[0.04,0.23,0.01,0.1,0.15,0.18,0.19,0.23,0.25,0.3,0.38],Tc_f=[-232.4,87.9,-116.6,90.1,206,275,305.7,369.1,385.7,453.7,650],M_f=[28.02,44.01,16.04,30.07,44.1,58.12,58.12,72.15,72.15,86.18,143]):
    #PR
    import math
    w_f = [0.04, 0.23, 0.01, 0.1, 0.15, 0.18, 0.19, 0.23, 0.25, 0.3, 0.38]
    z_f = [0.13, 0.18, 61.92, 14.08, 8.35, 0.97, 3.41, 0.84, 1.48, 1.79, 6.85]
    Tc_f = [-232.4, 87.9, -116.6, 90.1, 206, 275, 305.7, 369.1, 385.7, 453.7, 650]
    Pc_f = [493, 1070.6, 667.8, 707.8, 616.3, 529.1, 550.7, 490.4, 488.6, 436.9, 350]
    M_f = [28.02, 44.01, 16.04, 30.07, 44.1, 58.12, 58.12, 72.15, 72.15, 86.18, 143]
    z, Pc, w, Tc, M = z_f, Pc_f, w_f, Tc_f, M_f
    P=P/6895
    t=(t-273.15)*9/5+32
    #print(P,t)
    Tol=0.000001
    #determine kieguess
    kie=[0 for i in range(len(w))]
    for i in range(len(w)):
        kie[i]=(Pc[i]/P)*math.exp(5.37*(1+w[i])*(1-(Tc[i]+459.67)/(t+459.67)))
    #determine the first calculated kic based on the kie
    kicPR,zv,zl=PREOS(z,Pc,P,w,Tc,t,kie)
    for j in range(10000):
        error=0
        for i in range(len(z)):
            error=error+(kicPR[i]/kie[i]-1)**2
        if error>Tol:
            for i in range(len(z)):
                kie[i]=kicPR[i]
            kicPR,zv,zl=PREOS(z,Pc,P,w,Tc,t,kie)
            #x,y=xiyi(z,Pc,P,w,Tc,t,kie)
            #print(x)
        else:
            x,y=xiyi(z,Pc,P,w,Tc,t,kicPR)
            #print("x:",x)
            #print("y:",y)
            R=10.73
            t=t+459.67
            Tr=[0 for i in range(len(z))]
            ul=[0 for i in range(len(z))]
            vc=[0 for i in range(len(z))]
            xi=[0 for i in range(len(z))]
            Pch=[0 for i in range(len(z))]
            for i in range(len(z)):
                Tc[i]=Tc[i]+459.67
            for i in range(len(z)):
                Tr[i]=t/Tc[i]
            #gas density
            sum=0
            for i in range(len(z)):
                sum=sum+y[i]*M[i]
            denv=P*sum/zv/R/t
            #print("gas density=",denv*16.018) #kg/m3
            #liquid density
            sum=0
            for i in range(len(z)):
                sum=sum+x[i]*M[i]
            denl=P*sum/zl/R/t
            #print("liquid density=",denl*16.018) #kg/m3
            #gas viscosity calculation Lee et al for PR
            Mg=0
            for i in range(len(z)):
                Mg=M[i]*y[i]+Mg
            Kv=(9.4+0.02*Mg)*t**1.5/(209+19*Mg+t)
            Xv=3.5+986/t+0.01*Mg
            Yv=2.4-0.2*Xv
            visv=Kv/10000*math.exp(Xv*(denv/62.4)**Yv)
            #print("gas viscosity=",visv/1000) #cp
            #liquid viscosity calculation Lohrenz Bray and Clark correlation for PR
            for i in range(len(z)):
                xi[i]=5.4402*(Tc[i])**(1/6)/(M[i])**0.5/Pc[i]**(2/3)
                if Tr[i-1]>1.5:
                    ul[i]=17.78/100000*(4.58*Tr[i]-1.67)**0.625/xi[i]
                else:
                    ul[i]=34/100000*Tr[i]**0.94/xi[i]
            sum1=0
            sum2=0
            for i in range(len(z)):
                sum1=sum1+x[i]*ul[i]*M[i]**0.5
                sum2=sum2+x[i]*M[i]**0.5
            visol=sum1/sum2
            SG=0.75
            #Nitrogen,co2,methane,ethane,propane,i-butane,n-butane,i-pentane,n-pentane,hexane,heptane+
            vc[0]=0.051
            vc[1]=0.034
            vc[2]=0.099
            vc[3]=0.079
            vc[4]=0.074
            vc[5]=0.072
            vc[6]=0.07
            vc[7]=0.068
            vc[8]=0.067
            vc[9]=0
            for i in range(10,len(z)):
                vc[i]=21.573+0.015122*M[i]-27.656*SG+0.070615*M[i]*SG
            sum3=0
            TcL=0
            PcL=0
            ML=0
            for i in range(10):
                sum3=sum3+x[i]*M[i]*vc[i]
            for i in range(10,len(z)):
                sum3=sum3+x[i]*vc[i]
            for i in range(len(z)):
                ML=ML+M[i]*x[i]
                TcL=TcL+Tc[i]*x[i]
                PcL=PcL+Pc[i]*x[i]
            denr=denl/ML*sum3
            Xim=5.4402*TcL**(1/6)/(ML**0.5*PcL**(2/3))
            visl=visol+1/Xim*((0.1023+0.023364*denr+0.058533*denr**2-0.040758*denr**3+0.0093724*denr**4)**4-0.0001)
            #print("liquid viscosity=",visl/1000)
            #surface tension for PR
            ML=0
            Mv=0
            sum5=0
            Pch[0]=41
            Pch[1]=78
            Pch[2]=77
            Pch[3]=108
            Pch[4]=150.3
            Pch[5]=181.6
            Pch[6]=189.9
            Pch[7]=225
            Pch[8]=231.5
            Pch[9]=271
            for i in range(10,len(z)):
                Pch[i]=-4.6148734+2.558855*M[i]+3.4004065*0.0001*M[i]**2+3.767396*1000/M[i]
            for i in range(len(z)):
                ML=ML+M[i]*x[i]
                Mv=Mv+M[i]*y[i]
            a=denl/62.4/ML
            b=denv/62.4/Mv
            for i in range(len(z)):
                sum5=sum5+Pch[i]*(a*x[i]-b*y[i])
            Sigma=sum5**4
            #print("surface tension=",Sigma/145) #N/m
            return denv*16.018, denl*16.018, visv/1000, visl/1000, Sigma/145

def PbPR(z, Pc, w, Tc, t):
    import math
    Pb = 1000
    # for i in range(len(w)):
    # Pb=Pb+0.01*z[i]*Pc[i]*math.exp(5.37*(1+w[i])*(1-(Tc[i]+459.67)/(t+459.67)))
    # print(Pb)
    Tol = 0.000001
    kie = [0 for i in range(len(w))]
    for n in range(10000):
        for i in range(len(z)):
            kie[i] = (Pc[i] / Pb) * math.exp(5.37 * (1 + w[i]) * (1 - (Tc[i] + 459.67) / (t + 459.67)))
        kicPR, zv, zl = PREOS(z, Pc, Pb, w, Tc, t, kie)
        for j in range(10000):
            error = 0
            for i in range(len(z)):
                error = error + (kicPR[i] / kie[i] - 1) ** 2
            if error > Tol:
                for i in range(len(z)):
                    kie[i] = kicPR[i]
                kicPR, zv, zl = PREOS(z, Pc, Pb, w, Tc, t, kie)
                # x,y=xiyi(z,Pc,Pb,w,Tc,t,kicPR)
        s = 0
        # xx=0
        for k in range(len(z)):
            s = s + z[i] * kicPR[i]
            # xx=x[i]+xx

        if abs(1 - s) < 0.001:
            return Pb
        elif s < 1:
            Pb = Pb - 1
        elif s > 1:
            Pb = Pb + 1

def PREOS(z,Pc,P,w,Tc,t,kie):
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    Ca=0.45724
    Cb=0.0778
    s=[0 for i in range(len(z))]
    kic=[0 for i in range(len(z))]
    ALPHA=[0 for i in range(len(z))]
    x,y=xiyi(z,Pc,P,w,Tc,t,kie)
    for i in range(len(z)):
        s[i]=0.37464+1.54226*w[i]-0.26992*w[i]*w[i]
        ALPHA[i]=(1+s[i]*(1-((t+459.67)/(Tc[i]+459.67))**0.5))**2
        sum1=sum1+x[i]*(Tc[i]+459.67)*(ALPHA[i]/Pc[i])**0.5
        sum2=sum2+y[i]*(Tc[i]+459.67)*(ALPHA[i]/Pc[i])**0.5
        sum3=sum3+x[i]*(Tc[i]+459.67)/Pc[i]
        sum4=sum4+y[i]*(Tc[i]+459.67)/Pc[i]
    AL=Ca*P/(t+459.67)/(t+459.67)*sum1**2
    AV=Ca*P/(t+459.67)/(t+459.67)*sum2**2
    BL=Cb*P/(t+459.67)*sum3
    BV=Cb*P/(t+459.67)*sum4
    zv=cubicmax(1,BV-1,(AV-2*BV-3*BV*BV),-(AV*BV-BV**2-BV**3))
    zl=cubicmin(1,BL-1,(AL-2*BL-3*BL*BL),-(AL*BL-BL**2-BL**3))
    s1=0
    s2=0
    s3=0
    s4=0
    import math
    for i in range(len(z)):
        s1=s1+x[i]*math.sqrt(ALPHA[i])*(Tc[i]+459.67)/math.sqrt(Pc[i])
        s2=s2+y[i]*math.sqrt(ALPHA[i])*(Tc[i]+459.67)/math.sqrt(Pc[i])
        s3=s3+x[i]*(Tc[i]+459.67)/Pc[i]
        s4=s4+y[i]*(Tc[i]+459.67)/Pc[i]
    for i in range(len(z)):
        Aial=math.sqrt(ALPHA[i])*(Tc[i]+459.67)/math.sqrt(Pc[i])/s1
        Aiav=math.sqrt(ALPHA[i])*(Tc[i]+459.67)/math.sqrt(Pc[i])/s2
        Bibl=(Tc[i]+459.67)/Pc[i]/s3
        Bibv=(Tc[i]+459.67)/Pc[i]/s4
        FL=math.exp(Bibl*(zl-1)-math.log(zl-BL)-AL/BL*(2*Aial-Bibl)*math.log(1+BL/zl))
        FV=math.exp(Bibv*(zv-1)-math.log(zv-BV)-AV/BV*(2*Aiav-Bibv)*math.log(1+BV/zv))
        kic[i]=FL/FV
    return kic,zv,zl

def xiyi(z,Pc,P,w,Tc,t,kie):
    VF=NewtonVF(z,Pc,P,w,Tc,t,kie)
    x=[0 for i in range(len(z))]
    y=[0 for i in range(len(z))]
    for i in range(len(z)):
        x[i]=(z[i]*0.01)/(1+VF*(kie[i]-1))
        y[i]=x[i]*kie[i]
    return x,y

def NewtonVF(z,Pc,P,w,Tc,t,kie):
    #x0 is your initial guess
    x0=0.5
    Tol=0.000001
    xi_l=x0
    for i in range(1,1000):
        fxi_l=fFnc(xi_l,z,Pc,P,w,Tc,t,kie)
        dxi_l=dfFnc(xi_l,z,Pc,P,w,Tc,t,kie)
        xi=xi_l-fxi_l/dxi_l
        if abs(xi-xi_l)<Tol:
            NewtonVF=xi
            return NewtonVF
        else:
            xi_l=xi
    if i>1000:
        NewtonVF=-0.999
        print("Method did not converge")

def dfFnc(x,z,Pc,P,w,Tc,t,kie):
    dfFnc=0
    for i in range(len(z)):
        dfFnc=dfFnc+(z[i]*0.01)*(kie[i]-1)**2/(((kie[i]-1)*x+1)**2)
    dfFnc=-dfFnc
    return dfFnc

def fFnc(x,z,Pc,P,w,Tc,t,kie):
    fFnc=0
    for i in range(len(z)):
        fFnc=fFnc+(z[i]*0.01)*(kie[i]-1)/((kie[i]-1)*x+1)
    return fFnc

def compmax(a,b,c):
    if a>b:
        x=b
        b=a
        a=x
    if b>c:
        x=c
        c=b
        b=x
    if a>b:
        x=b
        b=a
        a=x
    compmax=c
    return compmax

def compmin(a,b,c):
    if a<b:
        x=b
        b=a
        a=x
    if b<c:
        x=c
        c=b
        b=x
    if a<b:
        x=b
        b=a
        a=x
    if c<0:
        if b<0:
            x=a
        else:
            x=b
    else:
        x=c
    compmin=x
    return compmin

def sgn(a):
    if a>0:
        sgn=1
    if a==0:
        sgn=0
    if a<0:
        sgn=-1
    return sgn

def cubicmax(a,b,c,d):
    e=b**2-3*a*c
    f=b*c-9*a*d
    g=c**2-3*b*d
    h=f**2-4*e*g
    root=[0,0,0]
    if round(e,8)==0 and round(f,8)==0:
        nroots=3
    #for i in range(3):
        #root[i-1]=-b/(3*a)
        cubicmax=-b/(3*a)
        import sys
        return cubicmax
        sys.exit(1)
    if round(h,10)==0:
        nroots=3
        root[0]=-b/a+f/e
        root[1]=-f/e/2
        root[2]=-f/e/2
        cubicmax=compmax(root[0],root[1],root[2])
    elif round(h,10)>0:
        nroots=1
        x=e*b+3*a*(-f+abs(h)**0.5*sgn(h))/2
        y=e*b+3*a*(-f-abs(h)**0.5*sgn(h))/2
        root[0]=(-b-abs(x)**(1/3)*sgn(x)-abs(y)**(1/3)*sgn(y))/(3*a)
        cubicmax=root[0]
    else:
        nroots=3
        import math
        tmp3=math.acos((2*e*b-3*a*f)/(2*(abs(e)**(3/2)*sgn(e))))/3
        root[0]=(-b-2*abs(e)**0.5*sgn(e)*math.cos(tmp3))/(3*a)
        root[1]=(-b+abs(e)**0.5*sgn(e)*(math.cos(tmp3)+3**0.5*math.sin(tmp3)))/(3*a)
        root[2]=(-b+abs(e)**0.5*sgn(e)*(math.cos(tmp3)-3**0.5*math.sin(tmp3)))/(3*a)
        cubicmax=compmax(root[0],root[1],root[2])
    return cubicmax

def cubicmin(a,b,c,d):
    e=b**2-3*a*c
    f=b*c-9*a*d
    g=c**2-3*b*d
    h=f**2-4*e*g
    root=[0,0,0]
    if round(e,8)==0 and round(f,8)==0:
        nroots=3
        cubicmin=-b/(3*a)
        return cubicmin
    if round(h,10)==0:
        nroots=3
        root[0]=-b/a+f/e
        root[1]=-f/e/2
        root[3]=-f/e/2
        cubicmin=compmin(root[0],root[1],root[2])
    elif round(h,10)>0:
        nroots=1
        x=e*b+3*a*(-f+abs(h)**0.5*sgn(h))/2
        y=e*b+3*a*(-f-abs(h)**0.5*sgn(h))/2
        root[0]=(-b-abs(x)**(1/3)*sgn(x)-abs(y)**(1/3)*sgn(y))/(3*a)
        cubicmin=root[0]
    else:
        nroots=3
        import math
        tmp3=math.acos((2*e*b-3*a*f)/(2*(abs(e)**(3/2)*sgn(e))))/3
        root[0]=(-b-2*abs(e)**0.5*sgn(e)*math.cos(tmp3))/(3*a)
        root[1]=(-b+abs(e)**0.5*sgn(e)*(math.cos(tmp3)+3**0.5*math.sin(tmp3)))/(3*a)
        root[2]=(-b+abs(e)**0.5*sgn(e)*(math.cos(tmp3)-3**0.5*math.sin(tmp3)))/(3*a)
        cubicmin=compmin(root[0],root[1],root[2])
    return cubicmin


def after_flow():

    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann

    index = 0
    Fgann_max = Fgres
    Fgann_min = - mga/dt
    Fgann = (Fgann_max + Fgann_min) / 2

    PtOri, PptOri, PtbOri = Pt, Ppt, Ptb
    Pt0, Ppt0 = Pt, Ppt

    Fltub = Flres
    Flann = 0

    try:
            
        while True:
            index += 1

            Fltub, Fgtub = Flres - Flann, Fgres - Fgann
            Ltb = (mltb + Fltub * dt) / (denl * A_tubing)
            La = (mla + Flann * dt) / (denl * A_casing)

            if La < 0: 
                Ltb += La * A_casing/ A_tubing
                La = 0

            i = 0
            while True:
                i += 1
                Fgout = fgout(gas_density(den_gas_relative, T_wellhead, Pt, fluid_type), Pt, Pl)
                Pt_ave, Tt_ave  = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well - L_plunger - Ltb)) / 2
                Zt_ave = z_factor_Starling(Tt_ave, Pt_ave, den_gas_relative)
                Pt_n = (mgtb + (Fgtub - Fgout) * dt) * Zt_ave * R * Tt_ave / (A_tubing * (H_well - L_plunger - Ltb) * M_gas)
                Ptb_n = p_bottom_open(Pt, Pt_ave, Tt_ave, ID_tubing, H_well - L_plunger - Ltb, Fgout, Zt_ave)

                Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
                Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
                Pc_n = (mga + Fgann * dt) * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
                Pcb_n = Pc * exp(M_gas * G / (Zc_ave * R * Tc_ave) * (H_well - La))

                if abs(Pt_n - Pt) / abs(Pt) > 1e-3 or abs(Ptb - Ptb_n) / abs(Ptb) > 1e-3 or \
                        abs(Pc_n - Pc) / abs(Pc) > 1e-3 or abs(Pcb_n - Pcb) / abs(Pcb) > 1e-3:
                    if i > 50:
                        break
                    Pt = (4*Pt + Pt_n) / 5
                    Ptb = (4*Ptb + Ptb_n) / 5
                    Pc = (4 * Pc + Pc_n) / 5
                    Pcb = (4 * Pcb + Pcb_n) / 5

                    if Pt < Pl:
                        Pt, Ppt = Pt0, Ppt0
                else:
                    break

            Pwft = Ptb + denl * G * Ltb
            Pwfa = Pcb + denl * G * La

            if Pwfa / Pwft > 1.001:
                Fgann_max = Fgann
                Fgann = (Fgann_max + Fgann_min) / 2
            elif Pwfa / Pwft < 0.999:
                Fgann_min = Fgann
                Fgann = (Fgann_max + Fgann_min) / 2
            else:
                break

            if index > 20:
                break

        # update gas mass and BHP
        mga += Fgann * dt
        mla += Flann * dt
        mgtb += (Fgtub - Fgout) * dt
        mltb += Fltub * dt
        Pwf = Pwfa

    except:
        print('after_flow model not converge')

def bhp_balance_downward():
    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann

    Fgann_min = -mga / dt
    Fgann_max = Fgres
    Fgann = (Fgann_max + Fgann_min) / 2
    mgtot = mgtt + mgtb
    mltot = mltt + mltb

    Flann = Flres
    Fltub = 0

    mla += Flann * dt
    La = mla / (denl * A_casing)

    if Xp > Ltb:
        Lt = (mltot + Fltub * dt) / (denl * A_tubing)
        Ltb = Lt
    else:
        Lt = (mltot + Fltub * dt) / (denl * A_tubing) + L_plunger

    if La < 0:
        Lt += La * A_casing / A_tubing
        La = 0
        mla = 0

    index = 0
    while True:
        index += 1
        Fgtub = Fgres - Fgann

        i = 0
        while True:
            i += 1
            Pt_ave, Tt_ave = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well - Lt)) / 2
            Zt_ave = z_factor_Starling(Tt_ave, Pt_ave, den_gas_relative)
            Pt_n = (mgtot + Fgtub * dt) * Zt_ave * R * Tt_ave / (A_tubing * (H_well - Lt) * M_gas)
            Ptb_n = Pt * exp(M_gas * G / (Zt_ave * R * Tt_ave) * (H_well - Lt))

            Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
            Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
            Pc_n = (mga + Fgann * dt) * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
            Pcb_n = Pc * exp(M_gas * G / (Zc_ave * R * Tc_ave) * (H_well - La))

            if abs(Pt_n - Pt) / abs(Pt) > 1e-3 or abs(Ptb - Ptb_n) / abs(Ptb) > 1e-3 or \
                    abs(Pc_n - Pc) / abs(Pc) > 1e-3 or abs(Pcb_n - Pcb) / abs(Pcb) > 1e-3:
                if i > 100:
                    break
                Pt = (4*Pt + Pt_n) / 5
                Ptb = (4*Ptb + Ptb_n) / 5
                Pc = (4*Pc + Pc_n) / 5
                Pcb = (4*Pcb + Pcb_n) / 5
            else:
                break

        Pwft = Ptb + denl * G * Lt
        Pwfa = Pcb + denl * G * La

        if Pwfa / Pwft > 1.001:
            Fgann_max = Fgann
            Fgann = (Fgann_max + Fgann_min) / 2
        elif Pwfa / Pwft < 0.999:
            Fgann_min = Fgann
            Fgann = (Fgann_max + Fgann_min) / 2
        else:
            break

        if index > 20:
            break

    # check if plunger falls into the liquid column
    if Xp >= Ltb:
        mgtot += Fgtub * dt
        mgtb = (Xp - Ltb) / (H_well - Ltb) * mgtot
        mgtt = (H_well - Xp) / (H_well - Ltb) * mgtot

        if mgtb < 0:
            mgtb = 0

        mltt = 0
        Ltt = 0
        mltb = denl * A_tubing * Lt 
        Ltb = Lt
        mga += Fgann * dt
        Pwf = Pwfa

    else:
        mltot = denl * A_tubing * Lt 
        mltt = (Lt - Xp) / Lt * mltot
        mltb = Xp / Lt * mltot
        mgtb = 0

        if mltb < 0:
            mltb = 0

        mgtt += Fgtub * dt
        mga += Fgann * dt
        mla += Flann * dt
        Ltt = mltt / (denl * A_tubing)
        Ltb = mltb / (denl * A_tubing)
        Pwf = Pwfa

def bhp_balance_upward():
    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann

    index = 0

    Fgann_min = -mga / dt
    Fgann_max = Fgres
    Fgann = Fgann_max * 0.99

    Fltub = Flres
    Flann = 0

    # check if the liquid below plunger has filled the empty space
    if  (mltb + Fltub * dt) / (denl * A_tubing) > Xp:
        Ltb = Xp
        La = ((mltb + Fltub * dt) / (denl * A_tubing) - Xp) * A_tubing / A_casing
       
        Fgtub = 0
        Fgann = Fgres
        mltb = denl * Ltb * A_tubing
        mga += Fgann * dt
        mla = denl * La * A_casing 

        k = 0
        while True:
            k += 1
            Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
            Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
            Pc_n = mga * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
            Pcb_n = p_bottom(Pc, Tc_ave, H_well - La, Zc_ave)

            if abs(Pcb_n - Pcb) / abs(Pcb) > 1e-4 or abs(Pc - Pc_n) / abs(Pc) > 1e-4:
                if k > 50:
                    break
                Pc = (Pc_n + 4 * Pc) / 5
                Pcb = (Pcb_n + 4 * Pcb) / 5
            else:
                break

        Pwf = Pcb + denl * G * La
        Ppb = Pwf - denl * G * Ltb
        Ptb = Ppb

    # gas begins filling the space below plunger
    else:
        while True:
            index += 1

            Fltub, Fgtub = Flres - Flann, Fgres - Fgann
            Ltb = (mltb + Fltub * dt) / (denl * A_tubing)
            La = (mla + Flann * dt) / (denl * A_casing)

            if La < 0:
                Ltb += La * A_casing/ A_tubing
                mltb = denl * A_tubing * Ltb
                La = 0
                mla = 0

            i = 0
            while True:
                i += 1
                Ppb_ave, Tpb_ave = (Ptb + Ppb) / 2, (temperature(H_well - Xp) + temperature(H_well - Ltb)) / 2
                Zt_ave = z_factor_Starling(Tpb_ave, Ppb_ave, den_gas_relative)
                Ppb_n = (mgtb + Fgtub * dt) * Zt_ave * R * Tpb_ave / (A_tubing * (Xp - Ltb) * M_gas)
                Ptb_n = Ppb * exp(M_gas * G / (Zt_ave * R * Tpb_ave) * (Xp - Ltb))

                if abs(Ppb - Ppb_n) / abs(Ppb) > 1e-3 or abs(Ptb - Ptb_n) / abs(Ptb) > 1e-3:
                    if i >= 50:
                        if mgtb == 0:
                            Ptb = Pwf - denl * G * Ltb
                            Ppb = Ptb                        
                        break
                    Ppb = (4 * Ppb + Ppb_n) / 5 
                    Ptb = (4 * Ptb + Ptb_n) / 5 
                else:
                    break

            i = 0
            while True:
                i += 1        
                Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
                Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
                Pc_n = (mga + Fgann * dt) * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
                Pcb_n = Pc * exp(M_gas * G / (Zc_ave * R * Tc_ave) * (H_well - La))

                if abs(Pc_n - Pc) / abs(Pc) > 1e-3 or abs(Pcb_n - Pcb) / abs(Pcb) > 1e-3:
                    if i > 20:
                        break
                    Pc =  Pc_n
                    Pcb = Pcb_n
                else:
                    break

            Pwft = Ptb + denl * Ltb * G
            Pwfa = Pcb + denl * G * La

            if Pwfa / Pwft > 1.001:
                Fgann_max = Fgann
                Fgann = (Fgann_max + Fgann_min) / 2
            elif Pwfa / Pwft < 0.999:
                Fgann_min = Fgann
                Fgann = (Fgann_max + Fgann_min) / 2
            else:
                break

            if index > 50:
                # Fltub = Flres is incorrect
                if Pwft > Pwfa:
                    Fltub = (Pwft - Pwfa) / G * A_tubing / dt
                    Flann = Flres - Fltub
                break

        # update gas mass and BHP
        mga += Fgann * dt
        mla += Flann * dt
        mgtb += Fgtub * dt
        mltb += Fltub * dt
        Pwf = Pwfa

def build_up():
    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann

    index = 0
    Fgann_min = -mga / dt
    Fgann_max = Fgres
    Fgann = (Fgann_max + Fgann_min) / 2
    
    Flann = Flres
    Fltub = 0
    mla += Flann * dt
    La = mla / (denl * A_casing)

    if La < 0:
        Ltt += La * A_casing/ A_tubing
        mltt = denl * A_tubing * Ltt
        La = 0
        mla = 0

    while True:
        index += 1

        Fgtub = Fgres - Fgann

        i = 0
        while True:
            i += 1
            Pt_ave, Tt_ave = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well - Ltt - L_plunger)) / 2
            Zt_ave = z_factor_Starling(Tt_ave, Pt_ave, den_gas_relative)
            Pt_n = (mgtt + Fgtub * dt) * Zt_ave * R * Tt_ave / (A_tubing * (H_well - Ltt - L_plunger) * M_gas)
            Ptb_n = Pt * exp(M_gas * G / (Zt_ave * R * Tt_ave) * (H_well - Ltb - L_plunger))

            Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
            Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
            Pc_n = (mga + Fgann * dt) * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
            Pcb_n = Pc * exp(M_gas * G / (Zc_ave * R * Tc_ave) * (H_well - La))

            if abs(Pt_n - Pt) / Pt > 1e-3 or abs(Ptb - Ptb_n) / Ptb > 1e-3 or \
                abs(Pc_n - Pc) / abs(Pc) > 1e-3 or abs(Pcb_n - Pcb) / abs(Pcb) > 1e-3:
                if i > 50:
                    break
                Pt = (4*Pt + Pt_n) / 5
                Ptb = (4*Ptb + Ptb_n) / 5
                Pc = (4*Pc + Pc_n) / 5
                Pcb = (4*Pcb + Pcb_n) / 5
            else:
                break

        Pwft = Ptb + denl * G * (Ltt + L_plunger)
        Pwfa = Pcb + denl * G * La

        if Pwfa / Pwft > 1.001:
            Fgann_max = Fgann
            Fgann = (Fgann_max + Fgann_min) / 2
        elif Pwfa / Pwft < 0.999:
            Fgann_min = Fgann
            Fgann = (Fgann_max + Fgann_min) / 2
        else:
            break

        if index > 20:
            break

    mgtt += Fgtub * dt
    mga += Fgann * dt
    Pwf = Pwfa

def downward():

    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, dt, PwfB, FgresB, FlresB, HLB

    v = 0
    Fgout, Flout = 0, 0
    Fgres, Flres = unified_model()

    # plunger in gas column
    if Xp > Ltb + Ltt + abs(Vp * dt):
        Pt_ave, Tt_ave = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well - Ltb - Ltt)) / 2
        deng = gas_density(den_gas_relative, Tt_ave, Pt, fluid_type) #Pt_ave
        #print("plunger in gas column density gas density:", deng)
        Vp = -plunger_fall_velocity(A_plunger, A_tubing, Cd_plunger, m_plunger, deng)   # downward move
        Xp += Vp * dt
        bhp_balance_downward()

    # plunger in liquid column
    else:
        if Xp > Ltb + Ltt:
            Pt_ave, Tt_ave = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well - Ltb - Ltt)) / 2
            deng = gas_density(den_gas_relative, Tt_ave, Pt, fluid_type) #Pt_ave
            #print("plunger in liquid column gas density:",deng)
            Vp = -plunger_fall_velocity(A_plunger, A_tubing, Cd_plunger, m_plunger, deng)
            delta_t = (Xp - Ltb - Ltt) / abs(Vp)

            denm=((denl-deng)/(Ltb+Ltt))*Ltt+deng
            #print(denl,deng,Ltb,Ltt,Ltt+Ltb,Xp,denm)
            Vp = -plunger_fall_velocity(A_plunger, A_tubing, Cd_plunger, m_plunger, denm)   # plunger in liquid
            print(1,Vp)
            Xp = Ltt + Ltb - abs(Vp * (dt - delta_t))
            print(1,Xp)
            mgtt += mgtb
            mgtb -= mgtb
            bhp_balance_downward()

        elif (Xp + Vp * dt > 0) and (Xp < Ltb + Ltt):
            Pt_ave, Tt_ave = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well - Ltb - Ltt)) / 2
            deng = gas_density(den_gas_relative, Tt_ave, Pt, fluid_type)
            print("plunger in liquid column gas density:",deng)
            denm = ((denl - deng) / (Ltb + Ltt)) * Ltt + deng
            Vp = -plunger_fall_velocity(A_plunger, A_tubing, Cd_plunger, m_plunger, denm)
            print(2,Vp)
            Xp += Vp * dt
            print(2,Xp)
            mgtb = 0
            bhp_balance_downward()

        # plunger reaches bottom, buildup stage begins
        else:
            Vp, Xp = 0, 0
            Ppt, Ppb = Ptb, Pwf
            mltt += mltb
            mltb -= mltb
            Ltt = mltt / (denl * A_tubing)
            Ltb = mltb / (denl * A_tubing)
            build_up()

def fgout(deng, pt, pl):
    """
    calculate gas flow rate through choke valve based on A. Gupta, 2017
    :param deng: gas density (kg/m3)
    :param pt: tubing pressure at surface (pa)
    :param pl: surface line pressure (pa)
    :return: gas flow rate in kg/s
    """

    if pt >= 2*pl:
        return deng * Cv * pt
    elif (pt > pl) and (pt < 2*pl):
        return 2*deng*Cv*sqrt((pt-pl)*pl)
    else:
        return 0

def fgreservoir(pwf):
    """
    Rawlins and Schellhardt (1935) IPR for a gas well
    :param pwf: bottom hole pressure in pa
    :return: reservoir gas and liquid flow rates in kg/s
    """
    t_std = 273.15
    p_std = 1.013e5
    z = z_factor_Starling(t_std, p_std, den_gas_relative)
    #deng_std = gas_density(den_gas_relative, t_std, p_std, fluid_type)
    deng_std=3488 * p_std * den_gas_relative / z / t_std * 1e-6
    fg_res = deng_std*C_res*(P_res**2 - pwf**2)**n
    fl_res = fg_res/GLR
    return fg_res, fl_res

def gas_density_std(den_gas_relative, t, p):
    """
        Get real gas density in Kg/m3
        :param den_gas_relative: relative gas density to air density (-)
        :param t: temperature in K
        :param p: pressure in Pa
        :return: real gas density in Kg/m3
    """
    z = z_factor_Starling(t, p, den_gas_relative)
    return 3488 * p * den_gas_relative / z / t * 1e-6

def gas_density(den_gas_relative, t, p,fluid_type):
    """
    Get real gas density in Kg/m3
    :param den_gas_relative: relative gas density to air density (-)
    :param t: temperature in K
    :param p: pressure in Pa
    :param model: 1: black oil model, 2 compositional model
    :return: real gas density in Kg/m3
    """
    #z = z_factor_Starling(t, p, den_gas_relative)
    #return 3488 * p * den_gas_relative / z / t * 1e-6
    if fluid_type ==1:
        deng = gas_density_std(den_gas_relative, t, p)
    else:
        try:
            try:
                deng, denl, visg, visl, sigma = calPR(p,t,z_f,Pc_f,w_f,Tc_f,M_f)
            except :
                deng = gas_density_std(den_gas_relative, t, p)
        except:
            print('Gas_density model not converge')
            deng=10

    return deng

def gas_viscosity_stg(den_gas_relative, t, p):
    """
    Based on Lee & Wattenbarger correlation
    :param den_gas_relative: relative gas density to air density (-)
    :param t: temperature in K
    :param p: pressure in Pa
    :return: gas viscosity in Pa.s
    """
    deng = gas_density_std(den_gas_relative, t, p) / 1000   # g/cm3
    Mg = den_gas_relative * M_air * 1000                # g/mol

    x = 3.448 + 986.4 / (1.8 * t) + 0.01 * Mg
    y = 2.447 - 0.2224 * x
    k = (9.379 + 0.01607 * Mg) * (1.8 * t) ** 1.5 / (209.2 + 19.26 * Mg + 1.8 * t)
    gas_vis = k * 1e-4 * exp(x * deng ** y)          # cp
    return gas_vis / 1000                               # Pa.s

def gas_viscosity(den_gas_relative, t, p, fluid_type):
    """
    Based on Lee & Wattenbarger correlation
    :param den_gas_relative: relative gas density to air density (-)
    :param t: temperature in K
    :param p: pressure in Pa
    :param model: 1: black oil model, 2 compositional model
    :return: gas viscosity in Pa.s
    """
    if fluid_type ==1:
        visg=gas_viscosity_stg(den_gas_relative, t, p)
    else:
        try:
            deng, denl, visg, visl, sigma = calPR(p, t, z_f, Pc_f, w_f, Tc_f, M_f)
        except:
            # if not converge, use standard model (black oil)
            visg=gas_viscosity_stg(den_gas_relative, t, p)
    
    return visg

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

def liquid_viscosity(den_liquid_relative, t):
    """
    Based on Beggs and Robinson
    :param den_liquid_relative: specific gravity of oil
    :param t: temperature in K
    :return: viscosity in Pa.s
    """
    if abs(den_liquid_relative - 1.) <= 0.1:
        t = t-273.15
        return exp(1.003 - 1.479 * 1e-2 * (1.8 * t + 32) + 1.982 * 1e-5 * (1.8 * t + 32)**2) * 1e-3
    else:
        tf = (t - 273.15) * 1.8 + 32
        API = 141.5 / den_liquid_relative - 131.5
        x = 1.8653 - 0.025086*API - 0.5644*log10(tf)
        oil_vis = 10**(10**x) - 1
        return oil_vis/1000

def p_bottom(p_top, t, l, z):
    """
    :param p_top: pressure at top of gas column
    :param t: gas average temperature in K
    :param l: column length in m
    :param z: z factor
    :return: p_bottom, pressure at the gas column bottom
    """
    alpha = M_gas * G / (z * R * t)
    return p_top * exp(alpha * l)

def p_bottom_open(p_top, p, t, dt, l, fgout, z):
    """
    return bottom pressure for an open gas column
    :param p_top: pressure at top of gas column in pa
    :param p: gas average compressibility
    :param t: gas average temperature in K
    :param dt: tubing diameter in m
    :param l: column length in m
    :param fgout: gas flow rate at the top in kg/s
    :param z: z factor
    :return: p_bottom, pressure at column bottom
    """
    alpha = M_gas * G / (z * R * t)
    deng = gas_density(den_gas_relative, t, p, fluid_type)
    visg = gas_viscosity(den_gas_relative, t, p, fluid_type)
    ug = fgout/deng/(0.25*PI*dt**2)
    reg = abs(deng * ug * dt / visg)
    fg = get_fff_churchill(reg, ED, dt)
    b_square = 8*G / (PI**2 * alpha**2 * dt**5) * fg
    p_bottom = sqrt(p_top**2*exp(2*alpha*l) + b_square*(fgout/deng)**2*(exp(2*alpha*l)-1))
    return p_bottom

def plunger_acc_gas(Ppb, Ppt, Ltt, mltt, Xp, Vp):
    """
    plunger acceleration with gas production
    :param Ppb: pressure below plunger in Pa
    :param Ppt: pressure above plunger in Pa
    :param Ltt: liquid slug length in m
    :param dt: diameter of tubing in m
    :param At: cross section area in m2
    :param mltt: liquid mass on top of plunger in kg
    :param mp: plunge mass in kg
    :param Xp: plunger position
    :param Vp: plunger velocity in m/s
    :return: plunger acceleration in m/s2
    """
    Tl_ave = (temperature(H_well - Xp) + temperature(H_well-Xp-Ltt-L_plunger)) / 2
    visl = liquid_viscosity(den_liquid_relative, Tl_ave)
    rel = abs(denl * Vp * ID_tubing / visl)
    f = get_fff_churchill(rel, ED, ID_tubing)
    Pfric = 0.5 * denl * Vp ** 2 * f * ((Ltt + L_plunger) / ID_tubing)

    if Vp > 0:
        Acc = (Ppb - Ppt - Pfric) * A_tubing / (m_plunger + mltt) - G
    else:
        Acc = (Ppb - Ppt + Pfric) * A_tubing / (m_plunger + mltt) - G
    return Acc

def plunger_acc_liquid(Ppb, Ppt, Ltt, mltt, Xp, Vp):
    """
    plunger acceleration with liquid production
    :param Ppb: pressure below plunger in Pa
    :param Ppt: pressure above plunger in Pa
    :param Ltt: liquid slug length in m
    :param dt: diameter of tubing in m
    :param At: cross section area in m2
    :param mltt: liquid mass on top of plunger in kg
    :param mp: plunge mass in kg
    :param Xp: plunger position
    :param Vp: plunger velocity in m/s
    :return: plunger acceleration in m/s2
    """
    Tl_ave = (temperature(H_well - Xp) + temperature(H_well-Xp-Ltt-L_plunger)) / 2
    visl = liquid_viscosity(den_liquid_relative, Tl_ave)
    rel = abs(denl * Vp * ID_tubing / visl)
    f = get_fff_churchill(rel, ED, ID_tubing)
    Pfric = 0.5 * denl * Vp ** 2 * f * ((Ltt + L_plunger) / ID_tubing)
    Pacc = denl * Vp ** 2
    Acc = (Ppb - Ppt - Pfric - Pacc) * A_tubing / (m_plunger + mltt) - G
    return Acc

def plunger_fall_velocity(an, at, cd, mp, den):
    """
    plunger falling velocity in either gas column or liquid based on "orifice flow" model
    :param an: cross section area of tubing and plunger gap in m2
    :param at: cross section area of tubing in m2
    :param cd: drag coefficient of plunger
    :param mp: plunger weight in kg
    :param den: density in kg/m3
    :return: plunger falling velocity in m/s
    """
    return cd*(an/at)*sqrt(2*mp*G/at)/sqrt(den)

def print_io():
    
    print('t:', round(t, 2), '\n',
          'Xp:', round(Xp, 2),  '\t',   'Vp:',  round(Vp, 2),   '\t',   'Acc:', round(Acc, 2),  '\t',
          'Pc:', round(Pc, 2),  '\t',   'Pt:',  round(Pt, 2),   '\t',   'Pwf',  round(Pwf, 2),  '\t',
          'Ppt', round(Ppt,2),  '\t',  'Ptb',round(Ptb,2),  '\t', 
          'La:', round(La, 2),  '\t',   'Ltt:', round(Ltt, 2),  '\t',   'Ltb:', round(Ltb, 2),  '\t',
          'mga:', round(mga, 2), '\t',   'mgtt:', round(mgtt, 2), '\t',   'mgtb:', round(mgtb, 2), '\n')

def updateUI(time,dtime, XpUI, VpUI, Ltt0UI, Trigger):
    #Trigger: whether to update XpUI, VpUI, and Ltt0UI or not
    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, t, dt
    t=time
    dt=dtime
    if Trigger == True:
        Xp = XpUI
        Vp = VpUI
        Ltt0 = Ltt0UI


    print('t:', round(t, 2), '\n',
          'Xp:', round(Xp, 2),  '\t',   'Vp:',  round(Vp, 2),   '\t',   'Acc:', round(Acc, 2),  '\t',
          'Pc:', round(Pc, 2),  '\t',   'Pt:',  round(Pt, 2),   '\t',   'Pwf',  round(Pwf, 2),  '\t',
          'Ppt', round(Ppt,2),  '\t',  'Ptb',round(Ptb,2),  '\t', 
          'La:', round(La, 2),  '\t',   'Ltt:', round(Ltt, 2),  '\t',   'Ltb:', round(Ltb, 2),  '\t',
          'mga:', round(mga, 2), '\t',   'mgtt:', round(mgtt, 2), '\t',   'mgtb:', round(mgtb, 2), '\n')
    return Xp, Vp, Acc, Pc, Pt, Pwf, Ppt, Ptb, La, Ltt, Ltb, mga, mgtt, mgtb, Flout, Fgout

def updateUIinitial(LttUI):
    global mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, PwfB, Fgout, Flout, t, dt, PGT, \
            Ppb, Ppt, Acc, Fgres, Flres, FgresB, FlresB, Fgtub, Fltub, Fgann, Flann, Ltt0, HLB
    mga = 0.                    # mass of gas in annulus (kg)
    mla = 0.                    # mass of liquid in annulus (kg)
    mgtt = 0.                   # mass of gas in tubing top (kg)
    mltt = 0.                   # mass of liquid in tubing top (kg)
    mgtb = 0.                   # mass of gas in tubing bottom (kg)
    mltb = 0.                   # mass of liquid in tubing bottom (kg)
    Xp = 0.                     # plunger position from tubing shoe (m), initially at the bottom
    Vp = 0.                     # plunger velocity (m/s)
    Ar = 0.                     # plunger arrival time (s)
    v = 0                       # motor valve open or close: 0 - close, 1 - open
    # intermediate variables
    La = 0.                     # ASSUME initial liquid column length in annulus (m)
    Pcb = Pc                    # annular pressure right above liquid level
    Ptb = Pt                    # tubing gas pressure right above plunger
    Pwf = 0                     # bottom pressure at tubing shoe
    PwfB = 0.                   # bottom pressure at wellbore
    Fgout = 0.                  # surface gas mass flow rate (kg/s)
    Flout = 0.                  # surface liquid mass flow rate (kg/s)
    t = 0.                      # calculation time
    dt = 0.01                   # delta t in s
    PGT = 0.                    # total pressure gradient in horizontal well (pa/m)
    Ppb = 0.                    # pressure on the bottom of plunger
    Ppt = 0.                    # pressure on top of the plunger
    Acc = -1                    # plunger acceleration in m/s2
    Fgres = 0.                  # gas flow from horizontal well kg/s
    Flres = 0.                  # liquid flow from horizontal well kg/s
    FgresB = 0.                 # gas flow from reservoir kg/s
    FlresB = 0.                 # liquid flow from reservoir kg/s
    Fgtub = 0.                  # gas flow into tubing kg/s
    Fltub = 0.                  # liquid flow into tubing kg/s
    Fgann = 0.                  # gas flow into annulus kg/s
    Flann = 0.                  # liquid flow into annulus kg/s
    Ltt0 = LttUI                # initial liquid height in tubing for each plunger cycle
    HLB = 1                     # liquid holdup in the inclined or horizontal section of wellbore

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
        #print("surface tension(a):",A * (38.085 - 0.259 * API) / 1000 )
        return A * (38.085 - 0.259 * API) / 1000.
    else:
        # water-gas
        Kw = 4.5579 * 18 ** 0.15178 * (1.0)**(-0.84573)         # Watson characterization factor, °R1/3
        Tc = 24.2787 * Kw ** 1.76544 * (1.0)**2.12504           # Riazi, M.R. and Daubert, T.E. 1980.
        T = (t - 273.15) * 1.8 + 491.67                         # Celsius to Rankine
        Tr = T / Tc                                             # reduced temperature
        sigma = ((1.58*(den_liquid_relative - den_gas_relative * 1.25e-3) + 1.76)/Tr**0.3125)**4 / 1000
        #print("surface tension(b):",sigma)
        return 0.072

def temperature(x):
    """
    :param x: position in m
    :return: temperature in at x
    """
    return T_wellhead + Tgrd * x

def z_factor_Starling(t, p, den_gas_relative):
    """
    The Dranchuk and Abou-Kassem (1973) equation of state is based on the generalized Starling equation of state
    :param t: Temperature in K
    :param p: Pressure in Pa
    :param den_gas_relative: specific gravity of gas
    :return: compressibility factor of real gas
    """

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
                  + A10 * (1 + A11 * rour ** 2) * (rour ** 2 / Tr ** 3) * exp(-A11 * rour ** 2)

            if abs(z_n - z) / abs(z) > 1e-3:
                z = (z_n + 4 * z) / 5

                if i > 100:
                    break
            else:
                break
        
        if z < 0:
            z = 1.

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

# def initialize(inputs, dfcompositional):
def initialize(inputs):
    """
    initialize the parameters
    """
    # global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Ppb, Ppt, \
    #         PwfB, FgresB, FlresB, HLB, PGT
    global  m_plunger, L_plunger, d_plunger, Cd_plunger, Epsilon_plunger, A_plunger, H_well, ID_tubing, \
                OD_tubing, ID_casing, ED, A_tubing, A_casing, AB, Ann, LB, VB, ANG, DB, Cv, T_wellhead, Pl, \
                den_liquid_relative, den_gas_relative, vis_liquid, denl, M_gas, P_res, C_res, n, GLR, Rsl, \
                Tgrd, Pc, Pt, Ltt, Ltb, dt_H, dt_U, dt_D, cycles, Period, T_open, mga, mla, mgtt, mltt, mgtb, \
                mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Ppb, Ppt, PwfB, FgresB, FlresB, HLB, PGT

    # for i in range(dfcompositional.shape[0]):
    #     z_f[i]= dfcompositional.iloc[i,0]
    #     Pc_f[i]= dfcompositional.iloc[i,1]
    #     w_f[i] = dfcompositional.iloc[i,2]
    #     Tc_f[i]= dfcompositional.iloc[i,3]
    #     M_f[i]= dfcompositional.iloc[i,4] 
    fluid_type=inputs["Fluid_type"]

    """Upload global variables for configuration"""
    # plunger parameters
    m_plunger = inputs["Plunger_weight"]               # plunger mass in kg
    L_plunger = inputs["Plunger_length"]             # plunger length in mD
    d_plunger = inputs["Plunger_diameter"]    # plunger diameter in
    Cd_plunger = inputs["Plunger_drag_coefficient"]         # plunger fall drag coefficient
    Epsilon_plunger = inputs["Plunger_rise"]    # plunger rise 
    A_plunger = 0.25*PI*d_plunger**2   # plunger cross sectional area in m2

    # well parameters
    H_well = inputs["Well_depth"]     # well depth in m
    ID_tubing = inputs["Tubing_ID"]  # tubing ID in m
    OD_tubing = inputs["Tubing_OD"]  # tubing OD in m
    ID_casing = inputs["Casing_ID"]   # casing ID in m
    ED = inputs["Vertical_roughness"]                 # tubing roughness

    A_tubing = 0.25*PI*(ID_tubing**2)                   # tubing cross sectional area in m2     ########################3
    A_casing = 0.25*PI*(ID_casing**2 - OD_tubing**2)    # annulus cross section area in m2      ########################3
    AB = 0.25*PI*ID_casing**2                           # cross-sectional area of casing        ########################3
    Ann = 0.25*PI*(ID_tubing**2 - d_plunger**2)         # cross section area between plunger and tubing in m2       ########################3

    # wellbore section
    LB = inputs["Horizontal_length"]          # deviated section of the wellbore
    VB = AB * LB                # horizontal section volume in m3       ########################3
    ANG = inputs["Inclination_angle"]                   # inclination angle of the deviated section
    DB = inputs["Inner_diameter"]              # assume inner diameter the same as casing ID

    # surface parameters
    Cv = inputs["Valve_Cv"]                 # coefficient of motor valve
    T_wellhead = inputs["Surface_T"]            # temperature at well head in K
    Pl = inputs["Line_pressure"]                  # production line pressure - fixed if no surface line or separator considered

    # fluid properties
    den_liquid_relative = inputs["Relative_density_L"]  # relative density of liquid
    den_gas_relative = inputs["Relative_density_G"]      # relative density of gas
    vis_liquid = inputs["Liquid_viscosity"]           # liquid viscosity in Pa.s
    denl = den_liquid_relative * 1000
    M_gas = M_air * den_gas_relative

    # reservoir IPR
    P_res = inputs["Reservoir_P"]                 # reservoir pressure in Pa
    C_res = inputs["Reservoir_C"]            # reservoir coefficient for gas flow rate in Vogel model
    n = inputs["Reservoir_n"]                       # reservoir index for gas flow rate in Vogel model
    GLR = inputs["GLR"]                    # gas-liquid-ratio 
    Rsl = inputs["Gas_solubility"]                   # average gas solubility in liquid at average T and P, m3/m3
    Tgrd = inputs["Geothermal_gradient"]                 # formation temperature gradient 30K/km

    """Upload global variables for initial flow conditions"""

    Pc = inputs["Casing_pressure"]                  # casing pressure
    Pt = inputs["Tubing_pressure"]                 # tubing pressure
    Ltt = inputs["L_above_plunger"]                    # initial liquid column length above plunger (m)
    Ltb = inputs["L_below_plunger"]                    # initial liquid column length below plunger (m)
    dt_H= inputs["Time_step_horizontal"]                    # time step for horizontal section
    dt_U= inputs["Time_step_upward"]                     # time step for plunger upward section
    dt_D= inputs["Time_step_downward"]                   # time step for plunger downward section
    cycles= inputs["Plunger_cycle"]                   # Plunger lift cycles to be computed
    Period= inputs["Plunger_period"]*60.              # Plunger lift period (s to min)
    T_open= inputs["Valve_open_T"]*60.              # Surface valve open time (s to min)



    # Pwf
    while True:
        Pt_ave, Tt_ave = (Pt + Ptb) / 2, (T_wellhead + temperature(H_well-Ltt-L_plunger)) / 2
        Zt_ave = z_factor_Starling(Tt_ave, Pt_ave, den_gas_relative)
        Ptb_n = p_bottom(Pt, Tt_ave, H_well-Ltt-L_plunger, Zt_ave)
        if abs(Ptb_n - Ptb)/Ptb > 1e-4:
            Ptb = Ptb_n
        else:
            break

    Pwft = Ptb + denl * G * (Ltt + L_plunger)
    Vt = A_tubing * (H_well - Ltt - L_plunger)
    mgtt = Pt / (Zt_ave * R * Tt_ave / (Vt * M_gas))

    # mga
    La_max = Ltt
    La_min = 0
    La = (La_max + La_min) / 2

    i = 0
    while True:
        i += 1
        Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
        Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
        Pcb = p_bottom(Pc, Tc_ave, H_well-La, Zc_ave)
        Pwfa = Pcb + denl * G * La

        if Pwfa/Pwft > 1.00001:
            La_max = La
            La = (La_max + La_min) / 2
        elif Pwfa/Pwft < 0.99999:
            La_min = La
            La = (La_max + La_min) / 2
        else:
            break

        if i > 20:
            break

    Vc = A_casing * (H_well - La)
    mga = Pc / (Zc_ave * R * Tc_ave / (Vc * M_gas))

    # other parameters in vertical section
    Xp = 0.
    Vp = 0.
    Ltb = 0.
    mla = denl * La * A_casing
    mltt = denl * Ltt * A_tubing
    mltb = denl * Ltb * A_tubing
    Pwf = Pwfa
    Ppb = Pwf
    Ppt = Ptb

    # initialize horizontal section
    # guess a PGT
    PGT0 = 0
    TB_ave = (temperature(H_well) + temperature(H_well + LB * sin(ANG * PI / 180))) / 2

    i = 0
    while True:
        PwfB = Pwf + abs(PGT) * LB
        PB_ave = (Pwf + PwfB) / 2
        deng = gas_density(den_gas_relative, TB_ave, PB_ave, fluid_type)
        visg = gas_viscosity(den_gas_relative, TB_ave, PB_ave, fluid_type)
        visl = liquid_viscosity(den_liquid_relative, TB_ave)
        sigl = surface_tension(den_liquid_relative, den_gas_relative, TB_ave)

        FgresB, FlresB = fgreservoir(PwfB)
        Vsg, Vsl = FgresB/deng/AB, FlresB/denl/AB
        _, _, _, _, _, _, _, HLB, PGT, PGA, PGF, PGG, _, _, _, _, _, _, _, _, _, _, _ = \
            GAL(ID_casing, ED, ANG, Vsl, Vsg, denl, deng, visl, visg, sigl, Pwf, [1])

        if abs((PGT0 - PGT) / PGT) < 1e-3 or i > 20:
            break
        else:
            PGT0 = PGT
            i += 1


def unified_model():
    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
            Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, dt, PwfB, FgresB, FlresB, HLB, PGT

    # liquid hold up in the last time-step
    HLB0 = HLB

    # guess a PGT
    PGT_n= PGT
    TB_ave = (temperature(H_well) + temperature(H_well + LB * sin(ANG * PI / 180))) / 2

    i = 0
    while True:
        i += 1
        PwfB = Pwf + abs(PGT) * LB
        PB_ave = (Pwf + PwfB) / 2
        deng = gas_density(den_gas_relative, TB_ave, PB_ave, fluid_type)
        visg = gas_viscosity(den_gas_relative, TB_ave, PB_ave, fluid_type)
        visl = liquid_viscosity(den_liquid_relative, TB_ave)
        sigl = surface_tension(den_liquid_relative, den_gas_relative, TB_ave)

        FgresB, FlresB = fgreservoir(PwfB)
        Vsg, Vsl = FgresB / deng / AB, FlresB / denl / AB
        _, _, _, _, _, _, _, HLB, PGT, PGA, PGF, PGG, _, _, _, _, _, _, _, _, _, _, _ = \
            GAL(ID_casing, ED, ANG, Vsl, Vsg, denl, deng, visl, visg, sigl, PwfB, [1])

        if abs((PGT_n - PGT) / PGT) < 1e-4 or i > 20:
            break
        else:
            PGT_n = (PGT + 4 * PGT_n) / 5

    # liquid and gas flow rates at the upstream of horizontal section
    Flres = FlresB + denl * VB * (HLB0 - HLB) / dt
    Fgres = FgresB + deng * VB * (HLB - HLB0) / dt

    return Fgres, Flres


def upward():

    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, dt, PwfB, FgresB, FlresB, HLB

    v = 1   # valve opens
    Fgres, Flres = unified_model()
    # Fgres, Flres = fgreservoir(Pwf)
    Pt0, Ppt0 = Pt, Ppt

    # gas production from tubing section above plunger
    if Xp+Ltt0+L_plunger+Vp*dt < H_well:
        i = 0
        if mgtt - Fgout * dt > 0:
            while True:
                i += 1
                Fgout = fgout(gas_density(den_gas_relative, T_wellhead, Pt, fluid_type), Pt, Pl)
                Pt_ave, Tt_ave = (Pt + Ppt) / 2, (T_wellhead + temperature(H_well-Xp-Ltt-L_plunger)) / 2
                Zt_ave = z_factor_Starling(Tt_ave, Pt_ave, den_gas_relative)
                Pt_n = (mgtt - Fgout * dt) * Zt_ave * R * Tt_ave / (A_tubing * (H_well-Xp-Ltt-L_plunger) * M_gas)

                if Fgout != 0:
                    Ppt_n = p_bottom_open(Pt, Pt_ave, Tt_ave, ID_tubing, H_well-Xp-Ltt-L_plunger, Fgout, Zt_ave)
                else:
                    Ppt_n = p_bottom(Pt, Tt_ave, H_well-Xp-Ltt-L_plunger, Zt_ave)

                if abs(Pt - Pt_n) / abs(Pt) > 1e-4 or abs(Ppt_n - Ppt) / abs(Ppt) > 1e-4:
                    if i > 50:
                        break
                    Pt = Pt_n
                    Ppt = (Ppt_n + 4 * Ppt) / 5

                    if Pt < Pl:
                        upward_mgtt_depletion(Pt0)
                else:
                    break

            if mgtt - Fgout * dt > 0:
                mgtt -= Fgout * dt
            else:
                upward_mgtt_depletion(Pt0)
        else:
            upward_mgtt_depletion(Pt0)

        if Vp != 0:
            Acc = plunger_acc_gas(Ppb, Ppt, Ltt, mltt, Xp, Vp)
        else:
            Acc = (Ppb - Ppt) * A_tubing / (m_plunger + mltt) - G
            #if (Pc-Pt)/(Pc-Pl) > 0.5:
                #Acc=0

        # plunger stays at bottom, buildup continues
        if Acc < 0 and Xp == 0:  #and (Pc-Pt)/(Pc-Pl) > 0.5:
            Fgann = Fgres
            Flann = Flres
            mga += Fgann * dt
            mla += Flann * dt
            La = mla / (denl * A_casing)
            j = 0
            while True:
                j += 1
                Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
                Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
                Pc_n = mga * Zc_ave * R * Tc_ave / (M_gas * A_casing * (H_well - La))
                Pcb_n = p_bottom(Pc, Tc_ave, H_well - La, Zc_ave)

                if abs(Pcb_n - Pcb) / abs(Pcb) > 1e-4 or abs(Pc - Pc_n) / abs(Pc) > 1e-4:
                    Pc = Pc_n
                    Pcb = Pcb_n
                else:
                    break
                if j > 50:
                    # print("upward j", t, Xp, Vp, Acc)
                    break
            Pwf = Pcb + denl * G * La
            Ppb = Pwf

        # plunger starts moving up
        else:
            Vp += Acc * dt
            Xp += Vp * dt

            # depletes annulus liquid first
            if La > 0 and Fgtub == 0:
                if Vp > 0:
                    if Vp * dt * A_tubing <= La * A_casing:
                        Fgtub, Fltub = 0, denl * Vp * A_tubing
                        mltb += Fltub * dt
                        Flann = Flres - Fltub
                        Fgann = Fgres - Fgtub
                        mga += Fgann * dt
                        mla += Flann * dt
                        Ltb = mltb / (denl * A_tubing)
                        La = mla / (denl * A_casing)

                        k = 0
                        while True:
                            k += 1
                            Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
                            Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
                            Pc_n = mga * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
                            Pcb_n = p_bottom(Pc, Tc_ave, H_well - La, Zc_ave)

                            if abs(Pcb_n - Pcb) / abs(Pcb) > 1e-4 or abs(Pc - Pc_n) / abs(Pc) > 1e-4:
                                if k > 50:
                                    break
                                Pc = (Pc_n + 4 * Pc) / 5
                                Pcb = (Pcb_n + 4 * Pcb) / 5
                            else:
                                break

                        Pwf = Pcb + denl * G * La
                        Ppb = Pwf - denl * G * Ltb
                        Ptb = Ppb

                    else:
                        mltb += denl * abs(La) * A_casing
                        mla, La = 0, 0
                        bhp_balance_upward()
                
                # Vp < 0
                else:
                    if abs(Xp - Ltb) <= abs(1.001 * Vp * dt):
                        # plunger sinks in liquid column
                        Fgtub, Fltub = 0, 0
                        mltt += denl * abs(Vp) * A_tubing * dt
                        mltb -= denl * abs(Vp) * A_tubing * dt
                        Flann = Flres - Fltub
                        Fgann = Fgres - Fgtub
                        mga += Fgann * dt
                        mla += Flann * dt
                        La = mla / (denl * A_casing)
                        Ltb = mltb / (denl * A_tubing)
                        Ltt = mltt / (denl * A_tubing)

                        ii = 0
                        while True:
                            ii += 1
                            Pc_ave, Tc_ave = (Pc + Pcb) / 2, (T_wellhead + temperature(H_well - La)) / 2
                            Zc_ave = z_factor_Starling(Tc_ave, Pc_ave, den_gas_relative)
                            Pc_n = mga * Zc_ave * R * Tc_ave / (A_casing * (H_well - La) * M_gas)
                            Pcb_n = p_bottom(Pc, Tc_ave, H_well - La, Zc_ave)

                            if abs(Pcb_n - Pcb) / abs(Pcb) > 1e-3 or abs(Pc - Pc_n) / abs(Pc) > 1e-3:
                                if ii > 20:
                                    # print('upward ii', t, Xp, Vp, Acc)
                                    break
                                Pc, Pcb = Pc_n, Pcb_n
                            else:
                                break

                        Pwf = Pcb + denl * G * La
                        Ppb = Pwf - denl * G * Ltb
                        Ptb = Ppb

                    else:
                        # plunger sinks in gas column
                        deng = gas_density_std(den_gas_relative, temperature(H_well-Xp), Ppb)
                        mgtt += deng * abs(Vp) * dt * A_tubing
                        mgtb -= deng * abs(Vp) * dt * A_tubing
                        bhp_balance_upward()

            # initial annulus liquid depletion completes
            else:
                bhp_balance_upward()

    # liquid slug reaches the top
    else:
        # gas still exists on top of plunger
        if Xp + L_plunger + Ltt0 < H_well:
            delta_t = (H_well - Xp - L_plunger - Ltt) / Vp
            # Fgout = mgtt / delta_t
            mgtt = 0
            Flout = Vp * A_tubing * denl
            mltt -= (dt - delta_t) * Flout
            Ltt = mltt / (denl * A_tubing)
            Ppt = Pt
            Xp += Vp * dt

            Acc = plunger_acc_liquid(Ppb, Ppt, Ltt, mltt, Xp, Vp)

            Vp += Acc * (dt - delta_t)
            Pt = Pt0
            Ppt = Pt
            bhp_balance_upward()
        else:
            # plunger still moves up, liquid slug produces
            if (mltt > 0) and (Xp + L_plunger + Vp * dt < H_well):
                Pt = Pt0
                Ppt = Pt
                mgtt = 0
                Flout = Vp * A_tubing * denl
                mltt -= Flout * dt
                Xp += Vp * dt

                if mltt < 0:
                    Pt = Ppb
                    mltt = 0

                Ltt = mltt / (denl * A_tubing)

                # calculate plunger acceleration
                Acc = plunger_acc_liquid(Ppb, Ppt, Ltt, mltt, Xp, Vp)
                Vp += Acc * dt

                if Xp > H_well:
                    Pt = Ppb
                    Xp = H_well

                bhp_balance_upward()

            # plunger caught by plunger catcher, moving stops, after-flow begins
            else:
                if Xp + L_plunger < H_well:
                    delta_t = (H_well - Xp - L_plunger) / Vp
                    Flout = mltt / delta_t
                else:
                    Flout = 0

                mltt, mgtt = 0, 0
                Ltt = mltt / (denl * A_tubing)
                Xp = H_well
                Vp = 0
                Acc = 0
                after_flow()
                

def upward_mgtt_depletion(Pt0):
    global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
           Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, dt

    Fgout = 0
    Pt = Pt0
    i = 0
    while True:
        i += 1
        Pt_ave, Tt_ave = (Pt + Ppt) / 2, (T_wellhead + temperature(H_well-Xp-Ltt-L_plunger)) / 2
        Zt_ave = z_factor_Starling(Tt_ave, Pt_ave, den_gas_relative)
        Ppt_n = p_bottom(Pt, Tt_ave, H_well-Xp-Ltt-L_plunger, Zt_ave)
        if abs(Ppt_n - Ppt) / abs(Ppt) > 1e-3:
            if i > 50:
                break
            Ppt = (Ppt_n + 4*Ppt) / 5
        else:
            break


# Main function
def run(inputs = {"Well_name": 'Well_1', "Well_depth":10000 * 0.3048 , "Tubing_ID":1.995 * 0.0254 ,
        "Tubing_OD":2.125 * 0.0254, "Casing_ID":4.85 * 0.0254,"Vertical_roughness":2.5e-4, "Horizontal_length":1500 * 0.3048,
        "Inclination_angle":1, "Inner_diameter":4.85 * 0.0254, "Horizontal_roughness":2.5e-4, 
        "Plunger_weight":5, "Plunger_length":0.4, "Plunger_diameter":1.9 * 0.0254, 
        "Plunger_drag_coefficient":0.1019, "Plunger_rise":0.0457,
        "Line_pressure": 1.1e6, "Valve_Cv":0.5e-7, "Surface_T": 288,
        "Relative_density_L":0.85, "Relative_density_G":0.7, "Liquid_viscosity":4e-4,
        "Reservoir_C": 2.58e-15, "Reservoir_n":1, "Reservoir_P":6e6,
        "GLR": 10, "Geothermal_gradient": 0.03, "Gas_solubility":0.1,
        "Casing_pressure":1.6e6, "Tubing_pressure": 1.57e6, "L_above_plunger": 5.,
        "L_below_plunger": 0., "Time_step_horizontal":0.5, "Time_step_upward":5.,
        "Time_step_downward": 10., "Plunger_cycle": 6., "Plunger_period": 30,
        "Valve_open_T": 12., "Fluid_type": 1}):
#     global Pc, Pt, Pl, Ltt, Ltb, mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
#            Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, t, dt, Ltt0
    global  m_plunger, L_plunger, d_plunger, Cd_plunger, Epsilon_plunger, A_plunger, H_well, ID_tubing, \
                OD_tubing, ID_casing, ED, A_tubing, A_casing, AB, Ann, LB, VB, ANG, DB, Cv, T_wellhead, Pl, \
                den_liquid_relative, den_gas_relative, vis_liquid, denl, M_gas, P_res, C_res, n, GLR, Rsl, \
                Tgrd, Pc, Pt, Ltt, Ltb, dt_H, dt_U, dt_D, cycles, Period, T_open, \
                mga, mla, mgtt, mltt, mgtb, mltb, Xp, Vp, Ar, v, La, Pcb, Ptb, Pwf, Fgout, Ppb, Ppt, \
                Acc, Flout, Fgtub, Fltub, Fgres, Flres, Fgann, Flann, t, dt, Ltt0
                
    
    """Upload global variables for configuration"""
    # plunger parameters
    m_plunger = inputs["Plunger_weight"]               # plunger mass in kg
    L_plunger = inputs["Plunger_length"]             # plunger length in mD
    d_plunger = inputs["Plunger_diameter"]    # plunger diameter in
    Cd_plunger = inputs["Plunger_drag_coefficient"]         # plunger fall drag coefficient
    Epsilon_plunger = inputs["Plunger_rise"]    # plunger rise 
    A_plunger = 0.25*PI*d_plunger**2   # plunger cross sectional area in m2

    # well parameters
    H_well = inputs["Well_depth"]     # well depth in m
    ID_tubing = inputs["Tubing_ID"]  # tubing ID in m
    OD_tubing = inputs["Tubing_OD"]  # tubing OD in m
    ID_casing = inputs["Casing_ID"]   # casing ID in m
    ED = inputs["Vertical_roughness"]                 # tubing roughness

    A_tubing = 0.25*PI*(ID_tubing**2)                   # tubing cross sectional area in m2     ########################3
    A_casing = 0.25*PI*(ID_casing**2 - OD_tubing**2)    # annulus cross section area in m2      ########################3
    AB = 0.25*PI*ID_casing**2                           # cross-sectional area of casing        ########################3
    Ann = 0.25*PI*(ID_tubing**2 - d_plunger**2)         # cross section area between plunger and tubing in m2       ########################3

    # wellbore section
    LB = inputs["Horizontal_length"]          # deviated section of the wellbore
    VB = AB * LB                # horizontal section volume in m3       ########################3
    ANG = inputs["Inclination_angle"]                   # inclination angle of the deviated section
    DB = inputs["Inner_diameter"]              # assume inner diameter the same as casing ID

    # surface parameters
    Cv = inputs["Valve_Cv"]                 # coefficient of motor valve
    T_wellhead = inputs["Surface_T"]            # temperature at well head in K
    Pl = inputs["Line_pressure"]                  # production line pressure - fixed if no surface line or separator considered

    # fluid propertiei
    den_liquid_relativi = inputs["Relative_density_L"]  # relative density of liquid
    den_gas_relaiive = inputs["Relative_density_G"]      # relative density of gas
    vis_liquid = inputs["Liquid_viscosity"]           # liquid viscosity in Pa.s
    denl = den_liquid_relative * 1000
    M_gas = M_air * den_gas_relative

    # reserioir IPR
    P_res =inputs["Reservoir_P"]                 # reservoir pressure in Pa
    C_ris = inputs["Reservoir_C"]            # reservoir coefficient for gas flow rate in Vogel model
    n = inputs["Reservoir_n"]                       # reservoir index for gas flow rate in Vogel model
    GLR = inputs["GLR"]                    # gas-liquid-ratio 
    Rsl = inputs["Gas_solubility"]                   # average gas solubility in liquid at average T and P, m3/m3
    Tgrd = inputs["Geothermal_gradient"]                 # formation temperature gradient 30K/km

    """Upload global variables for initial flow conditions"""

    Pc = inputs["Casing_pressure"]                  # casing pressure
    Pt = inputs["Tubing_pressure"]                 # tubing pressure
    Ltt = inputs["L_above_plunger"]                    # initial liquid column length above plunger (m)
    Ltb = inputs["L_below_plunger"]                    # initial liquid column length below plunger (m)
    dt_H=inputs["Time_step_horizontal"]                    # time step for horizontal section
    dt_U=inputs["Time_step_upward"]                     # time step for plunger upward section
    dt_D=inputs["Time_step_downward"]                   # time step for plunger downward section
    cycles=inputs["Plunger_cycle"]                   # Plunger lift cycles to be computed
    Period=inputs["Plunger_period"]*60.              # Plunger lift period (s to min)
    T_open=inputs["Valve_open_T"]*60.              # Surface valve open time (s to min)
    
    initialize(inputs)
    # print(Xp, La)

    time = []
    XP = []
    VP = []
    ACC = []
    Fout = []
    PC = []
    PL = []
    PT = []
    PWF = []
    LT = []
    LA = []
    LTT = []
    LTB = []
    MGA = []
    MGTT = []
    MGTB = []

    counter = 0


    try:
        while counter < cycles:
            Xp = 0
            Vp = 0
            Ltt0 = Ltt
            print("{} cycle starts at {} s".format(counter+1, t), '\n')
            # print_io()

            while t < Period * (counter + 1):
                print_io()
                time.append(t)
                XP.append(Xp)
                LA.append(La)
                LT.append(Ltt + Ltb)
                VP.append(Vp)
                ACC.append(Acc)
                qprod = Fgout / gas_density_std(den_gas_relative, 273.15, 1e5) + Flout / denl
                Fout.append(1000 * qprod)
                PC.append(Pc / 1e6)
                PL.append(Pl / 1e6)
                PT.append(Pt / 1e6)
                PWF.append(Pwf / 1e6)
                LTT.append(Ltt)
                LTB.append(Ltb)
                MGA.append(mga)
                MGTT.append(mgtt)
                MGTB.append(mgtb)

                if t < T_open + Period * counter:
                    if Xp != H_well:
                        dt = dt_H
                    else:
                        dt = dt_U
                    upward()
                else:
                    dt = dt_D
                    downward()
                t += dt

            # next cycle
            counter += 1
            if Ppt>50000000.0:
                try:
                    pass
                except expression as identifier:
                    pass

    except:
        
        data = pd.DataFrame({"time": time,  "Xp": XP,       "Acc": ACC, "Vp": VP,
                            "Pc": PC,      "Pt": PT,       "Pwf": PWF, "PL": PL,
                            "La": LA,      "Ltt": LTT,     "Ltb": LTB, "Fout": Fout,
                            "mga": MGA,    "mgtt": MGTT,   "mgtb": MGTB})
        data.to_excel("output.xlsx")

        # time_acc = []
        # ACC_upward = []
        # t = 0
        # while t < 65:
        #     dt = 0.1
        #     time_acc.append(t)
        #     ACC_upward.append(Acc)
        #     t += dt
        #     upward()

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(time, ACC, label="Plunger acceleration")
        ax1.set_xlabel(r'$t$ (s)')
        ax1.set_ylabel(r'$a_{p}$ $(m/s^2)$')
        ax1.set_ylim(-1)
        ax1.legend(frameon=False)
        fig1.show()


        # fig11 = plt.figure()
        # ax11 = fig11.add_subplot(111)
        # ax11.plot(time_acc, ACC_upward)
        # ax11.set_xlabel(r'$t$ (s)')
        # ax11.set_ylabel(r'$a_{p}$ $(m/s^2)$')
        # fig11.show()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(time, VP, label="Vp")
        ax2.set_xlabel(r'$t$ (s)')
        ax2.set_ylabel(r'$V_{p}$ (m/s)')
        ax2.legend(frameon=False)
        fig2.show()


        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(time, Fout, label="Production rate")
        ax3.set_xlabel(r'$t$ (s)')
        ax3.set_ylabel(r'$Q_{prod}$ $dm^3/s$')
        ax3.legend(frameon=False)
        fig3.show()


        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        ax4.plot(time, PC, label="Pc")
        ax4.plot(time, PT, label="Pt")
        ax4.plot(time, PWF, label="Pwf")
        ax4.plot(time, PL, label="Pl")
        ax4.legend(frameon=False)
        ax4.set_xlabel(r'$t$ (s)')
        ax4.set_ylabel(r'$P$ (MPa)')
        fig4.show()


        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        ax5.plot(time, LT, label="Tubing")
        ax5.plot(time, LA, label="Annulus")
        ax5.set_xlabel(r'$t$ (s)')
        ax5.set_ylabel(r'$H$ (m)')
        ax5.legend(frameon=False)
        fig5.show()
        plt.show()

    
    #Final result
    data = pd.DataFrame({"time": time,  "Xp": XP,       "Acc": ACC, "Vp": VP,
                         "Pc": PC,      "Pt": PT,       "Pwf": PWF, "PL": PL,
                         "La": LA,      "Ltt": LTT,     "Ltb": LTB, "Fout": Fout,
                         "mga": MGA,    "mgtt": MGTT,   "mgtb": MGTB})
    
    # try:
        # data.to_excel('/Users/haiwenzhu/Desktop/work/104 Plunger_Lift_Project/Plunger lift GUI/output/output.xlsx')
        # print('Output excel successfully')
        # data.to_excel("output.xlsx")
    # except:
        # exit()
    

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(time, ACC, label="Plunger acceleration")
    ax1.set_xlabel(r'$t$ (s)')
    ax1.set_ylabel(r'$a_{p}$ $(m/s^2)$')
    ax1.set_ylim(-1)
    ax1.legend(frameon=False)
    # fig1.show()
    # fig1.savefig('/Users/haiwenzhu/Desktop/work/104 Plunger_Lift_Project/Plunger lift GUI/output/Plunger acceleration.jpg')
    # print('Output Plunger acceleration figure successfully')


    # fig11 = plt.figure()
    # ax11 = fig11.add_subplot(111)
    # ax11.plot(time_acc, ACC_upward)
    # ax11.set_xlabel(r'$t$ (s)')
    # ax11.set_ylabel(r'$a_{p}$ $(m/s^2)$')
    # fig11.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(time, VP, label="Vp")
    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel(r'$V_{p}$ (m/s)')
    ax2.legend(frameon=False)
    # fig2.show()
    # fig2.savefig('/Users/haiwenzhu/Desktop/work/104 Plunger_Lift_Project/Plunger lift GUI/output/Vp.jpg')
    # print('Output Vp figure successfully')


    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(time, Fout, label="Production rate")
    ax3.set_xlabel(r'$t$ (s)')
    ax3.set_ylabel(r'$Q_{prod}$ $dm^3/s$')
    ax3.legend(frameon=False)
    # fig3.show()
    # fig3.savefig('/Users/haiwenzhu/Desktop/work/104 Plunger_Lift_Project/Plunger lift GUI/output/Production rate.jpg')
    # print('Output Production rate figure successfully')


    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(time, PC, label="Pc")
    ax4.plot(time, PT, label="Pt")
    ax4.plot(time, PWF, label="Pwf")
    ax4.plot(time, PL, label="Pl")
    ax4.legend(frameon=False)
    ax4.set_xlabel(r'$t$ (s)')
    ax4.set_ylabel(r'$P$ (MPa)')
    # fig4.show()
    # fig4.savefig('/Users/haiwenzhu/Desktop/work/104 Plunger_Lift_Project/Plunger lift GUI/output/Pressure.jpg')
    # print('Output Pressure figure successfully')

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(time, LT, label="Tubing")
    ax5.plot(time, LA, label="Annulus")
    ax5.set_xlabel(r'$t$ (s)')
    ax5.set_ylabel(r'$H$ (m)')
    ax5.legend(frameon=False)
    # fig5.show()
    # fig5.savefig('/Users/haiwenzhu/Desktop/work/104 Plunger_Lift_Project/Plunger lift GUI/output/Position.jpg')
    # print('Output Position figure successfully')




if __name__ == "__main__":


    starttime=time.time()

    #example
    inputs = {"Well_name": 'Well_1', "Well_depth":10000 * 0.3048 , "Tubing_ID":1.995 * 0.0254 ,
        "Tubing_OD":2.125 * 0.0254, "Casing_ID":4.85 * 0.0254,"Vertical_roughness":2.5e-4, "Horizontal_length":1500 * 0.3048,
        "Inclination_angle":1, "Inner_diameter":4.85 * 0.0254, "Horizontal_roughness":2.5e-4, 
        "Plunger_weight":5, "Plunger_length":0.4, "Plunger_diameter":1.9 * 0.0254, 
        "Plunger_drag_coefficient":0.1019, "Plunger_rise":0.0457,
        "Line_pressure": 1.1e6, "Valve_Cv":0.5e-7, "Surface_T": 288,
        "Relative_density_L":0.85, "Relative_density_G":0.7, "Liquid_viscosity":4e-4,
        "Reservoir_C": 2.58e-15, "Reservoir_n":1, "Reservoir_P":6e6,
        "GLR": 10, "Geothermal_gradient": 0.03, "Gas_solubility":0.1,
        "Casing_pressure":1.6e6, "Tubing_pressure": 1.57e6, "L_above_plunger": 5.,
        "L_below_plunger": 0., "Time_step_horizontal":0.5, "Time_step_upward":5.,
        "Time_step_downward": 10., "Plunger_cycle": 1., "Plunger_period": 30,
        "Valve_open_T": 12., "Fluid_type": 1}
    
    run(inputs)
    
    endtime=time.time()
    print('complete, time cost: ', starttime-endtime)
    plt.show()

