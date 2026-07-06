# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:16:20 2026

@author: kirbyn
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interpn,RegularGridInterpolator
import pandas as pd

#this file combines CT, dose, and structure information read from DICOM files
#######CT_path = '1slice/CTslice.dcm'
# struc_path = 'ray electron/struc.dcm'
dose_path = ("P:\\Share\\Parker\\Octavius 1500 w Brett\\RS_exports_061726\\trial.dcm")
# plan_path = 'ray electron/plan.dcm'
#read CT data and dose
#######CT = pydicom.read_file(CT_path)
Dose = pydicom.dcmread(dose_path)
#struc = pydicom.read_file(struc_path)
#plan = pydicom.read_file(plan_path)


#######Image = CT.pixel_array
dose = Dose.pixel_array*Dose.DoseGridScaling

#get zero points for coordinate systems
#######Cp0 = CT.ImagePositionPatient
Dp0 = Dose.ImagePositionPatient
#get the voxel dimensions
#######Csp = CT.PixelSpacing
Dsp = Dose.PixelSpacing
#######Cth = CT.SliceThickness
Dth = Dose.SliceThickness
#x, y, z position for CT
#######xC = Cp0[0] +Csp[0]*np.arange(Image.shape[1])
#######yC = Cp0[1] +Csp[1]*np.arange(Image.shape[0])
#######zC = Cp0[2] +Cth*np.arange(1)
#x, y, z position for Dose
xD = Dp0[0] +Dsp[0]*np.arange(dose.shape[2])
yD = Dp0[1] +Dsp[1]*np.arange(dose.shape[1])
zD = Dp0[2] +Dth*np.arange(dose.shape[0])
#interpolate the dose at each CT position



#assumes isocenter position of
#R-L 0.06 cm
#I-S 0 cm
#P-A 37.71 cm
# print(struc.StructureSetROISequence[3])
# iso = struc.ROIContourSequence[3].ContourSequence[0].ContourData

xiso = 0.6 #np.float64(iso[0])
yiso = -377.1# np.float64(iso[1])
ziso = 0.#np.float64(iso[2])

N = 2001
depth = 12
xpos = np.linspace(xiso-100,xiso+100,N)
ypos = np.linspace(yiso + depth,yiso+ depth,N)
zpos = np.linspace(ziso,ziso,N)

# [XCT,YCT,ZCT] = np.meshgrid(xC,yC,zC,indexing='ij')
fn = RegularGridInterpolator((zD,yD,xD), dose, method='linear',bounds_error=False,fill_value=0)
#coordinate_grid = np.stack([xpos,ypos,zpos],axis=3)
dose_int = fn((zpos,ypos,xpos))

xvect = xpos - xiso
plt.plot(xvect, dose_int)
plt.xlabel('X [mm]')
plt.ylabel('Dose [Gy]')






# Create a sample DataFrame
data = {
    "X [mm]": xvect,
    "Dose [Gy]": dose_int
}
df = pd.DataFrame(data)

# Save to CSV without the default numeric index column
df.to_csv("output.csv", index=False)
