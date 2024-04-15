import json
import os

import pandas as pd
import numpy as np

elem_df = pd.read_excel(os.path.join(os.getcwd(), "atomic_properties.xlsx"), sheet_name=0, header=0, index_col=1)
atom_init = {}

for _, prop in elem_df.iterrows():

    an = prop.loc['Z']
    p = np.zeros(90)

    # Set electornegativity feature [10 elements - tot feats 10]
    if prop['Electronegativity'] < 0.85:
        p[0] = 1
    elif 0.85 <= prop['Electronegativity'] < 1.2:
        p[1] = 1
    elif 1.2 <= prop['Electronegativity'] < 1.55:
        p[2] = 1
    elif 1.55 <= prop['Electronegativity'] < 1.9:
        p[3] = 1
    elif 1.9 <= prop['Electronegativity'] < 2.25:
        p[4] = 1
    elif 2.25 <= prop['Electronegativity'] < 2.6:
        p[5] = 1
    elif 2.6 <= prop['Electronegativity'] < 2.95:
        p[6] = 1
    elif 2.95 <= prop['Electronegativity'] < 3.3:
        p[7] = 1
    elif 3.3 <= prop['Electronegativity'] < 3.65:
        p[8] = 1
    elif 3.65 <= prop['Electronegativity']:
        p[9] = 1

    # Set First Ionization Energy feature [10 elements - tot feats 20]
    if prop['FIE Log scale'] < 1.5:
        p[10] = 1
    elif 1.5 <= prop['FIE Log scale'] < 1.7:
        p[11] = 1
    elif 1.7 <= prop['FIE Log scale'] < 1.9:
        p[12] = 1
    elif 1.9 <= prop['FIE Log scale'] < 2.1:
        p[13] = 1
    elif 2.1 <= prop['FIE Log scale'] < 2.3:
        p[14] = 1
    elif 2.3 <= prop['FIE Log scale'] < 2.5:
        p[15] = 1
    elif 2.5 <= prop['FIE Log scale'] < 2.7:
        p[16] = 1
    elif 2.7 <= prop['FIE Log scale'] < 2.9:
        p[17] = 1
    elif 2.9 <= prop['FIE Log scale'] < 3.1:
        p[18] = 1
    elif 3.1 <= prop['FIE Log scale']:
        p[19] = 1

    # Set Covalent Radius feature [10 elements - tot feats 30]
    if prop['Covalent Radius'] < 47.5:
        p[20] = 1
    elif 47.5 <= prop['Covalent Radius'] < 70:
        p[21] = 1
    elif 70 <= prop['Covalent Radius'] < 92.5:
        p[22] = 1
    elif 92.5 <= prop['Covalent Radius'] < 115:
        p[23] = 1
    elif 115 <= prop['Covalent Radius'] < 137.5:
        p[24] = 1
    elif 137.5 <= prop['Covalent Radius'] < 160:
        p[25] = 1
    elif 160 <= prop['Covalent Radius'] < 182.5:
        p[26] = 1
    elif 182.5 <= prop['Covalent Radius'] < 205:
        p[27] = 1
    elif 205 <= prop['Covalent Radius'] < 227.5:
        p[28] = 1
    elif 227.5 <= prop['Covalent Radius']:
        p[29] = 1

    # Set Valence Electrons feature [9 elements - tot feats 39]
    if prop['Valence Electrons'] == 18:
        p[38] = 1
    else:
        fea_idx = 29 + prop['Valence Electrons']
        p[int(fea_idx)] = 1

    # Set Group Number Feature [18 elements - tot feats 57]
    p[int(38 + prop['Group Number'])] = 1

    # Set Electron Affinity feature [10 elements - tot feats 67]
    if prop['Electron Affinity'] < -2.33:
        p[57] = 1
    elif -2.33 <= prop['Electron Affinity'] < -1.66:
        p[58] = 1
    elif -1.66 <= prop['Electron Affinity'] < -0.99:
        p[59] = 1
    elif -0.99 <= prop['Electron Affinity'] < -0.32:
        p[60] = 1
    elif -0.32 <= prop['Electron Affinity'] < 0.35:
        p[61] = 1
    elif 0.35 <= prop['Electron Affinity'] < 1.02:
        p[62] = 1
    elif 1.02 <= prop['Electron Affinity'] < 1.69:
        p[63] = 1
    elif 1.69 <= prop['Electron Affinity'] < 2.36:
        p[64] = 1
    elif 2.36 <= prop['Electron Affinity'] < 3.03:
        p[65] = 1
    elif 3.03 <= prop['Electron Affinity']:
        p[66] = 1

    # Set Period Number feature [9 elements - tot feats 76]
    p[int(66 + prop['Period Number'])] = 1

    # Set Block feature [4 elements - tot feats 80]
    p[int(75 + prop['Block'])] = 1

    # Set Atomic Volume feature [10 elements - tot feats 90]
    if prop['AV Log Scale'] < 1.78:
        p[80] = 1
    elif 1.78 <= prop['AV Log Scale'] < 2.06:
        p[81] = 1
    elif 2.06 <= prop['AV Log Scale'] < 2.34:
        p[82] = 1
    elif 2.34 <= prop['AV Log Scale'] < 2.62:
        p[83] = 1
    elif 2.62 <= prop['AV Log Scale'] < 2.9:
        p[84] = 1
    elif 2.9 <= prop['AV Log Scale'] < 3.18:
        p[85] = 1
    elif 3.18 <= prop['AV Log Scale'] < 3.46:
        p[86] = 1
    elif 3.46 <= prop['AV Log Scale'] < 3.74:
        p[87] = 1
    elif 3.74 <= prop['AV Log Scale'] < 4.02:
        p[88] = 1
    elif 4.02 <= prop['AV Log Scale']:
        p[89] = 1

    atom_init[str(an)] = list(p)

with open('atom_init.json', 'w') as f:
    json.dump(atom_init, f)


    




