import numpy as np
import pydicom
import os
import os.path

def read_dicoms(path):
    kv = []
    mA = []
    ms = []
    μAs = []
    pw = []
    wc = []
    ww = []
    im = []
    gamma = []
    for basepath, _, files in os.walk(path):
        for filename in files:
            dicom = pydicom.dcmread(os.path.join(basepath, filename))
            kv.append(float(dicom.KVP))
            mA.append(float(dicom.XRayTubeCurrent))
            ms.append(float(dicom.ExposureTime))
            μAs.append(float(dicom[0x0021,0x1004].value))
            pw.append(float(dicom.AveragePulseWidth))
            wc.append(float(dicom.WindowCenter))
            ww.append(float(dicom.WindowWidth))
            im.append(dicom.pixel_array)
            gamma.append(np.array(dicom[0x0021,0x1028][0][0x0021,0x1042].value))
    
    kv = np.array(kv)
    mA = np.array(mA)
    ms = np.array(ms)
    μAs = np.array(μAs)
    pw = np.array(pw)
    wc = np.array(wc)
    ww = np.array(ww)
    im = np.array(im)
    gamma = np.array(gamma)

    return {"kv": kv, "mA": mA, "ms": ms, "μAs": μAs, "pw": pw, "wc": wc, "ww": ww, "im": im, "gamma": gamma}

def inverse_lut(lut):
    ilut = np.zeros_like(lut)
    for i in range(len(lut)):
        pos = np.argmin(np.abs(lut-i))
        if np.isscalar(pos):
            ilut[i] = pos
        else:
            ilut[i] = pos[0]
    return ilut

def plot_luts(luts):
    import matplotlib.pyplot as plt
    for lut in luts:
        plt.plot(lut, range(len(lut)))
    plt.figure()
    for lut in luts:
        plt.plot(inverse_lut(lut), range(len(lut)))
    plt.show()

def parse_stpar(data):
    sh_stpar_data_format = '17h6c9h2c7h6c7h4ch4c4h4c17h51i54fh6c'

def calc_i0(s):
    im_norm = s["im"]
    print("invert gamma luts")
    igamma = np.array([inverse_lut(gamma) for gamma in s["gamma"]])
    print("apply inverted gamma lut")
    im_norm = np.array([igamma[im_norm] for igamma, im_norm in zip(igamma, im_norm)])
    print("averaging")
    im_norm = np.array([np.mean(i) for i in im_norm])
    print("reverse windowing")
    im_norm = im_norm*(s["ww"]/4096) + (s["wc"]-s["ww"]/2)
    s["im_norm"] = im_norm
    return s

def fit_i0(s):
    a, b = np.polyfit(s["μAs"]*0.001, s["im_norm"], 1)
    a1, b1 = np.polyfit(s["mA"]*s["pw"]*0.001, s["im_norm"], 1)
    avg = np.array([np.mean(i) for i in s["im"]])
    a2, b2 = np.polyfit(s["μAs"]*0.001, avg,  1)
    a3, b3 = np.polyfit(s["mA"]*s["pw"]*0.001, avg, 1)
    return (a,b), (a1,b1), (a2,b2), (a3,b3)

if __name__ == "__main__":
    s = read_dicoms(r"D:\lumbal_spine_13.10.2020\output\70kVp")
    
