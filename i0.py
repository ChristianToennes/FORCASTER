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
    wv = []
    dose = []
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
            dose.append(float(dicom[0x0021,0x1005].value))
            wv.append(float(dicom[0x0021,0x1049].value))
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
    dose = np.array(dose)
    wv = np.array(wv)
    gamma = np.array(gamma)

    return {"kv": kv, "mA": mA, "ms": ms, "μAs": μAs, "pw": pw, "wc": wc, "ww": ww, "im": im, "gamma": gamma, "wv": wv, "dose": dose}

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

def norm_images(s):
    im_norm = s["im"]
    print("invert gamma luts")
    igamma = np.array([inverse_lut(gamma) for gamma in s["gamma"]])
    print("apply inverted gamma lut")
    im_norm = np.array([igamma[im_norm] for igamma, im_norm in zip(igamma, im_norm)])
    print("averaging")
    im_norm = np.array([np.mean(i) for i in im_norm])
    #print("reverse windowing")
    #im_norm_windowed = im_norm*(s["ww"]/4096) + (s["wc"]-s["ww"]/2)
    #im_windowed = s["im"]*(s["ww"]/4096) + (s["wc"]-s["ww"]/2)
    s["im_norm"] = im_norm
    s["im_norm_windowed"] = s["wv"]
    s["im_windowed"] = s["dose"]
    return s

def fit_i0(s):
    print("fit functions")
    f1 = np.polyfit(s["μAs"]*0.001, s["im_norm"], 2)
    f2 = np.polyfit(s["μAs"]*0.001, s["im_norm_windowed"], 2)
    avg = np.array([np.mean(i) for i in s["im"]])
    avg_windowed = np.array([np.mean(i) for i in s["im_windowed"]])
    f3 = np.polyfit(s["μAs"]*0.001, s["im_windowed"],  2)
    f4 = np.polyfit(s["μAs"]*0.001, avg, 2)
    return f1, f2, f3, f4

def get_i0(input_path = r".\output\70kVp"):
    s = read_dicoms(input_path)
    avg = np.array([np.mean(i[:,100:-100,100:-100]) for i in s["im"]])
    image_count = np.array([len(i) for i in s["im"]])
    f = np.polyfit(s["μAs"]*0.001/image_count, avg, 2)
    #f = np.polyfit(s["mA"]*s["pw"]*0.001, avg, 2)
    return f, s["gamma"][0]


if __name__ == "__main__":
    s = read_dicoms(r".\output\70kVp")
    
    s = norm_images(s)
    f1, f2, f3, f4 = fit_i0(s)
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(s["μAs"]*0.001), np.max(s["μAs"]*0.001), len(s["μAs"]))
    y = s["μAs"]*0.001
    ax = plt.subplot(222)
    plt.title("gamma correction")
    plt.plot(x, np.polyval(f1, x), "r", label="{}".format(f1))
    plt.scatter(y, s["im_norm"], c="g", label="")
    plt.legend()
    ax = plt.subplot(224)
    plt.title("gamma correction + windowing reversed")
    plt.plot(x, np.polyval(f2, x), "r", label="{}".format(f2))
    plt.scatter(y, s["im_norm_windowed"], c="g", label="")
    plt.legend()
    ax = plt.subplot(223)
    plt.title("windowing reversed")
    plt.plot(x, np.polyval(f3, x), "r", label="{}".format(f3))
    avg = np.array([np.mean(i) for i in s["im_windowed"]])
    plt.scatter(y, s["im_windowed"], c="g", label="")
    plt.legend()
    ax = plt.subplot(221)
    plt.title("no corrections")
    plt.plot(x, np.polyval(f4, x), "r", label="{}".format(f4))
    avg = np.array([np.mean(i) for i in s["im"]])
    plt.scatter(y, avg, c="g", label="")
    plt.legend()
    plt.show()
