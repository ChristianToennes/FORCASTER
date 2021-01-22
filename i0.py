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
    gamma_in = []
    gamma_out = []
    percent_gain = []
    for basepath, _, files in os.walk(path):
        for filename in files:
            dicom = pydicom.dcmread(os.path.join(basepath, filename))
            if "KVP" not in dicom or dicom.KVP is None or abs(float(dicom.KVP)-70) > 1 or np.mean(dicom.pixel_array)>4000: continue
            #print(os.path.join(basepath, filename))
            try:

                im.append(dicom.pixel_array)
                kv.append(float(dicom.KVP))
                mA.append(float(dicom.XRayTubeCurrent))
                ms.append(float(dicom.ExposureTime))
                if dicom[0x0021,0x1004].VR == "SL":
                    μAs.append(float(dicom[0x0021,0x1004].value))
                else:
                    μAs.append(float(int.from_bytes(dicom[0x0021,0x1004].value, "little", signed=True)))
                pw.append(float(dicom.AveragePulseWidth))
                wc.append(float(dicom.WindowCenter))
                ww.append(float(dicom.WindowWidth))
                if dicom[0x0021,0x1005].VR == "SL":
                    dose.append(float(dicom[0x0021,0x1005].value))
                else:
                    dose.append(float(int.from_bytes(dicom[0x0021,0x1005].value, "little", signed=True)))
                if dicom[0x0021,0x1049].VR == "SS":
                    wv.append(float(dicom[0x0021,0x1049].value))
                else:
                    wv.append(float(int.from_bytes(dicom[0x0021,0x1049].value, "little", signed=True)))
                if dicom[0x0021,0x1028].VR != "SQ":
                    dicom[0x0021,0x1028].value = pydicom.values.convert_SQ(dicom[0x0021,0x1028].value, True, True)
                if dicom[0x0021,0x1028][0][0x0021,0x1042].VR == "US":
                    gamma.append(np.array(dicom[0x0021,0x1028][0][0x0021,0x1042].value))
                else:
                    gamma.append(np.array(pydicom.values.convert_numbers(dicom[0x0021,0x1028][0][0x0021,0x1042].value, True, "H")))
                if dicom[0x0021,0x1010].VR == "US":
                    gamma_in.append(float(dicom[0x021,0x1010].value))
                else:
                    gamma_in.append(float(int.from_bytes(dicom[0x021,0x1010].value, "little", signed=False)))
                if dicom[0x0021,0x1011].VR == "US":
                    gamma_out.append(float(dicom[0x021,0x1011].value))
                else:
                    gamma_out.append(float(int.from_bytes(dicom[0x021,0x1011].value, "little", signed=False)))
                if dicom[0x0019,0x1008].VR == "US":
                    percent_gain.append(dicom[0x0019,0x1008].value)
                else:
                    percent_gain.append(float(int.from_bytes(dicom[0x0019,0x1008].value, "little", signed=False)))
                #print(kv[-1], mA[-1], ms[-1], μAs[-1], pw[-1], percent_gain[-1])
            except Exception as e:
                print(os.path.join(basepath, filename))
                raise
    
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
    gamma_in = np.array(gamma_in)
    gamma_out = np.array(gamma_out)
    percent_gain = np.array(percent_gain)

    return {"kv": kv, "mA": mA, "ms": ms, "μAs": μAs, "pw": pw, "wc": wc, "ww": ww, "im": im, "gamma": gamma, "gamma_in":gamma_in, "gamma_out":gamma_out, "gain": percent_gain, "wv": wv, "dose": dose}

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
    s["im_norm_windowed"] = np.array([np.mean(i*gain) for i,gain in zip(s["im"], s["gain"])])
    s["im_windowed"] = np.array([np.mean(i*(1+gain/100)) for i,gain in zip(s["im"], s["gain"])])
    return s

def fit_i0(s):
    print("fit functions")
    f1 = np.polyfit(s["μAs"]*0.001, s["im_norm"], 1)
    f2 = np.polyfit(s["μAs"]*0.001, s["im_norm_windowed"], 1)
    avg = np.array([np.mean(i) for i in s["im"]])
    avg_windowed = np.array([np.mean(i) for i in s["im_windowed"]])
    f3 = np.polyfit(s["μAs"]*0.001, s["im_windowed"],  1)
    f4 = np.polyfit(s["μAs"]*0.001, avg, 1)
    return f1, f2, f3, f4

def get_i0(input_path = r".\output\70kVp"):
    s = read_dicoms(input_path)
    image_count = np.array([len(i) for i in s["im"]])
    #avg = np.array([np.mean(i[...,100:-100,100:-100]*gain) for i,gain in zip(s["im"], s["gain"])])
    avg = np.array([np.mean(i)*(1+gain/100) for i,gain in zip(s["im"], s["gain"])])
    #f = np.polyfit(s["μAs"]*0.001/image_count, avg, 1)
    f = np.polyfit(s["μAs"]*0.001, avg, 1)
    #f = np.polyfit(s["mA"]*s["pw"]*0.001, avg, 2)
    return f, s["gamma"][0]


if __name__ == "__main__":
    s = read_dicoms(r"C:\Users\ich\Source\CBCT-Reco\output\CKM\CircTomo\20201207-094635.313000-P16_Card_HD")
    
    s = norm_images(s)
    f1, f2, f3, f4 = fit_i0(s)
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(s["μAs"]*0.001), np.max(s["μAs"]*0.001),100)
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
