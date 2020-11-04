import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import os.path

indir = 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201013-123239.316000\\P16_DR_LD'
indir = 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201013-150514.166000\\P16_DR_LD'

kvs = []
mas = []
ts = []
thetas = []
phis = []

def conv_time(time):
    h = float(time[:2])*60*60
    m = float(time[2:4])*60
    s = float(time[4:])
    return h+m+s

for root, dirs, files in os.walk(indir):
    for entry in files:
        path = os.path.abspath(os.path.join(root, entry))
        #read DICOM files
        ds = pydicom.dcmread(path)

        kvs.append(float(ds.KVP))
        mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime))
        #ts.append(conv_time(ds.AcquisitionTime))
        #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
        ts.append(len(ts))
        thetas.append(float(ds.PositionerPrimaryAngle))
        phis.append(float(ds.PositionerSecondaryAngle))



kvs = np.array(kvs)
mas = np.array(mas)
ts = np.array(ts)
thetas = np.array(thetas)
phis = np.array(phis)
#ts = ts - ts[0]

filt = np.zeros_like(ts, dtype=bool)

for i, (theta, phi, ma) in enumerate(zip(thetas, phis, mas)):
    if ma <= np.min(mas[np.bitwise_and(thetas==theta, phis==phi)]):
        filt[i] = True

f_kvs = kvs[filt]
f_mas = mas[filt]
f_ts = ts[filt]
f_thetas = thetas[filt]
f_phis = phis[filt]

plt.figure(0)
plt.title("kvs")
plt.plot(ts, kvs)
plt.plot(f_ts, f_kvs)
plt.figure(1)
plt.title("mas")
plt.scatter(ts, mas)
plt.scatter(f_ts, f_mas)
plt.figure(2)
plt.title("angles")
plt.plot(ts, thetas)
plt.plot(ts, phis)
plt.plot(f_ts, f_thetas)
plt.plot(f_ts, f_phis)

#plt.figure(3)
#plt.title("filtered kvs")
#plt.plot(f_ts, f_kvs)
#plt.figure(4)
#plt.title("filtered mas")
#plt.scatter(f_ts, f_mas)
#plt.figure(5)
#plt.title("filtered angles")
#plt.plot(f_ts, f_thetas)
#plt.plot(f_ts, f_phis)

plt.show()