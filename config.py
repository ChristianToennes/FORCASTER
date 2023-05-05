import os
forcast_config = {"estimate": False, "threads": 6, "paralell": True}

pdim = 95
sdim = 95
tdim = 3

def get_proj_paths():
    projs = []

    if os.path.exists("F:\\output"):
        prefix = r"F:\output"
    elif os.path.exists("D:\\lumbal_spine_13.10.2020\\output"):
        prefix = r"D:\lumbal_spine_13.10.2020\output"
    else:
        prefix = r".\output"
    
    cbct_path = prefix + r"\CKM_LumbalSpine\20201020-151825.858000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    #cbct_path = prefix + r"\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    projs += [
    #('genA_trans', prefix+'\\gen_dataset\\only_trans', cbct_path, [4]),
    #('genA_angle', prefix+'\\gen_dataset\\only_angle', cbct_path, [4,20,21,22,23,24,25,26]),
    #('genA_both', prefix+'\\gen_dataset\\noisy', cbct_path, [4,20,21,22,23,24,25,26]),
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-58,42]),
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [50,52]), # normal noise
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-44,-58,-34,-24,33,42]), # normal noise
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-43,-57,34,41]), # reduced noise
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-73,-67]), # normal noise trans
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-72,-66]), # reduced noise trans
    #('201020_imbu_sin_', prefix + '\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD', cbct_path, [-58,42]),
    #('201020_imbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    #('201020_imbu_circ_', prefix + '\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD', cbct_path, [4, -34, -35, 28, 29]),
    #('201020_noimbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV', cbct_path, [4, 60, 61, 62]),
    #('201020_noimbu_arc_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\P16_DR_LD', cbct_path, [-70, 70, 60.5, 62]),
    #('201020_imbureg_noimbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    ]
    
    cbct_path = prefix + r"\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    projs += [
    #('genB_trans', prefix+'\\gen_dataset\\only_trans', cbct_path, [4]),
    #('genB_angle', prefix+'\\gen_dataset\\only_angle', cbct_path),
    #('genB_both', prefix+'\\gen_dataset\\noisy', cbct_path),
    ('2010201_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [60, 60.5, 62.5]),
    #('2010201_imbu_sin_', prefix + '\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD', cbct_path, [-70, 70, 4, 60.5, 62.5]),
    #('2010201_imbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD', cbct_path, [4,60,61,62]),
    #('2010201_imbu_circ_', prefix + '\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD', cbct_path, [4,60,61,62]),
    ('2010201_imbu_arc_', prefix + '\\CKM_LumbalSpine\\Arc\\20201020-150938.350000-P16_DR_LD', cbct_path, [60, 60.5, 62.5, 70]),
    #('2010201_imbu_arc_', prefix + '\\CKM_LumbalSpine\\20201020-150938.350000\\P16_DR_LD', cbct_path, [60.5, 62, 70]),
    #('2010201_noimbu_arc_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\P16_DR_LD', cbct_path, [60,62]),
    #('2010201_imbureg_noimbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV', cbct_path, [4, 28, 29]),
    #('2010201_imbureg_noimbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    ]
    return projs
    