if __name__ == "__main__":
    import astra
    print(astra.get_gpu_info(0))
    print(astra.get_gpu_info(1))
    print(astra.get_gpu_info(2))
    print(astra.get_gpu_info(3))
    astra.set_gpu_index(3)

    import numpy as np
    import SimpleITK as sitk
    import cal
    import utils
    import os
    import time
    import i0_data
    import pickle
    import cal_bfgs_both
    import load_data
    from config import *
    import multiprocessing as mp
    import threads
    import est_position
    from evaluate import evalAllResults

    def reg_rough(ims, ims_big, params, config, c=0):
        corrs = [None]*len(params)
        config["log_queue"] = mp.Queue()
        with open(config["data_dump_path"], "rb") as f:
            est_data_ser = pickle.load(f)
            config["est_data"] = utils.unserialize_est_data(est_data_ser)
            est_data_ser = None
        for i in reversed(range(len(params))):
            #if i != 486: continue
        #for i in [29]:
            print(i, end=",", flush=True)
            cur = params[i]
            print(cur)
            real_img = cal.Projection_Preprocessing(ims[i])
            config["real_img_small"] = real_img
            real_img_big = cal.Projection_Preprocessing(ims_big[i])
            config["real_img_big"] = real_img_big
            for si in range(1):
                try:
                    #old_cur = np.array(cur)
                    if config["estimate"]:
                        config["Ax"] = config["Ax_small"]
                        config["real_img"] = config["real_img_small"]
                        cur = cal.roughRegistration(cur, config, 61)
                    config["Ax"] = config["Ax_big"]
                    config["real_img"] = config["real_img_big"]
                    if c >= 0:
                        cur = cal.roughRegistration(cur, config, c)
                    else:
                        cur = cal_bfgs_both.bfgs(cur, config, c)
                except Exception as ex:
                    print(i, ex, cur)
                    raise
                #if (np.abs(old_cur-cur)<1e-8).all():
                #    print(si, end=" ", flush=True)
                #    break
            corrs[i] = cur
            #print(flush=True)
            
        corrs = np.array(corrs)
        #print(corrs)
        return corrs

    def reg_and_reco(ims_big, ims, in_params, config):
        name = config["name"]
        grad_width = config["grad_width"] if "grad_width" in config else (1,25)
        perf = config["perf"] if "perf" in config else False
        Ax = config["Ax"]
        Ax_big = config["Ax_big"]
        method = config["method"]
        use_saved = config["use_saved"] if "use_saved" in config else False
        real_image = config["real_cbct"]
        outpath = config["outpath"]

        print(name, grad_width)
        params = np.array(in_params[:])
        if False and not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name.split('_',1)[0]+"_reco-input.nrrd")):
            rec = sitk.GetImageFromArray(real_image)*100
            rec.SetOrigin(out_rec_meta[0])
            out_spacing = (out_rec_meta[2][0],out_rec_meta[2][1],out_rec_meta[2][2])
            rec.SetSpacing(out_spacing)
            sitk.WriteImage(rec, os.path.join(outpath, "forcast_"+name.split('_',1)[0]+"_reco-input.nrrd"))
        if not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_projs-input.nrrd")):
            sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(ims,0,1)))
            sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+("_est_" if config["estimate"] else "")+"_projs-input.nrrd"), True)
            del sino
            sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(ims_big,0,1)))
            sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+("_est_" if config["estimate"] else "")+"_projs-input2.nrrd"), True)
            del sino
        if False and not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_reco-input.nrrd")):
            reg_geo = Ax.create_geo(params)
            write_rec(reg_geo, ims, os.path.join(outpath, "forcast_"+name+"_reco-input.nrrd"), out_rec_meta)
        if not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_sino-input.nrrd")):
            sino = cal.Projection_Preprocessing(Ax(params))
            #img = cv2.drawMatchesKnn(np.array(255*(ims[-1]-np.min(ims[-1]))/(np.max(ims[-1])-np.min(ims[-1])),dtype=np.uint8), None,
            #    np.array(255*(sino[:,-1]-np.min(sino[:,-1]))/(np.max(sino[:,-1])-np.min(sino[:,-1])),dtype=np.uint8),None, None, None)
            #cv2.imwrite("img\\check_" + name + "_pre.png", img)
            sino = sitk.GetImageFromArray(sino)
            sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_sino-input.nrrd"), True)
            del sino

        cali = {}
        cali['feat_thres'] = 80
        cali['iterations'] = 50
        cali['confidence_thres'] = 0.025
        cali['relax_factor'] = 0.3
        cali['match_thres'] = 60
        cali['max_ratio'] = 0.9
        cali['max_distance'] = 20
        cali['outlier_confidence'] = 85

        perftime = time.perf_counter()
        if use_saved:
            vecs = utils.read_vectors(name+"-rough")
            corrs = utils.read_vectors(name+"-rough-corr")
        else:
            if config["paralell"] and  mp.cpu_count() > 1:
                corrs = threads.reg_rough_parallel(ims, ims_big, params, config, method)
            else:
                corrs = reg_rough(ims, ims_big, params, config, method)

        vecs = Ax.create_vecs(corrs)
        utils.write_vectors(name+"-rough-corr", corrs)
        utils.write_vectors(name+"-rough", vecs)

        
        perftime = time.perf_counter()-perftime

        #print(params, corrs)
        if not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_sino-input.nrrd")):
            sino = Ax(corrs)
            #img = cv2.drawMatchesKnn(np.array(255*(ims[-1]-np.min(ims[-1]))/(np.max(ims[-1])-np.min(ims[-1])),dtype=np.uint8), None,
            #    np.array(255*(sino[:,-1]-np.min(sino[:,-1]))/(np.max(sino[:,-1])-np.min(sino[:,-1])),dtype=np.uint8),None, None, None)
            #cv2.imwrite("img\\check_" + name + "_post.png", img)
            sino = sitk.GetImageFromArray(sino)
            sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+("_est_" if config["estimate"] else "")+"_sino-output.nrrd"), True)
            #evalPerformance(np.swapaxes(sino, 0, 1), ims, perftime, name)
            del sino

        print("rough reg done ", perftime)

        if not perf:
            reg_geo = Ax_big.create_geo(corrs)
            mult = 1
            utils.write_rec(reg_geo, ims_big, os.path.join(outpath, "forcast_"+name+("_est_" if config["estimate"] else "")+"_reco-output.nrrd"), out_rec_meta, mult)
            #reg_geo = Ax.create_geo(corrs)
            #mult = 1
            #utils.write_rec(reg_geo, ims, os.path.join(outpath, "forcast_"+name+("_est_" if config["estimate"] else "")+"_reco-output-small.nrrd"), out_rec_meta, mult)
            

        return vecs, corrs

    out_rec_meta = ()

    def reg_real_data():
        projs = get_proj_paths()

        #np.seterr(all='raise')

        for name, proj_path, cbct_path, methods in projs:
            
            try:
                ims, ims_un, mas, kvs, angles, coord_systems, sids, sods = load_data.read_dicoms(proj_path)
                
                if False and os.path.exists("Z:\\\\recos"):
                    outpath = "Z:\\\\recos"
                elif os.path.exists(r"D:\lumbal_spine_13.10.2020\recos"):
                    outpath = r"D:\lumbal_spine_13.10.2020\recos"
                elif os.path.exists("E:\\\\recos"):
                    outpath = "E:\\\\recos"
                else:
                    outpath = r".\recos"

                #target_sino = sitk.ReadImage(os.path.join(outpath, "target_sino.nrrd"))
                #target_sino = sitk.GetArrayFromImage(target_sino)
                #print("target_sino", target_sino.shape)
                print("ims shape", ims.shape)
                print("ims_un shape", ims_un.shape)

                #ims = ims[:20]
                #coord_systems = coord_systems[:20]
                #skip = max(1, int(len(ims_un)/500))
                skip = np.zeros(len(ims_un), dtype=bool)
                #skip[-480:-450] = True
                skip[::1] = True
                #skip[::max(1, int(len(ims_un)/500))] = True
                random = np.random.default_rng(23)
                #angles_noise = random.normal(loc=0, scale=0.5, size=(len(ims), 3))#*np.pi/180
                angles_noise = random.uniform(low=-2, high=2, size=(len(ims_un),3))
                #angles_noise = np.zeros_like(angles_noise)
                #trans_noise = random.normal(loc=0, scale=20, size=(len(ims), 3))
                min_trans, max_trans = -10, 10
                min_trans, max_trans = -5, 5
                trans_noise = random.uniform(low=min_trans, high=max_trans, size=(len(ims_un),2))
                #zoom_noise = random.uniform(low=0.95, high=1, size=len(ims_un))
                zoom_noise = random.uniform(low=0.98, high=1, size=len(ims_un))

                #skip = 4
                ims = ims[skip]
                ims_un = ims_un[skip]
                coord_systems = coord_systems[skip]
                #angles = angles[skip]
                sids = np.mean(sids[skip])
                sods = np.mean(sods[skip])
                angles_noise = angles_noise[skip]
                trans_noise = trans_noise[skip]
                zoom_noise = zoom_noise[skip]
                angles_noise = np.ones_like(angles_noise)*0
                trans_noise = np.ones_like(trans_noise)*0
                zoom_noise = np.ones_like(zoom_noise)

                coords_from_angles = utils.angles2coord_system(angles)

                origin, size, spacing, image = utils.read_cbct_info(cbct_path)

                detector_shape = np.array((1920,2480))
                detector_mult = int(np.floor(detector_shape[0] / ims_un.shape[1]))
                detector_mult1 = int(np.floor(detector_shape[0] / ims.shape[1]))
                
                detector_shape = np.array(ims_un.shape[1:])
                detector_shape1 = np.array(ims.shape[1:])
                #detector_spacing = np.array((0.125, 0.125)) * detector_mult
                detector_spacing = np.array((0.154, 0.154)) * detector_mult
                detector_spacing1 = np.array((0.154, 0.154)) * detector_mult1

                real_image = utils.fromHU(sitk.GetArrayFromImage(image))
                #print(real_image.shape)
                #real_image = sitk.GetArrayFromImage(sitk.ReadImage("Z:\\recos\\forcast_201020_imbu_cbct_4_reco-output.nrrd"))
                mask = np.zeros(real_image.shape, dtype=bool)
                mask = utils.create_circular_mask(real_image.shape)
                real_image = real_image*mask*0.001
                del mask
                print(real_image.shape)
                #real_image = np.swapaxes(np.swapaxes(real_image, 0,2), 0,1)[::-1,:,::-1]

                global out_rec_meta
                out_rec_meta = (image.GetOrigin(), image.GetSize(), image.GetSpacing(), real_image.shape)
                del image

                image_spacing = 1.0 / np.min(spacing)
                print(spacing, image_spacing, np.array((1920,2480))/np.array(ims_un[0].shape), detector_mult)

                Ax = utils.Ax_param_asta(real_image.shape, detector_spacing, detector_shape, sods, sids-sods, image_spacing, real_image)
                Ax_gen = (real_image.shape, detector_spacing, detector_shape, sods, sids-sods, image_spacing, real_image)
                
                Ax_big = utils.Ax_param_asta(real_image.shape, detector_spacing1, detector_shape1, sods, sids-sods, image_spacing, real_image)
                Ax_gen_big = (real_image.shape, detector_spacing1, detector_shape1, sods, sids-sods, image_spacing, real_image)

                #if coord_systems.shape[1] == 4:
                #    coord_systems, thetas, phis, params = interpol_positions(coord_systems, Ax, ims, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
                #    params = params[skip]
                #coord_systems = coord_systems[skip]

                geo = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
                geo_from_angles = utils.create_astra_geo_coords(coords_from_angles, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
                #geo = geo_from_angles
                
                r = utils.rotMat(90, [1,0,0]).dot(utils.rotMat(-90, [0,0,1]))

                if 'arc' in name:
                    #coord_systems, thetas, phis, params = interpol_positions(coord_systems, Ax, ims, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
                    #params = params[skip]
                    #coord_systems = coord_systems[skip]
                    target_sino = np.swapaxes(ims, 0,1)
                if 'angle' in name or 'both' in name:
                    params = np.zeros((len(geo_from_angles['Vectors']), 3, 3), dtype=float)
                    params[:,1] = np.array([r.dot(v) for v in geo_from_angles['Vectors'][:, 6:9]])
                    params[:,2] = np.array([r.dot(v) for v in geo_from_angles['Vectors'][:, 9:12]])
                else:
                    params = np.zeros((len(geo['Vectors']), 3, 3), dtype=float)
                    params0 = np.zeros((len(geo['Vectors']), 3, 3), dtype=float)
                    params0[:,1,0] = 1
                    params0[:,2,1] = 1
                    #params[:,0] = coord_systems[:,:,3]
                    params[:,1] = np.array([r.dot(v) for v in geo['Vectors'][:, 6:9]])
                    #params[:,1] = np.array(geo['Vectors'][:, 6:9])
                    params[:,2] = np.array([r.dot(v) for v in geo['Vectors'][:, 9:12]])
                    #params[:,2] = np.array(geo['Vectors'][:, 9:12])
                
                #print(params[0,1], params[0,2])

                if True:
                    for i, (α,β,γ) in enumerate(angles_noise):
                        params[i] = utils.applyRot(params[i], -α, -β, -γ)
                if True:
                    for i, (x,y) in enumerate(trans_noise):
                        params[i] = utils.applyTrans(params[i], x, y, 0)
                    for i, z in enumerate(zoom_noise):
                        params[i] = utils.applyTrans(params[i], 0, 0, 1-z)

                projs = Ax(params)
                #Ax0201i0 = utils.Ax_param_asta(real_image0201i0.shape, detector_spacing, detector_shape, sods, sids-sods, image_spacing, real_image0201i0)
                #projs0201i0 = Ax0201i0(params0)
                #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/projs.nrrd")
                #sitk.WriteImage(sitk.GetImageFromArray(projs0201i0), "recos/projs0201i0.nrrd")
                #sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(ims,0,1)), "recos/ims.nrrd")

                i0_ims, i0_mas, i0_kvs = i0_data.i0_data(detector_mult, ims_un.shape[1])

                res = np.mean( np.mean(i0_ims, axis=(1,2))[:,np.newaxis,np.newaxis] / i0_ims, axis=0)

                i0s = np.array([i0_data.i0_est(ims_un[i], projs[:,i])*res for i in range(ims_un.shape[0])])
                i0s = np.mean(i0s, axis=0)
                i0s[i0s==0] = 1e-8
                #i0s = np.mean(i0s)
                ims_un = -np.log(ims_un/i0s)

                #sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(-np.log(ims/i0s) ,0,1)))
                #sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_projs_est-input.nrrd"), True)
                #del sino

                #i0s = i0_interpol(i0_ims, i0_mas, np.mean(mas))

                #sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(-np.log(ims/i0s) ,0,1)))
                #sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_projs_int-input.nrrd"), True)
                #del sino
                i0_ims, i0_mas, i0_kvs = i0_data.i0_data(detector_mult1, ims.shape[1])

                res = np.mean( np.mean(i0_ims, axis=(1,2))[:,np.newaxis,np.newaxis] / i0_ims, axis=0)

                i0s = np.array([i0_data.i0_est(ims[i], projs[:,i])*res for i in range(ims.shape[0])])
                i0s = np.mean(i0s, axis=0)
                i0s[i0s==0] = 1e-8
                #i0s = np.mean(i0s)
                ims = -np.log(ims/i0s)
                
                #calc_images_matlab("input", ims, real_image, detector_shape, outpath, geo); 
                #calc_images_matlab("genA_trans", ims, real_image, detector_shape, outpath, geo); exit(0)

                config = dict(forcast_config)
                config.update({"Ax": Ax, "Ax_small": Ax, "Ax_big": Ax_big, "Ax_gen": Ax_gen, "Ax_gen_big": Ax_gen_big, "real_cbct": real_image,
                        "outpath": outpath, "estimate": False, "threads": 6, "paralell": True, "angles": angles})

                config["data_dump_path"] = os.path.join(outpath, name.split("_")[0]+"_est_data_"+str(tdim)+str(sdim)+str(pdim) + ".dump")

                if not os.path.exists(config["data_dump_path"]): 
                    perftime = time.perf_counter()
                    cur0 = np.zeros((3, 3), dtype=float)
                    cur0[1,0] = 1
                    cur0[2,1] = 1
                    est_data = est_position.simulate_est_data(cur0, Ax)
                    with open(config["data_dump_path"], "wb") as f:
                        pickle.dump(est_data, f)
                    #est_data = utils.unserialize_est_data(config["est_data_ser"])
                    est_data = None
                    print("est data", time.perf_counter()-perftime)
                    #print(config["est_data"][1][0], est_data[1][0])

                with open(config["data_dump_path"], "rb") as f:
                    est_data = pickle.load(f)
                    meta, shms = utils.into_shm(est_data)
                    del est_data
                    config["shm_meta"] = meta
                    config["shms"] = shms
                

                #for method in [3,4,5,0,6]: #-12,-2,-13,-3,20,4,26,31,0,-1
                for method in methods:
                    config["name"] = name + str(method)
                    config["method"] = method
                    vecs, corrs = reg_and_reco(ims, ims_un, np.array(params), config)
                    #iso = (geo['Vectors'][0,0:3]+(sods/sids)*(geo['Vectors'][0,3:6]-geo['Vectors'][0,0:3]))/image_spacing
                    #print((params-corrs)[:,0] / image_spacing, origin[0], params[:,0], corrs[:,0]/image_spacing, np.array(real_image.shape)*spacing)
                    #print(coord_systems[0,:,3]-iso, geo['Vectors'][0,0:3]/image_spacing-iso, vecs[0,0:3]/image_spacing)
                    #print(np.linalg.norm(geo['Vectors'][0,0:3]-geo['Vectors'][0,3:6])/image_spacing, np.linalg.norm(vecs[0,0:3]-vecs[0,3:6])/image_spacing, sids, sods)
                    #print(iso)
                    #print((vecs[0,0:3]+(sods/sids)*(vecs[0,3:6]-vecs[0,0:3])) /image_spacing)
                    #exit()
                
                for shm in config["shms"]:
                    shm.unlink()
                    shm.close()
                del shms
                del config["shms"]
            except Exception as e:
                print(name, "cali failed", e)
                raise

if __name__ == "__main__":
    #import cProfile, io, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    reg_real_data()
    #evalAllResults(True, True, "D:\\lumbal_spine_13.10.2020\\recos")
    
    #profiler.disable()
    #s = io.StringIO()
    #sortby = pstats.SortKey.TIME
    #ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    #ps.print_stats(20)
    #print(s.getvalue())