import sys
import io
import numpy as np
import cProfile
import astra
astra.set_gpu_index(3)
import cal
import cal_bfgs_both
import threads_utils as utils
import multiprocessing as mp


def reg_rough_parallel(ims, ims_big, params, config, c=0):
    corrs = []
    pool_size = config["threads"]
    #if c==28:
    #    pool_size = 2
    #elif c >= 40:
    #    pool_size = mp.cpu_count()-4
    pool = []
    proc_count = 0
    #corrsq = mp.Queue()
    ready = mp.Event()
    #for _ in range(pool_size):
    #    cons.append(mp.Pipe(True))
    #    pool.append(mp.Process(target=it_func, args=(cons[-1][1], config["Ax_gen"], ready)))
    #    pool[-1].start()
    
    corrs = np.array([None]*len(params))
    indices = list(range(len(params)))

    log_queue = mp.Queue()
    log_proc = mp.Process(target=it_log, args=(log_queue,), daemon=True)
    log_proc.start()

    #print(len(params), len(ims), config["target_sino"].shape)

    #d = pickle.dumps(config["est_data_ser"][0])
    #shm = mp_shm.SharedMemory(name="est_data_0", create=True, size=len(d))
    #shm.buf[:] = d
    #d = pickle.dumps(config["est_data_ser"][1])
    #shm1 = mp_shm.SharedMemory(name="est_data_1", create=True, size=len(d))
    #shm1.buf[:] = d

    while np.array([e is None for e in corrs]).any(): #len(indices)>0:
        ready_con = None
        while ready_con is None:
            for _ in range(len(pool), pool_size, 2):
                p = None
                for _ in range(4):
                    if(len(pool) == pool_size): break
                    p = mp.Pipe(True)
                    if "profile" in config and config["profile"]:
                        name = config["name"]+"_"+str(proc_count)
                    else:
                        name = None
                    proc = mp.Process(target=it_func, args=(p[1], config["Ax_gen"], config["Ax_gen_big"], log_queue, ready, name, config["shm_meta"]), daemon=True)
                    proc.start()
                    proc_count += 1
                    pool.append([p[0], p[1], proc, -1])
                if p is not None:
                    res = p[0].recv()
                    if res[0] != "loaded":
                        print(res)
                        return
            ready.clear()
            finished_con = []
            for con in pool:
                try:
                    if con[2].is_alive():
                        if con[0].poll():
                            res = con[0].recv()
                            if res[0] == "ready":
                                ready_con = con
                                break
                            elif res[0] == "loaded":
                                pass
                            elif res[0] == "result":
                                corrs[res[1]] = res[2]
                                #print(res[1], res[3], flush=True)
                                print(res[1], end='; ', flush=True)
                            elif res[0] == "error":
                                finished_con.append(con)
                                #corrs[con[3]] = params[con[3]]
                                exit(0)
                            else:
                                print("error", res)
                    else:
                        finished_con.append(con)
                except (OSError, BrokenPipeError, EOFError):
                    finished_con.append(con)

            for con in finished_con:
                #indices.append(con[3])
                pool.remove(con)
                con[0].close()
                con[1].close()
            if ready_con is None:
                ready.wait(1)
        if len(indices) > 0:
            i = indices.pop()
            ready_con[0].send((i, params[i], ims[i], ims_big[i], config["estimate"], c))
            ready_con[3] = i

    for con in pool:
        con[0].send((None,2,3,4,5,6))
        con[2].terminate()
        con[0].close()
        con[1].close()

    #shm.close()
    #shm.unlink()
    #shm1.close()
    #shm1.unlink()
    log_queue.put(("exit", 0))
        
    corrs = np.array(corrs.tolist())
    print()
    #print(corrs)
    return corrs
 

def it_func(con, Ax_params, Ax_params_big, log_queue, ready, name, shm_meta):
    if name != None:
        profiler = cProfile.Profile()
    try:
        
        #print("start")
        np.seterr(all='raise')
        Ax = utils.Ax_param_asta(*Ax_params)
        Ax_big = utils.Ax_param_asta(*Ax_params_big)
        del Ax_params
        del Ax_params_big
        est_data, shms = utils.from_shm(shm_meta)
        con.send(("loaded",))
        while True:
            try:
                con.send(("ready",))
                ready.set()
                (i, cur, im, im_big, estimate, method) = con.recv()
                if i == None:
                    break
                
                old_stdout = sys.stdout
                sys.stdout = stringout = io.StringIO()

                if name != None:
                    profiler.enable()    
                real_img = cal.Projection_Preprocessing(im)
                real_img_big = cal.Projection_Preprocessing(im_big)
                cur_config = {"real_img_small": real_img, "real_img_big": real_img_big, "Ax_small": Ax, "Ax_big": Ax_big, "log_queue": log_queue, "name": str(i), "est_data": est_data, "estimate": estimate}
                try:
                    if cur_config["estimate"]:
                        cur_config["Ax"] = cur_config["Ax_small"]
                        cur_config["real_img"] = cur_config["real_img_small"]
                        cur = cal.roughRegistration(cur, cur_config, 60.5)
                    cur_config["Ax"] = cur_config["Ax_big"]
                    cur_config["real_img"] = cur_config["real_img_big"]

                    if method >= 0:
                        cur = cal.roughRegistration(cur, cur_config, method)
                    else:
                        cur = cal_bfgs_both.bfgs(cur, cur_config, method)
                except Exception as ex:
                    print(ex, type(ex), i, cur, file=sys.stderr)
                    #traceback.print_exc(limit=5, file=sys.stderr)
                    con.send(("error",))
                if name != None:
                    profiler.disable()
                    profiler.dump_stats(name)
                stringout.flush()
                con.send(("result",i,cur,stringout.getvalue()))
                ready.set()
                stringout.close()
                sys.stdout = old_stdout
            except EOFError:
                break
            except BrokenPipeError:
                if name != None:
                    profiler.dump_stats(name)
                return
        if name != None:
            profiler.dump_stats(name)
        try:
            con.send(("error",))
        except EOFError:
            pass
        except BrokenPipeError:
            pass
        for shm in shms:
            shm.close()
        del shms
    except KeyboardInterrupt:
        exit()

def it_log(log_queue):
    while True:
        try:
            name, value = log_queue.get()
            if name == "exit":
                return
            with open("csv\\"+name+".csv", "a") as f:
                f.write("{};".format(value))
        except Exception as ex:
            print("logger faulty: ", ex)
