
import sys
import io
import numpy as np
import cProfile
import cal
import cal_bfgs_both
import threads_utils as utils

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
        #cur0 = np.zeros((3, 3), dtype=float)
        #cur0[1,0] = 1
        #cur0[2,1] = 1
        #est_data = cal.simulate_est_data(cur0, Ax)
        #est_data_ser = [None, None]
        #est_data = [None, None]
        #shm1 = mp_shm.SharedMemory(est_data_name+"_0")
        #est_data[0] = shm1.buf
        #est_data_ser[0] = pickle.loads(shm.buf)
        #shm2 = mp_shm.SharedMemory(est_data_name+"_1")
        #est_data[1] = shm2.buf
        #est_data_ser[1] = pickle.loads(shm.buf)
        #with open(est_data_name, "rb") as f:
        #    est_data_ser = pickle.load(f)
        #    est_data = utils.unserialize_est_data(est_data_ser)
        #    del est_data_ser
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
