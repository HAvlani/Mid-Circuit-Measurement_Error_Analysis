import rbbuild
import functools
import operator
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_provider.job.exceptions import IBMJobFailureError
import time
from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import os

shots = 3000
depth = 70
tapairs = [[0,1], [0,6], [4,6]] 
backends = ["ibm_lagos", "ibm_nairobi", "ibmq_jakarta", "ibm_perth"]
unordered_pairs = [0 ,1, 4, 6]


def errorcounts(jobid1, jobid2, jobid3):
    provider = IBMProvider()
    retrieved_job1 = provider.backend.retrieve_job(jobid1)
    retrieved_job2 = provider.backend.retrieve_job(jobid2)
    retrieved_job3 = provider.backend.retrieve_job(jobid3)
    errorcounts = []
    for i in retrieved_job1.result().get_counts():
        try:
            errorcounts.append(i['0000001'] + i['0000011'])
        except KeyError:
            errorcounts.append(i['0000001'])
    for i in retrieved_job2.result().get_counts():
        try:
            errorcounts.append(i['0000001'] + i['0000011'])
        except KeyError:
            errorcounts.append(i['0000001'])
    for i in retrieved_job3.result().get_counts():
        try:
            errorcounts.append(i['0000001'] + i['0000011'])
        except KeyError:
            errorcounts.append(i['0000001'])
    return errorcounts
        
    

def runcircuits(tapairs, backendname, depth, shots):
    while True:
        provider = IBMProvider()
        backend = provider.get_backend(backendname)
        print(f"{backendname} initialized")
        totalcircuits = []
        for i in tapairs:
            circuits = rbbuild.transpilecircuits(1, depth, i[0], i[1], backend)
            totalcircuits.append(circuits)
        circuitlist = functools.reduce(operator.iconcat, totalcircuits, [])
        circuitlist = functools.reduce(operator.iconcat, circuitlist, [])
        print(f"{backendname} transpiled")
        print(f'{backendname} {len(circuitlist[:90])} {len(circuitlist[90:180])} {len(circuitlist[180:])}')
        job1 = backend.run(circuitlist[:90], shots=shots, rep_delay=0.0005)
        job2 = backend.run(circuitlist[90:180], shots=shots, rep_delay=0.0005)
        job3 = backend.run(circuitlist[180:], shots=shots, rep_delay=0.0005)
        print(f"{backendname} submitted")
        try:
            result1 = job1.result()
            result2 = job2.result()
            result3 = job3.result()
        except IBMJobFailureError:
            print(f"job error {backendname}")
            result1 = None
            result2 = None
            result3 = None
        
        now = datetime.now()
        print(f"{backend} finished")
        if result1 != None and result2 != None and result3!=None:
            calibration_data = backend.properties(datetime=now)
            caldict = []
            for i in unordered_pairs:
                b = calibration_data.qubit_property(i)
                caldict.append(b)
            for i in range(len(caldict)):
                for k, v in caldict[i].items():
                    caldict[i][k] = v[0]
            caldf = pd.DataFrame.from_dict(caldict)
            caldf.insert(0, "Qubit", [0, 1, 4, 6], True)
            caldf.set_index('Qubit')
            errorcounter = errorcounts(job1.job_id(), job2.job_id(), job3.job_id())
            print(f'{2*(depth-1)} {4*(depth-1)} {6*(depth-1)}')
            print(f"{backend} parsing")
            deldf_0_1 = pd.DataFrame([errorcounter[:2*(depth-1):2]])
            mcmdf_0_1 = pd.DataFrame([errorcounter[1:2*(depth-1):2]])
            deldf_0_6 = pd.DataFrame([errorcounter[2*(depth-1):4*(depth-1):2]])
            mcmdf_0_6 = pd.DataFrame([errorcounter[2*(depth-1)+1:4*(depth-1):2]])
            deldf_4_6 = pd.DataFrame([errorcounter[4*(depth-1):6*(depth-1):2]])
            mcmdf_4_6 = pd.DataFrame([errorcounter[4*(depth-1)+1:6*(depth-1):2]])
            frames = [mcmdf_0_1, deldf_0_1, mcmdf_4_6, deldf_4_6, mcmdf_0_6, deldf_0_6]
            errordf = pd.concat(frames)
            errordf.insert(0, "Type", ['MCM', 'Del', 'MCM', 'Del', 'MCM', 'Del'], True)
            errordf.insert(0, "Pair", [[0, 1], [0, 1], [4, 6], [4, 6], [0, 6], [0, 6]], True)
            print(f"{backend} df created")
            print(now)
            os.mkdir(f'/Users/harshilavlani/MCM Project/{backendname}/{now}/')
            errordf.to_csv(f'/Users/harshilavlani/MCM Project/{backendname}/{now}/errorcounts {now}.csv')
            caldf.to_csv(f'/Users/harshilavlani/MCM Project/{backendname}/{now}/caldata {now}.csv')
        print(f"{backendname} wait started")
        time.sleep(600)
    

if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(runcircuits)(tapairs, i, depth, shots) for i in backends)

       