import numpy as np
from scipy.optimize import curve_fit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit.providers.aer import Aer
from qiskit_ibm_provider import IBMProvider
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
                              pauli_error, depolarizing_error, thermal_relaxation_error)
import matplotlib.pyplot as plt
import qiskit.providers.aer.noise as noise
#import scipy.stats.chisquare as chisquare
import ipdb

#ipdb.set_trace()


def makeinverseclifford(cliffords, depth):  # make inverse clifford
    composed = cliffords[0]
    for i in range(1, depth):
        composed = composed.compose(cliffords[i])
    inverse_clifford = composed.to_circuit().inverse()
    return inverse_clifford


def makemcmrb(depth, realqc, acq, cq):
    cliffords = []  # list of random clifford gates
    
    for _ in range(depth):
        cliffords.append(random_clifford(1))

    qc = QuantumCircuit(max([acq, cq]) + 1, 2)
    qc.name = 'mcm-rb cicuit'

    for i in range(depth):
        qc.append(cliffords[i].to_instruction(), [cq])
        qc.barrier()
        qc.measure(acq, 1)
        qc.reset(acq)
        qc.barrier()

    # make inverse clifford gate to return control qubit to |0>
    inverseclifford = makeinverseclifford(cliffords, depth)
    qc.append(inverseclifford, [cq])
    qc.barrier()
    if realqc is False:
        qc.save_probabilities()
    qc.measure(cq, 0)
    qc.measure(acq, 1)
   #print(qc)
    return qc


def makedelayrb(delay, depth, realqc, acq, cq):
    cliffords = []
    
    for _ in range(depth):
        cliffords.append(random_clifford(1))

    qc = QuantumCircuit(max([acq, cq]) + 1, 2)
    qc.name = 'delay-rb cicuit'
    for i in range(depth):
        qc.append(cliffords[i].to_instruction(), [cq])
        qc.barrier()
        qc.delay(delay, unit="ns")
        qc.barrier()

    # make inverse clifford gate to return control qubit to |0>
    inverseclifford = makeinverseclifford(cliffords, depth)
    qc.append(inverseclifford, [cq])
    if realqc is False:
        qc.save_probabilities()
    qc.measure(cq, 0)
    qc.measure(acq, 1)
    return qc


def addthermalerror():
    # T1 and T2 values for qubits 0-3
    # Sampled from normal distribution mean 50 microsec
    T1s = np.random.normal(50e3, 10e3, 4)
    # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, 4)
    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])
    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond
    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
        for t1a, t2a in zip(T1s, T2s)]
        for t1b, t2b in zip(T1s, T2s)]
    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(4):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(4):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
    return noise_thermal

    
def runmcmrb(realqc, mindepth, maxdepth, cqindex, acqindex, backend_name="", thermalerror=False, backenderror=False, depolerror=False):
    provider = IBMProvider()
    
    if realqc:
        qclist = []
        measurelist = []
        backend = provider.get_backend(backend_name)
        for i in range(mindepth, maxdepth, 1):
            transpiledqc = transpile(makemcmrb(i, realqc, acqindex, cqindex), backend)
            qclist.append(transpiledqc)
            measurelist.append(i)
            #print(qclist)
        #print(backend.configuration().rep_delay_range)
        job = backend.run(qclist, shots=10000, rep_delay=0.0005)
        result = job.result()
        #print(result.get_counts())
        #print(measurelist)
        return result.get_counts()
    else:
        if thermalerror:
            noise_model = addthermalerror()
            backend = AerSimulator(noise_model=noise_model)
        elif backenderror:
            backend = provider.get_backend('ibm_nairobi')
            noise_model = NoiseModel.from_backend(backend)
            backend = AerSimulator(noise_model=noise_model)
        elif depolerror:
            meas_channel = noise.depolarizing_error(
                1e-12, 1).tensor(noise.phase_damping_error(1.0))
            noise_model = noise.NoiseModel()
            noise_model.add_quantum_error(meas_channel, ['meas_u'], [
                                          1, 0])
            backend = AerSimulator(noise_model=noise_model)
        else:
            backend = AerSimulator()
        transpiledqc = transpile(makemcmrb(mindepth, realqc, acqindex, cqindex), backend)
        job = backend.run(transpiledqc, shots=100)
        result = job.result()
        probabilities = result.results[0].data.probabilities
        return probabilities[0]
        

def rundelayrb(realqc, mindepth, maxdepth, cqindex, acqindex, delay=0, backend_name="", thermalerror=False, backenderror=False, depolerror=False):
    provider = IBMProvider()

    if realqc:
        qclist = []
        measurelist = []
        backend = provider.get_backend(backend_name)
        delay = backend.properties().to_dict().get('qubits')[0][-1]["value"]
        for i in range(mindepth, maxdepth, 1):
            transpiledqc = transpile(makedelayrb(delay, i, realqc, acqindex, cqindex), backend)
            qclist.append(transpiledqc)
            measurelist.append(i)
            #print(qclist)
        job = backend.run(qclist, shots=10000, rep_delay=0.0005)
        result = job.result()
        #print(result.get_counts())
        #print(measurelist)
        return result.get_counts()

    else:
        if thermalerror:
            noise_model = addthermalerror()
            backend = AerSimulator(noise_model=noise_model)
        elif backenderror:
            backend = provider.get_backend('ibm_nairobi')
            noise_model = NoiseModel.from_backend(backend)
            backend = AerSimulator(noise_model=noise_model)
        elif depolerror:
            meas_channel = noise.depolarizing_error(
                1e-12, 1).tensor(noise.phase_damping_error(1.0))
            noise_model = noise.NoiseModel()
            noise_model.add_quantum_error(meas_channel, ['meas_u'], [
                                          1, 0])
            backend = AerSimulator(noise_model=noise_model)
        else:
            backend = AerSimulator()
        transpiledqc = transpile(makedelayrb(delay, mindepth, realqc, acqindex, cqindex), backend)
        job = backend.run(transpiledqc, shots=100)
        result = job.result()
        
        #try:
        #print(result.get_statevector())
        #except:
         #   print("no statevector available")
        #with open("./delayrbresults.json", "w") as outfile:
           # json.dump(result.to_dict(), outfile)
        #print(result.to_dict())
        #print(result.get_counts())
        probabilities = result.results[0].data.probabilities
        return probabilities[0]

def getexpectedcounts(results, gatelist):
    expcountlist = []
    for i in range(len(results)):
        counts = results[i]['00']
        
        expcountlist.append(counts)
    #print(dict(zip(gatelist, expcountlist)))
    return dict(zip(gatelist, expcountlist))

def getcountmatrix(resultsdelay, resultsmcm):
    x = []
    for i in range(len(resultsdelay)):
        zerocountdelay = resultsdelay[i]['00']
        zerocountmcm = resultsmcm[i]['00']
        x.append([zerocountdelay, zerocountmcm])
    #print(np.matrix(x))
    return np.matrix(x)
    
    

def expfunction_alpha(x, a, b, c):
    return a * b**x + c

def linearfunc(x, a, b):
    return a*x + b

def parafunc(x, a, b, c):
    return a*x**2 + b*x + c


#makedelayrb(2, 5, True, 4, 1)
#rundelaysimdata = {}
#runmcmsimdata = {}
#for i in range(2, 100, 5):
#rundelaysimdata[i] = rundelayrb(False, 4, backenderror = True)
#    runmcmsimdata[i] = runmcmrb(False, i, backenderror = True)
#rundelaydata = {}
#runmcmdata = {}
#runmcmrb(True, 1, 3, 0, 1, backend_name="ibm_nairobi")
#rundelayrb(True, 1, 2, 0, 1 backend_name = "ibm_nairobi")
#getproportion([{'00': 45, '01': 55}, {'00': 57, '01': 42, '10': 1}, {'00': 57, '01': 41, '10': 1, '11': 1}, {'00': 62, '01': 37, '10': 1}, {'00': 47, '01': 50, '10': 2, '11': 1}, {'00': 51, '01': 43, '10': 5, '11': 1}, {'00': 49, '01': 49, '10': 1, '11': 1}, {'00': 49, '01': 48, '10': 1, '11': 2}, {'00': 54, '01': 44, '11': 2}, {'00': 46, '01': 48, '10': 4, '11': 2}, {'00': 58, '01': 39, '10': 2, '11': 1}, {'00': 40, '01': 56, '10': 2, '11': 2}, {'00': 50, '01': 48, '10': 1, '11': 1}, {'00': 35, '01': 63, '10': 2}, {'00': 55, '01': 44, '11': 1}, {'00': 52, '01': 43, '10': 4, '11': 1}, {'00': 48, '01': 50, '10': 2}, {'00': 46, '01': 51, '10': 1, '11': 2}, {'00': 56, '01': 41, '10': 2, '11': 1}, {'00': 41, '01': 57, '10': 1, '11': 1}, {'00': 39, '01': 58, '10': 1, '11': 2}, {'00': 46, '01': 50, '10': 2, '11': 2}, {'00': 60, '01': 36, '10': 4}, {'00': 45, '01': 51, '10': 4}], [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95])
'''
x = np.array(list(runmcmsimdata.keys()))
y = np.array(list(runmcmsimdata.values()))
params, covs = curve_fit(expfunction_alpha, x, y)
a, b, c = params[0], params[1], params[2]
yfit1 = a * b**x + c


plt.scatter(rundelaysimdata.keys(), rundelaysimdata.values(), label = "delayrb")
plt.scatter(runmcmsimdata.keys(), runmcmsimdata.values(), label = "mcmrb")
plt.plot(x, yfit1)
plt.legend()
plt.show()
'''

#results_delay = [{'00': 322, '01': 173, '10': 1, '11': 4}, {'00': 255, '01': 240, '10': 3, '11': 2}, {'00': 205, '01': 286, '10': 4, '11': 5}, {'00': 235, '01': 262, '10': 2, '11': 1}, {'00': 230, '01': 264, '10': 5, '11': 1}, {'00': 252, '01': 243, '10': 2, '11': 3}, {'00': 240, '01': 257, '10': 2, '11': 1}, {'00': 245, '01': 248, '10': 2, '11': 5}, {'00': 234, '01': 259, '10': 6, '11': 1}, {'00': 252, '01': 239, '10': 3, '11': 6}, {'00': 238, '01': 255, '10': 4, '11': 3}, {'00': 258, '01': 237, '10': 1, '11': 4}, {
#    '00': 250, '01': 239, '10': 7, '11': 4}, {'00': 250, '01': 239, '10': 4, '11': 7}, {'00': 264, '01': 231, '10': 2, '11': 3}, {'00': 261, '01': 235, '10': 4}, {'00': 239, '01': 257, '10': 2, '11': 2}, {'00': 236, '01': 262, '10': 1, '11': 1}, {'00': 259, '01': 238, '10': 1, '11': 2}, {'00': 255, '01': 234, '10': 8, '11': 3}, {'00': 235, '01': 261, '10': 2, '11': 2}, {'00': 226, '01': 266, '10': 6, '11': 2}, {'00': 243, '01': 249, '10': 5, '11': 3}, {'00': 280, '01': 216, '10': 3, '11': 1}]
#gatelist_delay = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
#                  47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95]
#results_mcm = [{'00': 166, '01': 333, '10': 1}, {'00': 271, '01': 229}, {'00': 266, '01': 229, '10': 3, '11': 2}, {'00': 253, '01': 246, '10': 1}, {'00': 261, '01': 239}, {'00': 287, '01': 213}, {'00': 191, '01': 307, '10': 1, '11': 1}, {'00': 244, '01': 255, '10': 1}, {'00': 290, '01': 209, '10': 1}, {'00': 290, '01': 210}, {'00': 241, '01': 258, '10': 1}, {'00': 298, '01': 201, '11': 1}, {
#       '00': 293, '01': 207}, {'00': 267, '01': 231, '10': 2}, {'00': 252, '01': 247, '11': 1}, {'00': 261, '01': 238, '11': 1}, {'00': 263, '01': 237}, {'00': 254, '01': 243, '10': 2, '11': 1}, {'00': 247, '01': 250, '10': 2, '11': 1}, {'00': 257, '01': 241, '10': 1, '11': 1}, {'00': 253, '01': 246, '10': 1}, {'00': 232, '01': 267, '10': 1}, {'00': 261, '01': 238, '10': 1}, {'00': 248, '01': 251, '10': 1}]


#getexpectedcounts(results_delay, gatelist_delay)
