import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit import transpile
import pandas as pd

def makeinverseclifford(cliffords, depth):  # make inverse clifford
    composed = cliffords[0]
    for i in range(1, depth):
        composed = composed.compose(cliffords[i])
    inverse_clifford = composed.to_circuit().inverse()
    return inverse_clifford

def makecircuits(delay, depth, aq, tq, numqubits):
    cliffords = []  # list of random clifford gates
    qlist = range(numqubits)
    qlist = list(qlist)
    qlist.remove(aq)
    qlist.remove(tq)
    for _ in range(depth):
        cliffords.append(random_clifford(1))

    mcmqc = QuantumCircuit(numqubits, numqubits)
    mcmqc.name = f"mcm {aq}/{tq} {depth}"
    delqc = QuantumCircuit(numqubits, numqubits)
    delqc.name = f"del {aq}/{tq}  {depth}"
    
    for i in range(depth):
        mcmqc.append(cliffords[i].to_instruction(), [tq])
        mcmqc.barrier()
        mcmqc.measure(aq, 1)
        #mcmqc.measure(qlist, qlist)
        #mcmqc.reset(aq)
        #mcmqc.reset(qlist)
        mcmqc.barrier()
        
        delqc.append(cliffords[i].to_instruction(), [tq])
        delqc.barrier()
        delqc.delay(delay, unit="ns")
        delqc.barrier()
        
    inverseclifford = makeinverseclifford(cliffords, depth)
    delqc.append(inverseclifford, [tq])
    mcmqc.append(inverseclifford, [tq])
    
    
    mcmqc.barrier()
    delqc.barrier()
    #mcmqc.measure(qlist, qlist)
    #delqc.measure(qlist, qlist)
    mcmqc.measure(tq, 0)
    mcmqc.measure(aq, 1)
    delqc.measure(tq, 0)
    delqc.measure(aq, 1)
    return [delqc, mcmqc]

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

def transpilecircuits(mindepth, maxdepth, tqindex, aqindex, backend):
    delay = backend.properties().to_dict().get('qubits')[aqindex][-1]['value'] + backend.properties().to_dict().get('gates')[-1 - 6 + aqindex]['parameters'][0]['value']
    #delay = 10
    tcircuits = []
    num_qubits = backend.configuration().n_qubits
    for depth in range(mindepth, maxdepth):
        circuits = makecircuits(delay, depth, aqindex, tqindex, num_qubits)
        tmcmqc = transpile(circuits[1], backend, scheduling_method="alap")
        #tmcmqc = transpile(circuits[1], backend)
        tdelqc = transpile(circuits[0], backend, scheduling_method="alap")
        #tdelqc = transpile(circuits[0], backend)
        tcircuits.append([tmcmqc, tdelqc])
    return tcircuits

def getexpectedcounts(results, gatelist):
    expcountlist = []
    for i in range(len(results)):
        counts = results[i]['00'] + results[i]['01']
        expcountlist.append([counts])
    
    d = dict(zip(gatelist, expcountlist))
    return pd.DataFrame(data=d)

def expfunction_alpha(x, a, b, c):
    return a * b**x + c


