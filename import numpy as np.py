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
import json



def makeinverseclifford(cliffords, depth):  # make inverse clifford
    composed = cliffords[0]
    for i in range(1, depth):
        composed = composed.compose(cliffords[i])
    inverse_clifford = composed.to_circuit().inverse()
    return inverse_clifford


def makemcmrb(depth, realqc):
    cliffords = []  # list of random clifford gates
    depth -= 1
    for _ in range(depth):
        cliffords.append(random_clifford(1))

    qc = QuantumCircuit(2, 2)
    qc.name = '2 qubit mcm-rb cicuit'

    for i in range(depth):
        qc.append(cliffords[i].to_instruction(), [0])
        qc.barrier()
        qc.measure(1, 1)
        qc.barrier()

    # make inverse clifford gate to return control qubit to |0>
    inverseclifford = makeinverseclifford(cliffords, depth)
    qc.append(inverseclifford, [0])
    qc.barrier()
    if realqc is False:
        qc.save_probabilities()
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def makedelayrb(delay, depth, realqc):
    cliffords = []
    depth -= 1
    for _ in range(depth):
        cliffords.append(random_clifford(1))

    qc = QuantumCircuit(2, 2)
    qc.name = '2 qubit delay-rb cicuit'
    for i in range(depth):
        qc.append(cliffords[i].to_instruction(), [0])
        qc.barrier()
        qc.delay(delay, unit="ns")
        qc.barrier()

    # make inverse clifford gate to return control qubit to |0>
    inverseclifford = makeinverseclifford(cliffords, depth)
    qc.append(inverseclifford, [0])
    if realqc is False:
        qc.save_probabilities()
    qc.measure(0, 0)
    qc.measure(1, 1)
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

    
def runmcmrb(realqc, depth, backend_name="", thermalerror = True):
    provider = IBMProvider()
    
    if realqc:
        backend = provider.get_backend(backend_name)
        transpiledqc = transpile(makemcmrb(depth, realqc), backend)
        job = backend.run(transpiledqc, shots=100)
        result = job.result()
    else:
        if thermalerror:
            backend = AerSimulator(noise_model=addthermalerror())
        else:
            backend = AerSimulator()
        transpiledqc = transpile(makemcmrb(depth, realqc), backend)
        job = backend.run(transpiledqc, shots=100)
        result = job.result()
        probabilities = result.results[0].data.probabilities
        print(f"probabilities {probabilities}")
        

def rundelayrb(realqc, depth, delay=0, backend_name="", thermalerror=True):
    provider = IBMProvider()

    if realqc:
        backend = provider.get_backend(backend_name)
        delay = backend.properties().to_dict().get('qubits')[0][-1]["value"]
        transpiledqc = transpile(makedelayrb(delay, depth, realqc), backend)
        job = backend.run(transpiledqc, shots=100)
        result = job.result()
        try:
            print(result.get_statevector())
        except:
            print("no statevector available")
        with open("./delayrbresults.json", "w") as outfile:
            json.dump(result.to_dict(), outfile)
        print(result.get_counts())

    else:
        if thermalerror:
            backend = AerSimulator(noise_model=addthermalerror())
        else:
            backend = AerSimulator()
        transpiledqc = transpile(makedelayrb(delay, depth, realqc), backend)
        job = backend.run(transpiledqc, shots=100)
        result = job.result()
        #try:
        #print(result.get_statevector())
        #except:
         #   print("no statevector available")
        #with open("./delayrbresults.json", "w") as outfile:
           # json.dump(result.to_dict(), outfile)
        print(result.to_dict())
        print(result.get_counts())
        probabilities = result.results[0].data.probabilities
        print(f"probabilities {probabilities}")


rundelayrb(False, 8)
