import cirq

# 양자비트와 양자회로 만들기
q = cirq.NamedQubit('My Qubit')
circuit = cirq.Circuit(cirq.measure(q))
print(circuit)

# 만들어진 양자회로를 시뮬레이션을 통해 어떤 결과가 만들어지는지 확인
simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=3)
print(m_outputs.measurements)

q = cirq.NamedQubit('My Qubit')
circuit = cirq.Circuit(cirq.X(q), cirq.measure(q))
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
print(m_outputs.measurements['My Qubit'][:,0])

import numpy as np

q = cirq.NamedQubit('My Qubit')
circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
results = m_outputs.measurements['My Qubit'][:,0]
print('Results=',results,' Average=',np.mean(results))

# 충분히 반복하게되면 평균이 0.5에 더 가까워지는지 확인하기 위해 1000번 측정
m_outputs = simulator.run(circuit, repetitions=1000)
results = m_outputs.measurements['My Qubit'][:,0]
print('Average for 100 measurements=',np.mean(results))

q = [cirq.GridQubit(i, 0) for i in range(2)]
print(q[0], q[1])

circuit = cirq.Circuit()
circuit.append(cirq.CNOT(q[0], q[1]))
print(circuit)
circuit.append([cirq.measure(q[0]),cirq.measure(q[1])])
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
print(m_outputs)

circuit = cirq.Circuit(cirq.X(q[0]))
circuit.append(cirq.CNOT(q[0], q[1]))
circuit.append([cirq.measure(q[0]),cirq.measure(q[1])])
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
print(m_outputs)

q = [cirq.GridQubit(i, 0) for i in range(2)]
circuit = cirq.Circuit()
circuit.append(cirq.H(q[0]))
print(circuit)
circuit.append(cirq.CNOT(q[0], q[1]))
print(circuit)
circuit.append([cirq.measure(q[0]),cirq.measure(q[1])])
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
print(m_outputs)