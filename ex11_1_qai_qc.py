import cirq

# 1) 양자비트와 양자회로 만들기

q = cirq.NamedQubit('My Qubit')
circuit = cirq.Circuit(cirq.measure(q))
print(circuit)

# 만들어진 양자회로를 시뮬레이션을 통해 어떤 결과가 만들어지는지 확인
simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=3)
print(m_outputs.measurements)


# 2) 입력을 반전시키는 양자 회로
q = cirq.NamedQubit('My Qubit')
circuit = cirq.Circuit(cirq.X(q), cirq.measure(q))
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
print(m_outputs.measurements['My Qubit'][:,0])


# 3) 두 상태를 중첩하는 양자회로
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


# 4) 두 개 양자비트를 위한 계산 예: CNOT 연산
# 두 양자비트의 초기 상태가 |00>인 경우
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

# 두 양자비트의 초기 상태가 |10>인 경우
circuit = cirq.Circuit(cirq.X(q[0]))
circuit.append(cirq.CNOT(q[0], q[1]))
circuit.append([cirq.measure(q[0]),cirq.measure(q[1])])
print(circuit)

simulator = cirq.Simulator()
m_outputs = simulator.run(circuit, repetitions=10)
print(m_outputs)


# 5) 벨 상태 만들기
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