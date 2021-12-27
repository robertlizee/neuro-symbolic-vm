# MIT License

# Copyright (c) 2021 Robert Lizee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
from numba import jit
from numba.typed.typedlist import List as NList
import numpy as np
from typing import List, Dict, Tuple, Union, Any, Callable
import random
import math

@jit(nopython=True)
def snn_clear_states(states, next_states):
    states[:] = 0.0
    next_states[:] = 0.0

@jit(nopython=True)
def snn_set_states(states, next_states, samples):
    states[:] = 0.0
    next_states[:] = 0.0
    for i in samples:
        states[i] = next_states[i] = 1.0

@jit(nopython=True)
def snn_swap_states(states, next_states, leaking_factor):
    N, = states.shape
    states[:] = next_states[:] 
    if leaking_factor > 0.0:
        r = 1.0 - leaking_factor
        for i in range(N):
            v = states[i]
            next_states[i] = 0.0 if v > 1.0 or v < 0.0 else r * v
    else:
        for i in range(N):
            v = states[i]
            next_states[i] = 0.0 if v > 1.0 or v < 0.0 else v

@jit(nopython=True)
def snn_compute_hash(states):
    N, = states.shape
    hash_value = 0

    for i in range(N):
        hash_value *= 2
        if states[i] >= 1.0:
            hash_value += 1
        hash_value %= 5368709075634563245
    
    return hash_value


@jit(nopython=True)
def snn_tick(input_states, weights, weights_output, output_next_states, strength):
    N, M = weights.shape
    
    for i in range(N):
        if input_states[i] >= 1.0:
            for j in range(M):
                output_next_states[weights_output[i, j]] += strength * weights[i, j]

@jit(nopython=True)
def snn_tick_second_order(input_states_A, input_states_B, second_order_connections, output_next_state, strength):
    N, K, _= second_order_connections.shape

    for i in range(N):
        for k in range(K):
            if input_states_A[second_order_connections[i, k, 0]] >= 1.0 and input_states_B[second_order_connections[i, k, 1]] >= 1.0:
                output_next_state[i] += strength
                break
    
@jit(nopython=True)
def snn_tick_very_sparse(input_states: np.array, weights: NList[Tuple[int, float, int]], output_next_states: np.array, strength: float):
    for (index_input, weight, index_output) in weights:
        if input_states[index_input] >= 1.0:
            output_next_states[index_output] += weight

def second_order_samples(samples_A, N_A, samples_B, N_B, second_order_connections, M):
    input_states_A = np.zeros(N_A)
    input_states_B = np.zeros(N_B)
    output_next_states = np.zeros(M)
    S, _ = samples_A.shape
    result = []
    k = 0

    for s in range(S):
        input_states_A[:] = 0.0
        input_states_B[:] = 0.0
        output_next_states[:] = 0.0
        input_states_A[samples_A[s, :]] = 1.0
        input_states_B[samples_B[s, :]] = 1.0
        snn_tick_second_order(input_states_A, input_states_B, second_order_connections, output_next_states, 1.0)
        output_indices, = output_next_states.nonzero()
        k = max(k, len(output_indices))
        result.append(output_indices)

    samples_output = -np.ones((S, k), dtype=int)

    for s in range(S):
        r = result[s]
        samples_output[s, 0:len(r)] = r

    return samples_output

@jit(nopython=True)
def snn_bind(input_states, weights, weights_output, output_states, value):
    N, M = weights.shape
    count = 0
    for i in range(N):
        if input_states[i] >= 1.0:
            for j in range(M):
                if output_states[weights_output[i, j]] >= 1.0:
                    weights[i, j] = value
                    count += 1

    return count

@jit(nopython=True)
def snn_tick_continuous(input_freq_indices, weights, weights_output, output_freq):
    N, K = weights.shape
    S, A = input_freq_indices.shape

    output_freq[:,:] = 0.0

    for s in range(S):
        for a in range(A):
            i = input_freq_indices[s, a]
            for k in range(K):

                output_freq[s, weights_output[i, k]] += weights[i, k]

@jit(nopython=True)
def snn_tick_stats(input_freq_indices, weights, weights_output, output_freq_indices, M):
    N, K = weights.shape
    S1, A1 = input_freq_indices.shape
    S2, A2 = output_freq_indices.shape

    output_freq = np.zeros(M)

    sum_in = 0.0
    sum_out = 0.0
    sum_2_in = 0.0
    sum_2_out = 0.0

    for s in range(S1):
        output_freq[:] = 0.0
        for a in range(A1):
            i = input_freq_indices[s, a]
            for k in range(K):
                output_freq[weights_output[i, k]] += weights[i, k]

        for a in range(A2):
            j = output_freq_indices[s, a]
            x = output_freq[j]
            output_freq[j] = 0.0
            sum_in += x
            sum_2_in += x*x

        for j in range(M):
            x = output_freq[j]
            sum_out += x
            sum_2_out += x*x

    count_in = S2 * A2
    count_out = S2 * (M - A2)

    avg_in = sum_in / count_in
    avg_out = sum_out / count_out

    avg_2_in = sum_2_in / count_in
    avg_2_out = sum_2_out / count_out

    std_in = math.sqrt(avg_2_in - avg_in * avg_in)
    std_out = math.sqrt(avg_2_out - avg_out * avg_out)

    return (avg_in, std_in, avg_out, std_out)

@jit(nopython=True)
def snn_best_freq(freq, target_freq_indices, l = -1) -> Tuple[int, float, int, float]:
    N = freq.shape
    S, K = target_freq_indices.shape

    l = S if l <= 0 else l
    invK = 1.0 / K

    best_score = 0.0
    best_s = 0
    second_best_score = 0.0
    second_best_s = 0

    for s in range(l):
        score = 0.0
        for k in range(K):
            if freq[target_freq_indices[s, k]] >= 1.0:
                score += invK

        if score > best_score:
            second_best_score = best_score
            second_best_s = best_s
            best_score = score
            best_s = s
        elif score > second_best_score:
            second_best_score = score
            second_best_s = s

    return (best_s, best_score, second_best_s, second_best_score)
        
@jit(nopython=True)
def snn_score_of(freq, target_freq_indices, s) -> float:
    N = freq.shape
    S, K = target_freq_indices.shape

    invK = 1.0 / K

    score = 0.0
    for k in range(K):
        if freq[target_freq_indices[s, k]] >= 1.0:
            score += invK

    return score       

@jit(nopython=True)
def snn_compute_relevant_gradient_and_cost(freq, desired_freq_indices, relevant_gradient, min_value = -0.1):
    S, N = freq.shape
    _, K = desired_freq_indices.shape

    relevant_gradient[:,:] = 0.0

    for s in range(S):
        for i in range(N):
            f = freq[s, i]
            if f > min_value:
                d = f - min_value
                relevant_gradient[s, i] = 2.0 * d

        for k in range(K):
            i = desired_freq_indices[s, k]
            f = freq[s, i]
            if f < 1.0:
                d = f - 1.0
                relevant_gradient[s, i] = 2.0 * d
            else:
                relevant_gradient[s, i] = 0.0
   
    cost = 0.0

    for s in range(S):
        for i in range(N):
            rg = relevant_gradient[s, i]
            cost += rg * rg


    return 0.25 * cost


@jit(nopython=True)
def snn_compute_weight_gradient(input_freq_indices, weights, weights_output, output_relevant_gradient, weights_gradient):
    N, M = weights.shape

    S, K = input_freq_indices.shape

    weights_gradient[:, :] = 0.0
    
    for k in range(K):
        for s in range(S):
            i = input_freq_indices[s, k]
            for j in range(M):
                weights_gradient[i, j] += output_relevant_gradient[s, weights_output[i, j]]

@jit(nopython=True)
def update_gradient(weights, weights_gradient, cost):
    N, M = weights.shape
    impact = 0.0
    for i in range(N):
        for j in range(M):
            g = weights_gradient[i, j]
            impact += g*g

    if impact > 0.0:
        f = cost / impact

        for i in range(N):
            for j in range(M):
                weights[i, j] -= f * weights_gradient[i, j]

def optimize(input_freq_indices, weights, weights_output, output_desired_freq_indices, factor, output_freq, output_relevant_gradient, weights_gradient, min_value = -0.1): 
    snn_tick_continuous(input_freq_indices, weights, weights_output, output_freq)
    cost = snn_compute_relevant_gradient_and_cost(output_freq, output_desired_freq_indices, output_relevant_gradient, min_value)
    snn_compute_weight_gradient(input_freq_indices, weights, weights_output, output_relevant_gradient, weights_gradient)
    update_gradient(weights, weights_gradient, factor*cost)

    return cost

def computed_costs(input_freq_indices, weights, weights_output, output_desired_freq_indices, output_freq, output_relevant_gradient, min_value = -0.1): 
    snn_tick_continuous(input_freq_indices, weights, weights_output, output_freq)
    snn_compute_relevant_gradient_and_cost(output_freq, output_desired_freq_indices, output_relevant_gradient, min_value)

    return np.sum(output_relevant_gradient * output_relevant_gradient, axis=1)

def make_random_samples(S, N, k):
    result = np.zeros((S, k), dtype=int)
    for s in range(S):
        result[s,:] = np.random.randint(0, N, k)
    return result

class Network(object):
    def __init__(self, layers: List[Layer], connections: List[BaseConnection], sources: List[Source] = []):
        self.layers: List[Layer] = layers
        self.connections: List[BaseConnection] = connections
        weights_connections: List[Any] = [connection for connection in connections if connection is Connection]
        weights_connections_typed: List[Connection] = weights_connections
        self.weights = set([connection.weights for connection in weights_connections_typed])
        self.sources: List[Source] = sources

    def clear_states(self):
        for layer in self.layers:
            layer.clear_states()

    def tick(self):
        for weights in self.weights:
            weights.tick()

        for layer in self.layers:
            layer.swap_states()

        for source in self.sources:
            source.tick()
    
        for connection in self.connections:
            connection.tick()

class Source(object):
    def __init__(self, value, layer):
        self.value = value
        self.layer = layer

    def tick(self):
        self.layer.next_states += self.value

class Layer(object):

    frozen = False
    
    def __init__(self, n : int, leaking_factor = 0.0, name: str = None):
        self.name = name
        self.n : int = n
        self.states : np.array = np.zeros(n)
        self.next_states : np.array = np.zeros(n)
        self.key : int = 0
        self.binding = 0
        self.unbinding = 0
        self.leaking_factor = leaking_factor

    def change_n(self, n: int):
        self.n : int = n
        self.states : np.array = np.zeros(n)
        self.next_states : np.array = np.zeros(n)


    def clear_states(self):      
        snn_clear_states(self.states, self.next_states)
        self.key = 0

    def swap_states(self):
        if not self.frozen:
            snn_swap_states(self.states, self.next_states, self.leaking_factor)
            self.key = snn_compute_hash(self.states)
            self.binding = 2 if self.binding == 1 else 0
            self.unbinding = 2 if self.unbinding == 1 else 0

    def bind(self):
        self.binding = 1

    def unbind(self):
        self.unbinding = 1

    def is_binding(self):
        return self.binding > 0

    def is_unbinding(self):
        return self.unbinding > 0

    def init_states_to_one(self):
        self.clear_states()
        self.states[:] = self.next_states[:] = 1.0
        self.key = snn_compute_hash(self.states)  

    def set_states(self, samples):
        snn_set_states(self.states, self.next_states, samples)
        self.key = snn_compute_hash(self.states)

    def inhibit_layer(self):
        self.next_states[:] -= 2.0

class BaseConnection(object):
    opened: bool = True

    def tick(self):
        pass
class Connection(BaseConnection):

    def __init__(self, weights: ConnectionWeights, from_layer: Layer, to_layer: Layer, strength: float = 1.0):
        M, = to_layer.states.shape
        self.M = M
        self.weights = weights
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.strength = strength

    def tick(self):
        if self.opened and not self.to_layer.frozen:
            if self.from_layer.key != 0:
                cache = self.weights.get_cache(self.from_layer.key)
                if cache is None:
                    cache = np.zeros(self.M)
                    snn_tick(self.from_layer.states, self.weights.weights, self.weights.weights_output, cache, self.strength)
                    self.weights.cache(self.from_layer.key, cache)

                self.to_layer.next_states += cache

    def bind(self, value = 1.0):
        count = snn_bind(self.from_layer.states, self.weights.weights, self.weights.weights_output, self.to_layer.states, value)
        self.weights.invalidate_cache()
        return count

    def unbind(self):
        count = snn_bind(self.from_layer.states, self.weights.weights, self.weights.weights_output, self.to_layer.states, 0.0)
        self.weights.invalidate_cache()
        return count

class BindingConnection(Connection):
    def tick(self):
        super().tick()

        if (self.from_layer.is_binding()):
            count = self.bind()
            #print("binding ", count)
        elif (self.from_layer.is_unbinding()):
            count = self.unbind()
            #print("unbinding", count)

class SecondOrderConnection(BaseConnection):
    def __init__(self, from_layer1: Layer, k1: int, from_layer2: Layer, k2: int, to_layer: Layer, l: int, strength = 1.0):

        N1, = from_layer1.states.shape
        N2, = from_layer2.states.shape
        M, = to_layer.states.shape

        self.from_layer1 = from_layer1
        self.from_layer2 = from_layer2
        self.to_layer = to_layer
        self.strength = strength
        self.key1 = 0
        self.key2 = 0
        self.cache = np.zeros(M)

        k = (l * N1 * N2) // (M * k1 * k2)
        self.second_order_connections = np.zeros((M, k, 2), dtype=int)
        self.second_order_connections[:,:,0] = np.random.randint(0, N1, (M, k))
        self.second_order_connections[:,:,1] = np.random.randint(0, N2, (M, k))

    def tick(self):
        if self.opened and not self.to_layer.frozen:
            if self.from_layer1.key != 0 and self.from_layer2.key != 0:
                if self.from_layer1.key != self.key1 or self.from_layer2.key != self.key2:
                    self.cache[:] = 0.0
                    snn_tick_second_order(self.from_layer1.states, self.from_layer2.states, self.second_order_connections, self.cache, self.strength)
                    self.key1 = self.from_layer1.key
                    self.key2 = self.from_layer2.key
            
                self.to_layer.next_states += self.cache

class ConnectionWeights(object):
    def __init__(self, M, N, k : int):
        self.M = M
        self.N = N
        self.weights_output = np.zeros((M, k), dtype=int)

        for i in range(M):
            self.weights_output[i,:] = np.random.permutation(N)[0:k]

        self.weights : np.array = np.zeros(self.weights_output.shape)

        self.current_cache : Dict[int, np.array] = {}
        self.previous_cache : Dict[int, np.array] = {}
    
    def train(self, 
            input_samples: PrimeAttractors, 
            output_samples: PrimeAttractors, 
            factor, 
            output_function, 
            mapping: Dict[str, str] = None, 
            min_value=-0.1):

        if mapping == None:
            input = input_samples.samples
            output = output_samples.samples
        else:
            map_input = [input_samples.name_to_index[name] for name in mapping]
            map_output = [output_samples.name_to_index[mapping[name]] for name in mapping]

            input = input_samples.samples[map_input, :]
            output = output_samples.samples[map_output, :]
            
            pass

        S, K = output.shape
        N = self.N
        output_values = np.zeros((S, N))
        output_relevant_gradient = np.zeros((S, N))
        weights_gradient = np.zeros(self.weights.shape)

        for _ in range(100000000):
            cost = (1.0 / (S * K)) * optimize(input, self.weights, self.weights_output, output, factor, output_values, output_relevant_gradient, weights_gradient, min_value)
            if output_function(cost):
                return (1.0 / K) * computed_costs(input, self.weights, self.weights_output, output, output_values, output_relevant_gradient)

        return (1.0 / K) * computed_costs(input, self.weights, self.weights_output, output, output_values, output_relevant_gradient)

    #def second_order_samples(samples_A, N_A, samples_B, N_B, second_order_connections, M):

    def train_second_order(self, 
            input1_samples: PrimeAttractors, 
            input2_samples: PrimeAttractors, 
            second_order_connections: SecondOrderConnection, 
            output_samples: PrimeAttractors, 
            factor, 
            output_function, 
            mapping: Dict[Tuple[str, str], str], 
            min_value=-0.1):

        M, _ = self.weights.shape

        map_input1 = [input1_samples.name_to_index[name] for (name, _) in mapping]
        map_input2 = [input2_samples.name_to_index[name] for (_, name) in mapping]
        map_output = [output_samples.name_to_index[mapping[name]] for name in mapping]

        hash_samples = second_order_samples(
            input1_samples.samples[map_input1, :], 
            input1_samples.N,
            input2_samples.samples[map_input2, :], 
            input2_samples.N,
            second_order_connections.second_order_connections,
            M)

        S, k = hash_samples.shape

        return self.train(
            PrimeAttractors(0, M, k, [], hash_samples), 
            PrimeAttractors(0, output_samples.N, output_samples.k, [], output_samples.samples[map_output, :]), 
            factor, 
            output_function, 
            None, 
            min_value)

    def stats(self, input_samples: PrimeAttractors, output_samples: PrimeAttractors, M):
        return snn_tick_stats(input_samples.samples, self.weights, self.weights_output, output_samples.samples, M)

    def tick(self):
        self.previous_cache = self.current_cache
        self.current_cache = {}

    def cache(self, key: int, value: np.array):
        self.current_cache[key] = value

    def get_cache(self, key: int) -> Union[np.array, None]:
        if key in self.current_cache:
            return self.current_cache[key]
        elif key in self.previous_cache:
            value = self.previous_cache[key]
            self.current_cache[key] = value
            return value
        else:
            return None

    def invalidate_cache(self):
        self.previous_cache = {}
        self.current_cache = {}

class PrimeAttractors(object):
    def __init__(self, S: int, N: int, k: int, names: List[str] = [], samples = None):
        self.N = N
        self.k = k

        if samples is None:
            self.samples = make_random_samples(S + len(names), N, k)
        else:
            self.samples = samples

        self.names = names
        self.name_to_index = { names[i]: i for i in range(len(names))}

    def best_named_attractor(self, layer: Layer) -> Tuple[str, float, str, float]:
        first_s, first_score, second_s, second_score = snn_best_freq(layer.states, self.samples, len(self.names))

        return (self.names[first_s], first_score, self.names[second_s], second_score)

    def score_of(self, layer: Layer, name: str) -> float:
        return snn_score_of(layer.states, self.samples, self.name_to_index[name])

    def init_states(self, layer: Layer, name: Union[str, int]):
        sample = self.samples[name if isinstance(name, int) else self.name_to_index[name], :]
        layer.set_states(sample)

    def very_sparse_weights_to_recognize_input(self, name: Union[str, int], output_index: int, output_strength: float):
        sample = self.samples[name if isinstance(name, int) else self.name_to_index[name], :]
        k, = sample.shape
        s = output_strength / k
        return [(input_index, s, output_index) for input_index in sample]
        
class ActionConnection(Connection):
    def __init__(self, from_layer: Layer, actions: List[Tuple[Layer, Callable[[Layer], None]]]):
        self.from_layer = from_layer
        self.actions = actions

    def set_actions(self, actions: List[Tuple[Layer, Callable[[Layer], None]]]):
        self.actions = actions

    def tick(self):
        states = self.from_layer.states

        for i in range(len(self.actions)):
            if states[i] >= 1.0:
                layer, action = self.actions[i]
                action(layer)

class VerySparseConnection(Connection):
    def __init__(self, from_layer: Layer, weights: List[Tuple[int, float, int]], to_layer: Layer, strength = 1.0):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.strength = strength
        self.set_weights(weights)

    def tick(self):
        snn_tick_very_sparse(self.from_layer.states, self.weights, self.to_layer.next_states, self.strength)

    def set_weights(self, weights: List[Tuple[int, float, int]]):
        self.weights = NList()
        for a in weights:
            self.weights.append(a)
