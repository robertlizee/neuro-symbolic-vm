# Copyright (c) 2021 Robert Lizee

from __future__ import annotations
from typing import List, Dict, Tuple, Union, Any, Callable
from random import randrange

from VMdef import *
from NN import *

class VM:
    instance: VM
    
    executers: Dict[str, Callable[[], None]] = {}
    values: PrimeAttractors

    input_string = ""
    input_index = 0
    output_string = ""

    def __init__(self, code: List[Instruction], debug = False) -> None:
        VM.instance = self

        self.code = code

        #neurons_per_layer = 10000
        #neurons_in_attractor = 40
        #fan_out = 10000
        neurons_per_layer = 10000
        neurons_in_attractor = 30
        fan_out = 3000
        additional_samples = 300
        additional_opcode_samples = 60

        self_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        self_opcode_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)

        class RegisterImpl:
            def __init__(self, name: str):
                self.layer_input = Layer(neurons_per_layer, 0.0, name + "_input")
                self.layer_output = Layer(neurons_per_layer, 0.0, name + "_input")
                self.layer = Layer(neurons_per_layer, 0.0, name)
                self.self_connection = Connection(self_weights, self.layer, self.layer, 1.5)
                self.self_input_connection = Connection(self_weights, self.layer_input, self.layer, 1.5)
                self.self_output_connection = Connection(self_weights, self.layer, self.layer_output, 1.5)
    
        r_cons = RegisterImpl("cons")
        r_car = RegisterImpl("car")
        r_cdr = RegisterImpl("cdr")
        r_stack = RegisterImpl("stack")
        r_alloc = RegisterImpl("alloc")
        r_alloc_next = RegisterImpl("alloc_next")
        r_cont = RegisterImpl("cont")
        r_table = RegisterImpl("table")
        r_key = RegisterImpl("key")
        r_value = RegisterImpl("value")
        r_r1 = RegisterImpl("r1")
        r_r2 = RegisterImpl("r2")
        r_reserved = RegisterImpl("reserved")
        r_arg = RegisterImpl("arg")
        r_code = RegisterImpl("code")
        r_code_next = RegisterImpl("code_next")

        registers: Dict[Register, RegisterImpl] = { 
            r.cons: r_cons,
            r.car: r_car,
            r.cdr: r_cdr,
            r.stack: r_stack,
            r.alloc: r_alloc,
            r.alloc_next: r_alloc_next,
            r.cont: r_cont,
            r.table: r_table,
            r.key: r_key,
            r.value: r_value,
            r.r1: r_r1,
            r.r2: r_r2,
            r.reserved: r_reserved,
            r.arg: r_arg,
            r.code: r_code,
            r.code_next: r_code_next
        }

        self.registers = registers

        registers_output16 = [registers[key].layer_output for key in registers]
        registers_output8 = [Layer(neurons_per_layer) for _ in range(8)]
        registers_16to8 = [Connection(self_weights, registers_output16[i], registers_output8[i//2], 1.5) for i in range(16)]
        registers_output4 = [Layer(neurons_per_layer) for _ in range(4)]
        registers_8to4 = [Connection(self_weights, registers_output8[i], registers_output4[i//2], 1.5) for i in range(8)]
        registers_output2 = [Layer(neurons_per_layer) for _ in range(2)]
        registers_4to2 = [Connection(self_weights, registers_output4[i], registers_output2[i//2], 1.5) for i in range(4)]
        registers_io = Layer(neurons_per_layer)
        registers_2to1 = [Connection(self_weights, registers_output2[i], registers_io, 1.5) for i in range(2)]
        registers_input2 = [Layer(neurons_per_layer) for _ in range(2)]
        registers_1to2 = [Connection(self_weights, registers_io, registers_input2[i], 1.5) for i in range(2)]
        registers_input4 = [Layer(neurons_per_layer) for _ in range(4)]
        registers_2to4 = [Connection(self_weights, registers_input2[i//2], registers_input4[i], 1.5) for i in range(4)]
        registers_input8 = [Layer(neurons_per_layer) for _ in range(8)]
        registers_4to8 = [Connection(self_weights, registers_input4[i//2], registers_input8[i], 1.5) for i in range(8)]
        registers_input16 = [registers[key].layer_input for key in registers]
        registers_8to16 = [Connection(self_weights, registers_input8[i//2], registers_input16[i], 1.5) for i in range(16)]

        opcode_layer = Layer(neurons_per_layer)
        opcode_eq_layer = Layer(neurons_per_layer)
        opcode_neq_layer = Layer(neurons_per_layer)

        self.opcode_layer = opcode_layer

        microinstruction_layer = Layer(0, 0.5)
        microinstruction_self_connection = VerySparseConnection(microinstruction_layer, [], microinstruction_layer)
        opcode_to_microinstruction_connection = VerySparseConnection(opcode_layer, [], microinstruction_layer)
        microinstruction_action_layer = Layer(0, 0.5)
        microinstruction_action_connection = ActionConnection(microinstruction_action_layer, [])
        microinstruction_to_microinstruction_action_connection = VerySparseConnection(microinstruction_layer, [], microinstruction_action_layer)

        cons_control_layer = Layer(neurons_per_layer)
        cons_to_cons_control_connection = Connection(self_weights, r_cons.layer, cons_control_layer, 1.5)

        cons_control_to_car_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        cons_control_to_car_connection = BindingConnection(cons_control_to_car_weights, cons_control_layer, r_car.layer, 0.2 * neurons_per_layer / (neurons_in_attractor * fan_out))

        cons_control_to_cdr_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        cons_control_to_cdr_connection = BindingConnection(cons_control_to_cdr_weights, cons_control_layer, r_cdr.layer, 0.2 * neurons_per_layer / (neurons_in_attractor * fan_out))

        alloc_control_layer = Layer(neurons_per_layer)
        alloc_to_alloc_control_connection = Connection(self_weights, r_alloc.layer, alloc_control_layer, 1.5)
        alloc_control_to_alloc_next_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        alloc_control_to_alloc_next_connection = BindingConnection(alloc_control_to_alloc_next_weights, alloc_control_layer, r_alloc_next.layer, 0.2 * neurons_per_layer / (neurons_in_attractor * fan_out))
        
        hash_layer = Layer(neurons_per_layer)
        table_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        default_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        hashing_connection = SecondOrderConnection(r_table.layer, neurons_in_attractor, r_key.layer, neurons_in_attractor, hash_layer, neurons_in_attractor)
        table_connection = BindingConnection(table_weights, hash_layer, r_value.layer, 0.2 * neurons_per_layer / (neurons_in_attractor * fan_out))
        hash_to_value = Connection(default_weights, hash_layer, r_value.layer, 0.075 * neurons_per_layer / (neurons_in_attractor * fan_out))

        code_control_layer = Layer(neurons_per_layer)
        code_to_code_control_connection = Connection(self_weights, r_code.layer, code_control_layer, 1.5)

        code_control_to_code_next_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        code_control_to_code_next_connection = Connection(code_control_to_code_next_weights, code_control_layer, r_code_next.layer, 1.5)

        code_control_to_arg_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        code_control_to_arg_connection = Connection(code_control_to_arg_weights, code_control_layer, r_arg.layer, 1.5)

        code_control_to_opcode_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        code_control_to_opcode_neq_weights = ConnectionWeights(neurons_per_layer, neurons_per_layer, fan_out)
        code_control_to_opcode_eq_connection = Connection(code_control_to_opcode_weights, code_control_layer, opcode_eq_layer, 1.5)
        code_control_to_opcode_neq_connection = Connection(code_control_to_opcode_neq_weights, code_control_layer, opcode_neq_layer, 1.5)
        opcode_eq_to_opcode_connection = Connection(self_opcode_weights, opcode_eq_layer, opcode_layer, 1.5)
        opcode_neq_to_opcode_connection = Connection(self_opcode_weights, opcode_neq_layer, opcode_layer, 1.5)

        self.code_control_to_opcode_eq_connection = code_control_to_opcode_eq_connection
        self.code_control_to_opcode_neq_connection = code_control_to_opcode_neq_connection

        mov_code_next_to_code = op.mov(r.code_next, r.code)

        def layer_name(layer: Layer):
            if layer.name is None:
                return "r" + str(control_and_register_layer_index[layer])
            else:
                return layer.name

        def nop_layer(layer: Layer):
            pass

        def inhibit_layer(layer: Layer):
            layer.inhibit_layer()

        def bind_layer(layer: Layer):
            layer.bind()

        def unbind_layer(layer: Layer):
            layer.unbind()

        def read_layer(layer: Layer):
            if self.input_index < len(self.input_string):
                c = ord(self.input_string[self.input_index])
                #if debug:
                print("Read ", chr(c))
                self.input_index += 1
                c0 = ord('0')
                c9 = ord('9')
                if c >= c0 and c <= c9:
                    values.init_states(layer, t.digits[c - c0].name)
                else:
                    values.init_states(layer, t.unknown.name)
            else:
                #if debug:
                print("Read eol")
                values.init_states(layer, t.false.name)

            if debug:
                (v, sv, v2, sv2) = values.best_named_attractor(layer)
                print("read {4}: {0} ({1}) - {2} ({3})".format(v, sv, v2, sv2, layer_name(layer)))
                
        def write_layer(layer: Layer):
            (v, sv, v2, sv2) = values.best_named_attractor(layer)

            #if debug:
            print("write {4}: {0} ({1}) - {2} ({3})".format(v, sv, v2, sv2, layer_name(layer)))

            for i in range(10):
                if t.digits[i].name == v:
                    self.output_string += chr(ord('0') + i)
                    return
            self.output_string += ':'

        nb_ticks = 0

        def debug_layer(layer: Layer):
            nonlocal nb_ticks
            print("debug {0}-{1}".format(nb_ticks, control_and_register_layer_index[layer]))

        exit_called = False

        def exit_layer(layer: Layer):
            nonlocal exit_called
            exit_called = True

        test_layer = Layer(4, 0.5)
        self_test_connection = VerySparseConnection(test_layer, [(0, 1.0, 0), (1, 1.0, 2), (0, 1.0, 3), (1, -1.0, 3)], test_layer)
        test_layer.next_states[0] = 1.0

        test_value_connection = VerySparseConnection(r_value.layer, [], test_layer)

        test_action_connection = ActionConnection(test_layer, [
            (test_layer, nop_layer), 
            (test_layer, nop_layer), 
            (opcode_eq_layer, inhibit_layer), 
            (opcode_neq_layer, inhibit_layer)
        ])

        connections: List[BaseConnection] = [registers[register].self_connection for register in registers]

        connections += [registers[register].self_output_connection for register in registers]
        connections += [registers[register].self_input_connection for register in registers]
        connections += registers_16to8 + registers_8to4 + registers_4to2 + registers_2to1
        connections += registers_1to2 + registers_2to4 + registers_4to8 + registers_8to16
        connections += [cons_to_cons_control_connection, alloc_to_alloc_control_connection, code_to_code_control_connection]
        connections += [self_test_connection, test_value_connection, test_action_connection]

        connections += [cons_control_to_car_connection, cons_control_to_cdr_connection, alloc_control_to_alloc_next_connection,
                hashing_connection, table_connection, hash_to_value, code_control_to_code_next_connection,
                code_control_to_arg_connection, code_control_to_opcode_eq_connection, code_control_to_opcode_neq_connection,
                opcode_eq_to_opcode_connection, opcode_neq_to_opcode_connection, opcode_to_microinstruction_connection,
                microinstruction_self_connection, microinstruction_action_connection, microinstruction_to_microinstruction_action_connection]

        all_control_layers = registers_output16 + registers_input16 + [cons_control_layer, alloc_control_layer, code_control_layer, hash_layer]

        control_and_register_layers = all_control_layers + [registers[register].layer for register in registers]

        control_and_register_layer_index: Dict[Layer, int] = {control_and_register_layers[i]: i for i in range(len(control_and_register_layers))}

        network = Network(
            [registers[register].layer for register in registers] + 
            registers_output16 + registers_output8 + registers_output4 + registers_output2 +
            [registers_io] +
            registers_input2 + registers_input4 + registers_input8 + registers_input16 + 
            [cons_control_layer, alloc_control_layer, code_control_layer] +
            [hash_layer, opcode_layer, opcode_eq_layer, opcode_neq_layer, test_layer, microinstruction_layer, microinstruction_action_layer],
            connections
        )

        exit_label = Label(None)
        code_to_code_next_dict: Dict[str, str] = {}
        code_to_arg_dict: Dict[str, str] = {}
        code_to_opcode_dict: Dict[str, str] = {}
        code_to_opcode_neq_dict: Dict[str, str] = {}

        code = code + [(exit_label, op.exit, op.exit, exit_label), (Label(None), op.nop, op.nop, None)]

        self.code = code

        for i in range(len(code)-1):
            label, opcode, opcode_neg, arg = code[i]
            label_next, _, _, _ = code[i+1]

            code_to_code_next_dict[label.name] = label_next.name
            if not arg is None:
                code_to_arg_dict[label.name] = arg.name
            code_to_opcode_dict[label.name] = opcode.name
            code_to_opcode_neq_dict[label.name] = opcode_neg.name

        last_label = code[len(code) - 2][0]
        code_to_code_next_dict[last_label.name] = last_label.name

        values = PrimeAttractors(
            additional_samples, 
            neurons_per_layer, 
            neurons_in_attractor, 
            [token.name for token in t.tokens] + [label for label in code_to_code_next_dict]
        )

        def set_register(register: RegisterImpl, value: str):
            values.init_states(register.layer, value)

        def training_output(cost):
            print(str(100.0 * cost), flush=True)
            return 100.0 * cost < 0.2

        print("Training values")
        costs = self_weights.train(values, values, 0.2, training_output, min_value=-0.3)
        for i in range(20):
            e = i / 20
            if np.sum(100.0*costs > e) <= additional_samples:
                values.samples = values.samples[100.0*costs <= e, :]
                break

        self.values = values

        opcodes = PrimeAttractors(additional_opcode_samples, neurons_per_layer, neurons_in_attractor, 
            [opcode.name for opcode in op.get_all_opcodes()]
        )

        self.opcodes = opcodes



        print("Training code_to_code_next_weights")
        costs = code_control_to_code_next_weights.train(values, values, 0.2, training_output, min_value=-0.3, mapping=code_to_code_next_dict)

        print("Training code_to_arg_weights")
        costs = code_control_to_arg_weights.train(values, values, 0.2, training_output, min_value=-0.3, mapping=code_to_arg_dict)

        print("Training self_opcode_weights")
        costs = self_opcode_weights.train(opcodes, opcodes, 0.2, training_output, min_value=-0.3)
        for i in range(20):
            e = i / 20
            if np.sum(100.0*costs > e) <= additional_opcode_samples:
                opcodes.samples = opcodes.samples[100.0*costs <= e, :]
                break

        print("Training code_to_opcode_weights")
        costs = code_control_to_opcode_weights.train(values, opcodes, 0.2, training_output, min_value=-0.3, mapping=code_to_opcode_dict)

        print("Training code_to_opcode_neq_weights")
        costs = code_control_to_opcode_neq_weights.train(values, opcodes, 0.2, training_output, min_value=-0.3, mapping=code_to_opcode_neq_dict)


        microinstruction_current_index = 0
        microinstruction_next_index = 0
        microinstruction_self_weights: List[Tuple[int, float, int]] = []
        opcode_to_microinstruction_weights: List[Tuple[int, float, int]] = []
        microinstruction_to_microinstruction_action_weights: List[Tuple[int, float, int]] = []

        microinstruction_action_connection.set_actions(
            [(layer, inhibit_layer) for layer in control_and_register_layers] +
            [(layer, bind_layer) for layer in control_and_register_layers] +
            [(layer, unbind_layer) for layer in control_and_register_layers] +
            [(layer, read_layer) for layer in control_and_register_layers] +
            [(layer, write_layer) for layer in control_and_register_layers] +
            [(layer, debug_layer) for layer in control_and_register_layers] +
            [(control_and_register_layers[0], exit_layer)]
        )

        microinstruction_action_layer.change_n(6 * len(control_and_register_layers) + 1)


        def microinstruction_start():
            nonlocal microinstruction_current_index
            nonlocal microinstruction_next_index

            index = microinstruction_current_index = microinstruction_next_index
            microinstruction_self_weights.append((index, 1.0, index+1))
            microinstruction_self_weights.append((index, 1.0, index+2))
            microinstruction_self_weights.append((index+1, -1.0, index+2))
            microinstruction_next_index = index + 2

        def microinstruction_next():
            nonlocal microinstruction_current_index
            nonlocal microinstruction_next_index

            index = microinstruction_current_index = microinstruction_next_index
            microinstruction_self_weights.append((index, 1.0, index+1))
            microinstruction_next_index = index + 1

        def microinstruction_stop():
            nonlocal microinstruction_current_index
            nonlocal microinstruction_next_index

            microinstruction_next_index += 1
            microinstruction_current_index = microinstruction_next_index

        def microinstruction_return():
            nonlocal microinstruction_current_index
            nonlocal microinstruction_next_index

            microinstruction_self_weights.append((microinstruction_current_index, 1.0, 0))
            for layer in all_control_layers:
                microinstruction_inhibit_layer(layer)

            microinstruction_next_index += 1
            microinstruction_current_index = microinstruction_next_index

        def microinstruction_tick(steps: int, layer_non_inhibited: List[Layer]):
            for _ in range(steps):
                for layer in all_control_layers:
                    if not layer in layer_non_inhibited:
                        microinstruction_inhibit_layer(layer)
                microinstruction_next()

        def microinstruction_start_opcode(opcode: Opcode):
            nonlocal opcode_to_microinstruction_weights
            
            microinstruction_start()
            opcode_to_microinstruction_weights += opcodes.very_sparse_weights_to_recognize_input(opcode.name, microinstruction_current_index, 1.5)

            # wait for the parallel microcode to stop
            microinstruction_tick(4, []) 
            
            # move code_next to code
            #microinstruction_debug(5)
            microinstruction_tick(20, [r_code_next.layer_output, r_code.layer_input])
            microinstruction_inhibit_layer(r_code.layer)
            microinstruction_tick(4, [r_code_next.layer_output, r_code.layer_input])

        def microinstruction_inhibit_layer(layer: Layer):
            microinstruction_to_microinstruction_action_weights.append((
                                        microinstruction_current_index, 
                                        1.0, 
                                        control_and_register_layer_index[layer]))

        def microinstruction_bind(input_layer: Layer):
            microinstruction_to_microinstruction_action_weights.append(
                                        (microinstruction_current_index, 
                                        1.0, 
                                        len(control_and_register_layers) + control_and_register_layer_index[input_layer]))

        def microinstruction_unbind(input_layer: Layer):
            microinstruction_to_microinstruction_action_weights.append(
                                        (microinstruction_current_index, 
                                        1.0, 
                                        2 * len(control_and_register_layers) + control_and_register_layer_index[input_layer]))

        def microinstruction_read(layer: Layer):
            microinstruction_to_microinstruction_action_weights.append(
                                        (microinstruction_current_index, 
                                        1.0, 
                                        3 * len(control_and_register_layers) + control_and_register_layer_index[layer]))

        def microinstruction_write(layer: Layer):
            microinstruction_to_microinstruction_action_weights.append(
                                        (microinstruction_current_index, 
                                        1.0, 
                                        4 * len(control_and_register_layers) + control_and_register_layer_index[layer]))


        def microinstruction_debug(index: int):
            microinstruction_to_microinstruction_action_weights.append(
                                        (microinstruction_current_index, 
                                        1.0, 
                                        5 * len(control_and_register_layers) + control_and_register_layer_index[control_and_register_layers[index]]))

        def microinstruction_exit():
            microinstruction_to_microinstruction_action_weights.append(
                                        (microinstruction_current_index, 
                                        1.0, 
                                        6 * len(control_and_register_layers)))




        #instruction 0
        microinstruction_start()
        #microinstruction_debug(1)
        #microinstruction_write(r_code.layer)
        microinstruction_tick(6, [])
        #microinstruction_debug(2)
        microinstruction_inhibit_layer(r_code_next.layer)
        microinstruction_inhibit_layer(r_arg.layer)
        microinstruction_tick(3, [code_control_layer])
        #microinstruction_debug(3)
        #microinstruction_write(r_code_next.layer)
        microinstruction_tick(4, [code_control_layer])
        #microinstruction_debug(4)
        microinstruction_stop()

        microinstruction_start_opcode(op.alloc_recall)
        microinstruction_inhibit_layer(r_alloc_next.layer)
        microinstruction_tick(1, [alloc_control_layer])
        microinstruction_inhibit_layer(r_alloc_next.layer)
        microinstruction_tick(1, [alloc_control_layer])
        microinstruction_inhibit_layer(r_alloc_next.layer)
        microinstruction_tick(1, [alloc_control_layer])
        microinstruction_inhibit_layer(r_alloc_next.layer)
        microinstruction_tick(25, [alloc_control_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.alloc_bind)
        microinstruction_tick(5, [alloc_control_layer])
        microinstruction_bind(alloc_control_layer)
        microinstruction_tick(1, [alloc_control_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.alloc_unbind)
        microinstruction_tick(5, [alloc_control_layer])
        microinstruction_unbind(alloc_control_layer)
        microinstruction_tick(1, [alloc_control_layer])
        microinstruction_return()
        
        microinstruction_start_opcode(op.cons_recall)
        microinstruction_inhibit_layer(r_car.layer)
        microinstruction_inhibit_layer(r_cdr.layer)            
        microinstruction_tick(1, [cons_control_layer])
        microinstruction_inhibit_layer(r_car.layer)
        microinstruction_inhibit_layer(r_cdr.layer)            
        microinstruction_tick(1, [cons_control_layer])
        microinstruction_inhibit_layer(r_car.layer)
        microinstruction_inhibit_layer(r_cdr.layer)            
        microinstruction_tick(1, [cons_control_layer])
        microinstruction_inhibit_layer(r_car.layer)
        microinstruction_inhibit_layer(r_cdr.layer)            
        microinstruction_tick(25, [cons_control_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.cons_bind)
        microinstruction_tick(5, [cons_control_layer])
        microinstruction_bind(cons_control_layer)
        microinstruction_tick(1, [cons_control_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.cons_unbind)
        microinstruction_tick(5, [cons_control_layer])
        microinstruction_unbind(cons_control_layer)
        microinstruction_tick(1, [cons_control_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.table_recall)
        microinstruction_inhibit_layer(hash_layer)
        microinstruction_inhibit_layer(r_value.layer)            
        microinstruction_tick(1, [hash_layer])
        microinstruction_inhibit_layer(hash_layer)
        microinstruction_inhibit_layer(r_value.layer)            
        microinstruction_tick(1, [hash_layer])
        microinstruction_inhibit_layer(hash_layer)
        microinstruction_inhibit_layer(r_value.layer)            
        microinstruction_tick(25, [hash_layer])
        #microinstruction_debug(2)
        #microinstruction_write(r_table.layer)
        #microinstruction_write(r_key.layer)
        #microinstruction_write(r_value.layer)
        microinstruction_tick(1, [])
        microinstruction_return()

        microinstruction_start_opcode(op.table_bind)
        microinstruction_tick(5, [hash_layer])
        microinstruction_bind(hash_layer)
        microinstruction_tick(1, [hash_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.table_unbind)
        microinstruction_tick(5, [hash_layer])
        microinstruction_unbind(hash_layer)
        microinstruction_tick(1, [hash_layer])
        microinstruction_return()

        microinstruction_start_opcode(op.read)
        microinstruction_tick(1, [])
        microinstruction_read(r_reserved.layer)
        microinstruction_tick(1, [])
        microinstruction_return()
        
        microinstruction_start_opcode(op.write)
        microinstruction_tick(1, [])
        microinstruction_write(r_reserved.layer)
        microinstruction_tick(1, [])
        microinstruction_return()

        microinstruction_start_opcode(op.exit)
        microinstruction_tick(1, [])
        microinstruction_exit()
        microinstruction_tick(1, [])
        microinstruction_return()

        microinstruction_start_opcode(op.nop)
        microinstruction_return()

        def make_mov(opcode: Opcode):
            r_from, r_to = op.mov_opcodes[opcode]
            microinstruction_start_opcode(opcode)
            #microinstruction_debug(1)
            #microinstruction_tick(1, [])
            #microinstruction_write(registers[r_from].layer)
            #microinstruction_tick(1, [])
            #microinstruction_write(registers[r_to].layer)
            microinstruction_tick(20, [registers[r_from].layer_output, registers[r_to].layer_input])
            microinstruction_inhibit_layer(registers[r_to].layer)
            microinstruction_tick(4, [registers[r_from].layer_output, registers[r_to].layer_input])
            #microinstruction_write(registers[r_to].layer)
            #microinstruction_tick(1, [])
            microinstruction_return()

        for opcode in op.mov_opcodes:
            make_mov(opcode)

        microinstruction_layer.change_n(microinstruction_next_index + 1)
        microinstruction_self_connection.set_weights(microinstruction_self_weights)
        opcode_to_microinstruction_connection.set_weights(opcode_to_microinstruction_weights)
        microinstruction_to_microinstruction_action_connection.set_weights(microinstruction_to_microinstruction_action_weights)       

        test_value_connection.set_weights(values.very_sparse_weights_to_recognize_input(t.false.name, 1, 1.5))

        def tick(count = 1, except_layers: List[Layer] = []):
            for _ in range(count):
                for layer in all_control_layers:
                    if not layer in except_layers:
                        microinstruction_action_layer.next_states[control_and_register_layer_index[layer]] = 1.0

                network.tick()

        print("Binding table with default operations")
        def bind_table(table: str, key: str, value: str):
            set_register(r_table, table)
            set_register(r_key, key)
            set_register(r_value, value)
            tick(3, [hash_layer])

            table_connection.bind()

        for (table, key) in t.operations:
            bind_table(table.name, key.name, t.operations[(table, key)].name )

        print("Binding default for table")
        hash_layer.init_states_to_one()
        values.init_states(r_value.layer, t.false.name)
        hash_to_value.bind()            

        print("Binding linked list of allocable memory")
        def bind_alloc(cell0: str, cell1: str):
            alloc_control_to_alloc_next_connection.opened = False
            values.init_states(alloc_control_layer, cell0)
            set_register(r_alloc_next, cell1)
            alloc_control_to_alloc_next_connection.bind()

        for i in range(len(t.memory_cells)-1):
            bind_alloc(t.memory_cells[i].name, t.memory_cells[i+1].name)

        alloc_control_to_alloc_next_connection.opened = True
        
        set_register(r_alloc, t.memory_cells[0].name)

        def execute(entrypoint: str, input_string: str):
            nonlocal exit_called
            nonlocal nb_ticks
            
            self.input_string = input_string
            self.input_index = 0
            self.output_string = ""

            (alloc_cell, sv, v2, sv2) = values.best_named_attractor(r_alloc.layer)

            network.clear_states()

            for register in registers:
                set_register(registers[register], t.false.name)

            set_register(r_code, entrypoint)
            set_register(r_cont, exit_label.name)
            set_register(r_alloc, alloc_cell)
            set_register(r_stack, t.false.name)

            microinstruction_layer.next_states[0] = 1.0
            test_layer.next_states[0] = 1.0

            for layer in all_control_layers:
                microinstruction_action_layer.next_states[control_and_register_layer_index[layer]] = 1.0
            
            exit_called = False
            nb_ticks = 0

            last_opcode_score = 0
            
            while not exit_called:
                if debug:
                    print("ticks {0}".format(nb_ticks))
                    ml = (microinstruction_layer.next_states >= 1.0).nonzero()[0].tolist()
                    print(ml)
                    print((microinstruction_action_layer.next_states >= 1.0).nonzero()[0].tolist())
                    print(test_layer.next_states.tolist())
                    #(v, sv, v2, sv2) = opcodes.best_named_attractor(opcode_eq_layer)
                    #if sv > 0.0:
                    #    print("opcode_eq {0} ({1}) - {2} ({3})".format(v, sv, v2, sv2))
                    #(v, sv, v2, sv2) = opcodes.best_named_attractor(opcode_neq_layer)
                    #if sv > 0.0:
                    #    print("opcode_neq {0} ({1}) - {2} ({3})".format(v, sv, v2, sv2))
                    if len(ml) == 0:
                        break

                    (v, sv, v2, sv2) = opcodes.best_named_attractor(opcode_layer)
                    if sv > 0.0 and last_opcode_score == 0:
                        print("opcode {0} ({1}) - {2} ({3})".format(v, sv, v2, sv2))
                    last_opcode_score = sv

                network.tick()
                nb_ticks += 1

            print("Ticks = {0}".format(nb_ticks))
            print("Result = '{0}'".format(self.output_string))

        self.execute = execute

        def set_debug(d = True):
            nonlocal debug
            debug = d

        self.set_debug = set_debug

