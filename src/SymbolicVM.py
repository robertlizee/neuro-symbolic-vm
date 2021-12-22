# Copyright (c) 2021 Robert Lizee

from __future__ import annotations
from typing import List, Dict, Tuple, Union, Any, Callable
from random import randrange

from VMdef import *

class VM:
    registers: Dict[Register, Union[Token, Label]] = { 
        r.cons: t.false,
        r.car: t.false,
        r.cdr: t.false,
        r.stack: t.false,
        r.alloc: t.false,
        r.alloc_next: t.false,
        r.cont: t.false,
        r.table: t.false,
        r.key: t.false,
        r.value: t.false,
        r.r1: t.false,
        r.r2: t.false,
        r.reserved: t.false,
        r.arg: t.false,
        r.code: t.false,
        r.code_next: t.false
    }

    cons: Dict[Union[Token, Label], Tuple[Union[Token, Label], Union[Token, Label]]] = {}
    free: Dict[Union[Token, Label], Union[Token, Label]] = {}
    table: Dict[Tuple[Union[Token, Label], Union[Token, Label]], Union[Token, Label]] = {}

    executers: Dict[Opcode, Callable[[], None]] = {}

    input_string = ""
    input_index = 0
    output_string = ""

    code: Dict[str, Tuple[Opcode, Opcode, Union[Token, Label, None], Label]] = {}
    entrypoints: Dict[str, Label] = {}

    exit = Label(None)

    def __init__(self, code: List[Instruction]) -> None:
        for i in range(len(t.memory_cells)-1):
            self.free[t.memory_cells[i]] = t.memory_cells[i+1]

        self.registers[r.alloc] = t.memory_cells[0]

        def alloc_recall():
            alloc = self.registers[r.alloc]
            if alloc in self.free:
                self.registers[r.alloc_next] = self.free[alloc]
            else:
                raise(Exception("alloc_recall failed"))

        def alloc_bind():
            alloc = self.registers[r.alloc]
            if not alloc in self.free:
                self.free[alloc] = self.registers[r.alloc_next]
            else:
                raise(Exception("alloc_bind failed"))

        def alloc_unbind():
            alloc = self.registers[r.alloc]
            if alloc in self.free and self.registers[r.alloc_next] == self.free[alloc]:
                del self.free[alloc]
            else:
                raise(Exception("alloc_unbind failed"))
        
        def cons_recall():
            cons = self.registers[r.cons]
            if cons in self.cons:
                self.registers[r.car], self.registers[r.cdr] = self.cons[cons]
            else:
                raise(Exception("cons_recall failed"))
        
        def cons_bind():
            cons = self.registers[r.cons]
            if not cons in self.cons:
                self.cons[cons] = (self.registers[r.car], self.registers[r.cdr])
            else:
                raise(Exception("cons_bind failed"))
        
        def cons_unbind():
            cons = self.registers[r.cons]
            if cons in self.cons and self.cons[self.registers[r.cons]] == (self.registers[r.car], self.registers[r.cdr]):
                del self.cons[cons]
            else:
                raise(Exception("cons_unbind failed"))

        def table_recall():
            key = (self.registers[r.table], self.registers[r.key])
            if key in self.table:
                self.registers[r.value] = self.table[key]
            else:
                self.registers[r.value] = t.false

        def table_bind():
            key = (self.registers[r.table], self.registers[r.key])
            value = self.registers[r.value]
            if key in self.table:
                raise(Exception("table_bind failed"))
            else:
                self.table[key] = value

        def table_unbind():
            key = (self.registers[r.table], self.registers[r.key])
            value = self.registers[r.value]
            if key in self.table:
                del self.table[key]

        def read():
            if self.input_index < len(self.input_string):
                c = ord(self.input_string[self.input_index])
                self.input_index += 1
                c0 = ord('0')
                c9 = ord('9')
                if c >= c0 and c <= c9:
                    self.registers[r.reserved] = t.digits[c - c0]
                else:
                    self.registers[r.reserved] = t.unknown
            else:
                self.registers[r.reserved] = t.false
                
        def write():
            token = self.registers[r.reserved]
            for i in range(10):
                if token == t.digits[i]:
                    self.output_string += chr(ord('0') + i)
                    return
            self.output_string += ':'

        def nop():
            pass

        def make_mov(r1_r2: Tuple[Register, Register]):
            def action():
                self.registers[r1_r2[1]] = self.registers[r1_r2[0]]
            return action

        self.executers = {
            op.alloc_recall: alloc_recall,
            op.alloc_bind: alloc_bind,
            op.alloc_unbind: alloc_unbind,
            op.cons_recall: cons_recall,
            op.cons_bind: cons_bind,
            op.cons_unbind: cons_unbind,
            op.table_recall: table_recall,
            op.table_bind: table_bind,
            op.table_unbind: table_unbind,
            op.read: read,
            op.write: write,
            op.nop: nop
        }

        for opcode in op.mov_opcodes:
            self.executers[opcode] = make_mov(op.mov_opcodes[opcode])

        for i in range(len(code) - 1):
            label, opcode, opcode_neq, arg = code[i]
            label_next, _, _, _ = code[i+1]

            self.code[label.name] = (opcode, opcode_neq, arg, label_next)

            if label.name != None:
                self.entrypoints[label.name] = label

        label, opcode, opcode_neq, arg = code[len(code) - 1]
        self.code[label.name] = (opcode, opcode_neq, arg, self.exit)

        if label.name != None:
            self.entrypoints[label.name] = label

        self.table = { key: t.operations[key] for key in t.operations }
    

    def execute(self, entrypoint: str, input_string: str):
        self.input_string = input_string
        self.input_index = 0
        self.output_string = ""
        self.registers[r.code] = self.entrypoints[entrypoint]
        self.registers[r.cont] = self.exit

        while self.registers[r.code] != self.exit:
            opcode, opcode_neq, arg, next = self.code[self.registers[r.code].name]
            if arg != None:
                self.registers[r.arg] = arg
            self.registers[r.code] = self.registers[r.code_next] = next
            self.executers[opcode if self.registers[r.value] != t.false else opcode_neq]()

        print(self.output_string)

