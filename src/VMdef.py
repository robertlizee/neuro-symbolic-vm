from __future__ import annotations
from typing import List, Dict, Tuple, Union, Any, Callable


class Token:
    def __init__(self, name: str):
        self.name = "t_" + name

class Tokens:
    true = Token("true")
    false = Token("false")
    zero = Token("zero")
    one = Token("one")
    two = Token("two")
    three = Token("three")
    four = Token("four")
    five = Token("five")
    six = Token("six")
    seven = Token("seven")
    eight = Token("eight")
    nine = Token("nine")
    is_digit = Token("is-digit")
    plus_mod_10 = Token("plus-mod-10")
    plus_carry = Token("plus-carry")
    unknown = Token("unknown")

    digits: List[Token]
    non_digits: List[Token]
    digits_combined: List[Token]
    memory_cells: List[Token]
    tokens: List[Token]

    operations: Dict[Tuple[Token|Label, Token|Label], Token|Label] = {}

    def __init__(self):
        self.digits = [self.zero, self.one, self.two, self.three, self.four, self.five, self.six, self.seven, self.eight, self.nine]
        self.non_digits = [self.true, self.false, self.is_digit, self.plus_carry, self.plus_mod_10, self.unknown]

        self.digits_combined = []

        for i in range(10):

            di = self.digits[i]
            self.operations[(self.is_digit, di)] = self.true

            for j in range(10):

                dj = self.digits[j]

                dij = Token(di.name + "+" + dj.name)

                self.digits_combined.append(dij)

                self.operations[(di, dj)] = dij
                self.operations[self.plus_mod_10, dij] = self.digits[(i+j)%10]
                self.operations[self.plus_carry, dij] = self.false if (i+j)//10 == 0 else self.true

        self.memory_cells = []

        for i in range(200):
            self.memory_cells.append(Token("mem" + str(i)))

        self.tokens = self.digits + self.non_digits + self.digits_combined + self.memory_cells

t = Tokens()

class Register:
    def __init__(self, name: str):
        self.name = name


class Registers:
    cons = Register("cons")
    car = Register("car")
    cdr = Register("cdr")
    stack = Register("stack")
    alloc = Register("alloc")
    alloc_next = Register("alloc-next")
    cont = Register("cont")
    table = Register("table")
    key = Register("key")
    value = Register("value")
    r1 = Register("r1")
    r2 = Register("r2")
    reserved = Register("reserved")
    arg = Register("arg")
    code = Register("code")
    code_next = Register("code_next")

r = Registers()

label_index = 0

class Label: 
    def __init__(self, name: str = None):
        global label_index

        if name == None:
            self.name = "__label__" + str(label_index)
            label_index += 1
        else:
            self.name = name

class Opcode: 
    def __init__(self, name: str):
        self.name = name

class Opcodes:
    alloc_recall = Opcode("alloc-recall")
    alloc_bind = Opcode("alloc-bind")
    alloc_unbind = Opcode("alloc-unbind")
    cons_recall = Opcode("cons-recall")
    cons_bind = Opcode("cons-bind")
    cons_unbind = Opcode("cons-unbind")
    table_recall = Opcode("table-recall")
    table_bind = Opcode("table-bind")
    table_unbind = Opcode("table-unbind")
    read = Opcode("read")
    write = Opcode("write")
    nop = Opcode("nop")
    exit = Opcode("exit")

    mov_table: Dict[Tuple[Register, Register], Opcode] = {}
    mov_opcodes: Dict[Opcode, Tuple[Register, Register]] = {}
    
    def get_all_opcodes(self):
        return [self.alloc_recall, self.alloc_bind, self.alloc_unbind, self.cons_recall, self.cons_bind, self.cons_unbind, 
            self.table_recall, self.table_bind, self.table_unbind, self.read, self.write, self.nop, self.exit] + \
            [opcode for opcode in self.mov_opcodes]

    def mov(self, r1: Register, r2: Register) -> Opcode:
        if not (r1, r2) in self.mov_table:
            opcode = Opcode("mov-" + r1.name + "-" + r2.name)
            self.mov_table[(r1, r2)] = opcode
            self.mov_opcodes[opcode] = (r1, r2)

        return self.mov_table[(r1, r2)]

op = Opcodes()

Instruction = Tuple[Label, Opcode, Opcode, Union[Token, Label, None]]

def asm(opcode: Opcode, arg: Union[Token, Label] = None, opcode_neq: Opcode = None):
    return (opcode, opcode if opcode_neq is None else opcode_neq, arg)

def mov(r1: Register, r2: Register, arg: Union[Token, Label] = None):
    return asm(op.mov(r1, r2), arg)

def exit():
    return asm(op.exit)

def alloc_recall():
    return asm(op.alloc_recall)

def alloc_bind():
    return asm(op.alloc_bind)

def alloc_unbind():
    return asm(op.alloc_unbind)

def cons_recall():
    return asm(op.cons_recall)

def cons_bind():
    return asm(op.cons_bind)

def cons_unbind():
    return asm(op.cons_unbind)

def table_recall():
    return asm(op.table_recall)

def table_bind():
    return asm(op.table_bind)

def table_unbind():
    return asm(op.table_unbind)

def read(r1: Register):
    return [
        asm(op.read),
        mov(r.reserved, r1)
    ]

def write(r1: Register):
    return [
        mov(r1, r.reserved),
        asm(op.write)
    ]

def assign(t1: Union[Token, Label], r1: Register):
    return [
        mov(r.arg, r1, t1)
    ]

def beq(code: List[Any]):
    label = Label()
    return [
        asm(op.nop, label, op.mov(r.arg, r.code)),
        code,
        label,
    ]

def bne(code: List[Any]):
    label = Label()
    return [
        asm(op.mov(r.arg, r.code), label, op.nop),
        code,
        label,
    ]

def loop(code: List[Any]):
    label = Label()
    return [
        label,
        code,
        mov(r.arg, r.code, label)
    ]

def function(label: Label, code: List[Any]):
    return [
        label,
        code,
        ret()
    ]

def jump(r1: Register):
    return [
        mov(r1, r.code),
    ]

def alloc(arg: Register):
    return [
        alloc_recall(),
        alloc_unbind(),
        mov(r.alloc, arg),
        mov(r.alloc_next, r.alloc)
    ]

def dealloc(arg: Register):
    return [
        mov(r.alloc, r.alloc_next),
        mov(arg, r.alloc),
        alloc_bind()
    ]

def cons(cons: Register, car: Register, cdr: Register):
    return [
        mov(car, r.car),
        mov(cdr, r.cdr),
        alloc(r.cons),
        cons_bind(),
        mov(r.cons, cons)   
    ]

def uncons(cons: Register, car: Register, cdr: Register):
    return [
        mov(cons, r.cons),
        cons_recall(),
        cons_unbind(),
        dealloc(cons),
        mov(r.car, car),
        mov(r.cdr, cdr)
    ]

def push(arg: Register):
    return [
        cons(r.stack, arg, r.stack)
    ]

def pop(arg: Register):
    return [
        uncons(r.stack, arg, r.stack)
    ]

def jsr(label: Label):
    return_label = Label()
    return [
        push(r.cont),
        assign(return_label, r.cont),
        mov(r.arg, r.code, label),
        return_label,
        pop(r.cont),
    ]

def ret(r1: Register = None):
    if r1 == None:
        return [
            jump(r.cont)
        ]
    else:        
        return [
            mov(r1, r.value),
            jump(r.cont)
        ]

def flatten_code(code: List[Any]) -> List[Instruction]:
    result : List[Instruction] = []

    for instruction in code:
        if isinstance(instruction, Label):
            result.append((instruction, op.nop, op.nop, None))
        elif isinstance(instruction, List):
            for instruction2 in flatten_code(instruction):
                result.append(instruction2)
        else:
            opcode, opcode_neq, arg = instruction
            result.append((Label(), opcode, opcode_neq, arg))

    return result

def print_code(code):
    for instruction in code:
        label, opcode, arg = instruction
        print("{0}: {1}{2}".format(label.name, opcode.name,  "" if arg == None else '(' + arg.name + ')'))

