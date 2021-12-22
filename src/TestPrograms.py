from __future__ import annotations
from typing import List, Dict, Tuple, Union, Any, Callable

from VMdef import *

adder = flatten_code([
    function(Label('read-number'), [ # returns value in r.out
        assign(t.false, r.r1), # init r.r1 to t.nil

        assign(t.is_digit, r.table), # set r.table to test for a digit

        loop([
            read(r.value), # read token and test if it is a digit
            
            bne([
                ret(r.r1)
            ]),
            
            mov(r.value, r.key),
            table_recall(),

            bne([   # if not unread the token and return the number we have so far
                ret(r.r1)
            ]),

            cons(r.r1, r.key, r.r1), # push the digit found on the number
        ])    
    ]),
    

    function(Label('add-numbers'), [ # input r1, r2, cont: continuation
        mov(r.r1, r.value), # if r.r1 == t.false then return r.r2
        bne([
            ret(r.r2)                
        ]),
        mov(r.r2, r.value), # if r.r2 == t.false then return r.r1
        bne([
            ret(r.r1)                
        ]),

        uncons(r.r1, r.key, r.r1), # r.arg = pop(r.r1)
        uncons(r.r2, r.table, r.r2), # r.op = pop(r.r2)

        table_recall(), # push result of combining first digit of r.r1 and first digit of r.r2 
        push(r.value),

        jsr(Label('add-numbers')), # recursive call with r.r1 and r.r2 having their first digit removed

        pop(r.key),
        mov(r.value, r.r2), # put the addition in r.r2
        assign(t.plus_carry, r.table), # check if the addition of the two first digits generate a carry
        table_recall(),
        beq([  # if so increment r.r2
            push(r.key),
            assign(t.false, r.r1),
            assign(t.one, r.key),
            cons(r.r1, r.key, r.r1),
            jsr(Label('add-numbers')), # recursive call with r.r1 and r.r2 having their first digit removed
            pop(r.key),
            mov(r.value, r.r2),
        ]),
        assign(t.plus_mod_10, r.table), # append the sum mod 10 of the two first digits and append it to the result
        table_recall(),
        cons(r.r2, r.value, r.r2),
        ret(r.r2),

    ]),

    function(Label('write-number'), [
        mov(r.r1, r.value), # test if the number == t.false and return if so
        bne([
            ret()                
        ]),
        uncons(r.r1, r.r2, r.r1), # pop the first digit from the number r.r1 and push on the stack
        push(r.r2),

        jsr(Label('write-number')), # write the number without the first digit recursively
        
        pop(r.r2), # pop the first digit from the stack and write it
        write(r.r2)
    ]),

    function(Label('main'), [
        jsr(Label('read-number')), # read first number and push it
        push(r.value),

        jsr(Label('read-number')), # read the second number and push it
        push(r.value),

        pop(r.r2), # pop the second number and the first number and add them
        pop(r.r1),
        jsr(Label('add-numbers')),

        mov(r.value, r.r1), # write the resulting number
        jsr(Label('write-number'))
    ]),
    
    function(Label('increment'), [ # returns r.1 + 1 in r.out 
        assign(t.false, r.r2),
        assign(t.one, r.key),
        cons(r.r2, r.key, r.r2),
        jsr(Label('add-numbers'))
    ]),

    function(Label('count-digits'), [ # after this register r.table holds a table with digits as keys and numbers as values
        alloc(r.table),


        loop([
            read(r.value), # read token and test if it is a digit
            
            bne([
                ret(r.table)
            ]),

            mov(r.table, r.r1),
            mov(r.value, r.r2),

            mov(r.value, r.key),
            assign(t.is_digit, r.table), # set r.op to test for a digit
            table_recall(),

            bne([   # if not unread the token and return the table
                ret(r.r1)
            ]),

            mov(r.r1, r.table),
            mov(r.r2, r.key),

            table_recall(),
            table_unbind(),

            mov(r.value, r.r1),

            push(r.table),
            push(r.key),
            jsr(Label('increment')),

            pop(r.key),
            pop(r.table),
            table_bind(),
        ])    
    ]),

    function(Label('main-count-digits'), [  # print r.table which maps digits to number and deallocates it

        jsr(Label('count-digits')),
        mov(r.value, r.table),
        assign(t.zero, r.key),
        jsr(Label('write-digit-count')),
        assign(t.one, r.key),
        jsr(Label('write-digit-count')),
        assign(t.two, r.key),
        jsr(Label('write-digit-count')),
        assign(t.three, r.key),
        jsr(Label('write-digit-count')),
        assign(t.four, r.key),
        jsr(Label('write-digit-count')),
        assign(t.five, r.key),
        jsr(Label('write-digit-count')),
        assign(t.six, r.key),
        jsr(Label('write-digit-count')),
        assign(t.seven, r.key),
        jsr(Label('write-digit-count')),
        assign(t.eight, r.key),
        jsr(Label('write-digit-count')),
        assign(t.nine, r.key),
        jsr(Label('write-digit-count')),
        dealloc(r.table)
    ]),

    function(Label('write-digit-count'), [
        table_recall(),
        table_unbind(),

        bne([
            ret()
        ]),
        assign(t.unknown, r.r1),
        write(r.r1),
        write(r.key),
        write(r.r1),
        mov(r.value, r.r1),
        push(r.table),
        jsr(Label('write-number')),
        pop(r.table),
        assign(t.unknown, r.r1),
        write(r.r1),
    ]),

    function(Label('test'), [
        assign(t.one, r.r1),
        write(r.r1)
    ]),

    function(Label('echo'), [
        loop([
            read(r.value),
            bne([
                ret()
            ]),
            write(r.value)
        ])
    ]),

    function(Label('echo2'), [
        jsr(Label('read-number')),

        loop([
            bne([
                ret()
            ]),
            uncons(r.value, r.r1, r.value),
            write(r.r1),
        ])
    ]),

    function(Label('echo3'), [
        jsr(Label('read-number')),
        mov(r.value, r.r1), # write the resulting number
        jsr(Label('write-number'))
    ]),


])

