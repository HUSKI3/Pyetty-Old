import pprint
from re import L
import sys

from lexer import PyettyLexer
from parser import PyettyParser

def pretty(value, htchar='\t', lfchar='\n', indent=0):
    nlch = lfchar + htchar * (indent + 1)
    if type(value) is dict:
        items = [
            nlch + repr(key) + ': ' + pretty(value[key], htchar, lfchar, indent + 1)
            for key in value
        ]
        return '{%s}' % (','.join(items) + lfchar + htchar * indent)
    elif type(value) is list:
        items = [
            nlch + pretty(item, htchar, lfchar, indent + 1)
            for item in value
        ]
        return '[%s]' % (','.join(items) + lfchar + htchar * indent)
    elif type(value) is tuple:
        items = [
            nlch + pretty(item, htchar, lfchar, indent + 1)
            for item in value
        ]
        return '(%s)' % (','.join(items) + lfchar + htchar * indent)
    else:
        return repr(value)

import sys
if not sys.argv[1:]:
    print('Please supply source')
    quit()

text = open(sys.argv[1], 'r').read()

if '-d' in sys.argv:
    debug = True
else:
    debug = False

lexer = PyettyLexer()
parser = PyettyParser()
pprint = pprint.PrettyPrinter(indent=2).pprint
if debug:
    for tok in lexer.tokenize(text):
        pprint(tok)

tree = parser.parse(lexer.tokenize(text))
if debug:
    pprint(tree)


# Compiler

class Fail(Exception):
    def __init__(self, *msg):
        print('\n')
        print('_main .Fail -> Caught')
        print('\n')
        super().__init__(
            '\n' + 
            '\u001b[31m' +
            ' '.join(msg) +
            '\u001b[0m'
        )
        

from llvmlite import ir

class Vibe:

    def __init__(self) -> None:
        self.types = {
            'int':ir.IntType(32),
            'int8':ir.IntType(8),
            'int16':ir.IntType(16),
            'int32':ir.IntType(32),
            'int64':ir.IntType(64),
            'bool':ir.IntType(1),
            'string': ir.ArrayType(ir.IntType(8), 1),
            'pstring': ir.IntType(8).as_pointer(),
            'float':ir.FloatType(),
            'double':ir.DoubleType(),
            'none':ir.VoidType(),
        }

        self.defaults = {
            'VARIABLE_ASSIGNMENT': self.create_var,
            'FUNCTION_DECLARATION': self.create_func,
            'FUNCTION_CALL': self.call_function,
            'RETURN': self.create_return,
            'CONDITIONAL': self.conditional,
            'POINTER': self.create_pointer
        }

        self.module  = ir.Module("_main")
        self.objects = self.create_default_methods()
        self.builder = ir.IRBuilder()

    def create_default_methods(self):
        print_type = ir.FunctionType(
            self.types['int'], 
            [ir.IntType(8).as_pointer()], 
            var_arg=True
        )
        print_func = ir.Function(self.module, print_type, 'printf')
        

        values = {
            'print':{'_func':print_func, '_ret':ir.IntType(32), '_format':True}
        }
        return values
    

    def create_var(self, tree):

        _name  = tree['ID']

        _value, _type = self.assume(tree['EXPRESSION']) 

        if 'TYPE' in tree:
            if type(_type) == ir.types.PointerType:
                print("POinter needds to be resolveddlm,qwldmwdql")
                _value = self.builder.load(_value)
            #if _type != self.types[tree['TYPE']]:
            #    t=self.types[tree['TYPE']]
            #    raise Fail(f"Invalid typed object supplied for {tree['ID']} which is assumed to be {_type} but was declared as {t}")
            #else:
            print(type(_type))
            _type = self.types[tree['TYPE']]


        if _name not in self.objects:
            ptr = self.builder.alloca(_type)
            self.builder.store(_value, ptr)
            self.objects[_name] = {'VALUE':_value, 'TYPE':_type}
        else:
            ptr, _ = self.objects[_name]
            print(f"{_name} -> {ptr}")
            self.builder.store(_value, ptr)
    
    def resolve_expression(self, op, lhs, rhs):
        lhs, lhs_type = self.assume(lhs)
        rhs, rhs_type = self.assume(rhs)

        if isinstance(rhs_type,ir.FloatType) and isinstance(lhs_type,ir.FloatType):
            Type = ir.FloatType()
            if op == '+':
                value = self.builder.fadd(lhs,rhs)
            elif op == '*':
                value = self.builder.fmul(lhs,rhs)
            elif op == '/':
                 value = self.builder.fdiv(lhs,rhs)
            elif op == '%':
                 value = self.builder.frem(lhs,rhs)
            elif op == '-':
                 value = self.builder.fsub(lhs,rhs)
            elif op == '<':
                value = self.builder.fcmp_ordered('<',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '<=':
                value = self.builder.fcmp_ordered('<=',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '>':
                value = self.builder.fcmp_ordered('>',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '>=':
                value = self.builder.fcmp_ordered('>=',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '!=':
                value = self.builder.fcmp_ordered('!=',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '==':
                value = self.builder.fcmp_ordered('==',lhs,rhs)
                Type = ir.IntType(1)
        
        elif isinstance(rhs_type,ir.IntType) and isinstance(lhs_type,ir.IntType):
            Type = ir.IntType(32)
            if op == '+':
                value = self.builder.add(lhs,rhs)
            elif op == '*':
                value = self.builder.mul(lhs,rhs)
            elif op == '/':
                value = self.builder.sdiv(lhs,rhs)
            elif op == '%':
                value = self.builder.srem(lhs,rhs)
            elif op == '-':
                value = self.builder.sub(lhs,rhs)
            elif op == '<':
                value = self.builder.icmp_signed('<',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '<=':
                value = self.builder.icmp_signed('<=',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '>':
                value = self.builder.icmp_signed('>',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '>=':
                value = self.builder.icmp_signed('>=',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '!=':
                value = self.builder.icmp_signed('!=',lhs,rhs)
                Type = ir.IntType(1)
            elif op == '==':
                value = self.builder.icmp_signed('==',lhs,rhs)
                Type = ir.IntType(1)
            elif op == 'and':
                value = self.builder.and_(lhs,rhs)
                Type = ir.IntType(1)
            elif op == 'or':
                value = self.builder.or_(lhs,rhs)
                Type = ir.IntType(1)
            
                
        return value,Type
    
    def conditional(self, tree):
        _if = tree['IF'][1] if tree['IF'][0] else None
        _else = tree['ELSE'][1] if tree['ELSE'][0] else None
        _elif = tree['ELSE_IF'] if tree['ELSE_IF'] else None
        elif_ran = False


        if _if:
            _code = _if['CODE']
            _code_else = _else['CODE']
            _condition = _if['CONDITION']
            #if _condition in 
            _condition = self.resolve_expression('>', _condition[1], _condition[2])[0]
            
            if not _else:
                with self.builder.if_then(_condition):
                    self.run(tree=_code)
            else:
                with self.builder.if_else(_condition) as (true,otherwise):

                  with true:
                      # Runs this if true
                      self.run(tree=_code)

                  with otherwise:
                      # Runs this if false
                      self.run(tree=_code_else)
        
    
    def create_func(self, tree):
        _name = tree['ID']
        _code = tree['PROGRAM']
        _arg_types = [] #[x[1]['TYPE'].lower() for x in tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS']]
        _arg_names = [] #[x[1]['ID'] for x in tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS']]
        _returns = tree['RETURNS_TYPE']

        if _returns[1]['VALUE'].lower() in self.types:
            _returns = self.types[_returns[1]['VALUE'].lower()]

        fnty = ir.FunctionType(_returns,[])
        func = ir.Function(self.module, fnty, name=_name)

        block = func.append_basic_block(f'{_name}_entry')

        prv_builder = self.builder

        self.builder = ir.IRBuilder(block)

        _arg_types_ptr = []

        for i, typ in enumerate(_arg_types):
            ptr = self.builder.alloca(typ)
            self.builder.store(func.args[i],ptr)
            _arg_types_ptr.append(ptr)
        
        prv_objects = self.objects.copy()

        for i,x in enumerate(zip(_arg_types,_arg_names)):
            typ = _arg_types[i]
            ptr = _arg_types_ptr[i]
            
            # Add function's parameter to stored variables
            self.objects[x[1]] = {"VALUE":ptr, "TYPE":typ}

        
        self.run(_code)

        self.objects = prv_objects
        self.objects[_name] = {"_func":func, "_ret":_returns}

        self.builder = prv_builder
    
    def print(self, params, Type):
        '''
            C's builtin Printf function
        '''
        print("Params:", params)
        input()
        format = params[0]
        params = params[1:]
        zero = ir.Constant(ir.IntType(32),0)
        ptr = self.builder.alloca(Type)
        self.builder.store(format,ptr)
        format = ptr
        format = self.builder.gep(format, [zero, zero])
        format = self.builder.bitcast(format, ir.IntType(8).as_pointer())
        func = self.objects['print']['_func']
        return self.builder.call(func,[format,*params])
    
    def create_pointer(self, tree):
        if tree[0]['ID'] in self.objects:
            value, Type = self.objects[tree[0]['ID']]['VALUE'],\
                            self.objects[tree[0]['ID']]['TYPE']
        else:
            value, Type = self.assume(tree[0]['ID'])
        zero = ir.Constant(ir.IntType(32),0)
        ptr = self.builder.alloca(Type)
        format = ptr
        format = self.builder.gep(format, [zero, zero])
        format = self.builder.bitcast(format, ir.IntType(8).as_pointer())
        return (format, value)
    
    def resolve(self, tree):
        # Lmao fuck it
        val = self.builder.load(self.objects[tree[0]['ID']]['VALUE'])
        return val, self.objects[tree[0]['ID']]['TYPE']

    def call_function(self, tree):
        if isinstance(tree, tuple):
            tree = tree[0]
        print(tree)
        _name = tree['ID'][1]['VALUE']
        _posargs = tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS'] if 'POSITIONAL_ARGS' in tree['FUNCTION_ARGUMENTS'] else None
        _kwargs = None #WIP

        _posargs_types = []
        _posargs_values = []

        if _posargs:
            for arg in _posargs:
                value, type = self.assume(arg)
                _posargs_types.append(type)
                _posargs_values.append(value)

        print(_posargs_values)
        print('\n')
        print(_posargs_types)
        #quit()
        func, ret_type = self.objects[_name]['_func'], self.objects[_name]['_ret']
        if 'print' == _name:
            ret = self.print(_posargs_values, _posargs_types[0])
        else:
            ret = self.builder.call(func, _posargs_values)
        print(ret, ret_type)
        input()
        return ret, ret_type


    def create_return(self, tree):
        if type(tree) == dict:
            value, __Type = self.assume(tree['EXPRESSION'])
        else:
            value, __Type = self.assume(tree[0]['EXPRESSION'])
        print(value)
        self.builder.ret(value)
    
    def create_int(self, tree):
        value, Type = tree[0]['VALUE'], self.types['int']
        return ir.Constant(Type, value), Type

    def to_array(self, value, type):
        string = value.replace('\\n','\n\0')
        n = len(string)+1
        buf = bytearray((' ' * n).encode('ascii'))
        buf[-1] = 0
        buf[:-1] = string.encode('utf8')
        return ir.Constant(ir.ArrayType(type, n), buf),ir.ArrayType(type, n)

    def create_string(self, tree):
        value = tree[0]['VALUE']
        string, Type = self.to_array(value, ir.IntType(8))
        return string, Type
    
    def fetch_id(self, tree):
        if tree[0]['VALUE'].lower() in self.types:
            return self.types[tree['VALUE'].lower()]
        else:
            return (
                    self.objects.get(tree[0]['VALUE'])['VALUE'], 
                    self.objects.get(tree[0]['VALUE'])['TYPE']
                )

    def assume(self, tree):
        actions = {
            # Types
            'INT': self.create_int,
            'ID': self.fetch_id,
            'STRING': self.create_string,
            # Functions
            'FUNCTION_CALL': self.call_function,
            # Logical?
            'POINTER': self.create_pointer,
            'RESOLVE': self.resolve
        }
        _body = tree[1:]

        if tree[0] in actions:
            print(tree)
            _val = actions[tree[0]](_body)
            if _val:
                return _val
            else:
                raise Fail(f"[ASSUME] -> Variable failed to resolve.\nAction: {tree[0]}\nValue: {_val}\nLine: {tree[2]}\nSource: {_body}\nObjects:\n{pretty(self.objects, indent=0)}")
        else:
            raise Fail(f"[ASSUME] -> Unknown action '{tree[0]}' found.\nAction: {tree[0]} \nLine: {tree[2] if len(tree) > 2 else 'Unknown - Ran in a function'}")

    def run(self, tree):
        for node in tree:
            if node[0] in self.defaults:
                print(f'[{node[0]}] -> {node[1]}')
                self.defaults[node[0]](node[1])
            else:
                raise Fail(f"[RUN] -> Unknown action '{node[0]}' found. Line {node[2]}")

comp = Vibe()

comp.run(tree)

print(comp.objects)


import llvmlite.binding as llvm
from ctypes import CFUNCTYPE, c_int, c_float
from time import time

module = comp.module
module.triple = llvm.get_default_triple()
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

print('Module:')
try:
    print(str(module))
except AttributeError:
    print('Failed :(')

print('Module end')
llvm_ir_parsed = llvm.parse_assembly(str(module))
llvm_ir_parsed.verify()
target_machine = llvm.Target.from_default_triple().create_target_machine()
engine = llvm.create_mcjit_compiler(llvm_ir_parsed, target_machine)
engine.finalize_object()
# Run the function with name func_name. This is why it makes sense to have a 'main' function that calls other functions.
entry = engine.get_function_address('main')
cfunc = CFUNCTYPE(c_int)(entry)
start_time = time()
result = cfunc()
end_time = time()
print(f'It returns {result}')
print('\nExecuted in {:f} sec'.format(end_time - start_time))
