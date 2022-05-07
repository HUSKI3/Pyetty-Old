import pprint
import logging
from re import L, M
import sys

from lexer import PyettyLexer
from parser import PyettyParser

class Locale:
    def __init__(self, d):
        for _ in d:
            #print('Adding',_)
            setattr(self, _, d[_])
        #print(self.__dict__)

import numba
from numba import cuda


class PyettyInterpreter:
    def __init__(
        self,
        tree,
        name,
        local_args,
        global_args,
        lexer=None,
        parser=None,
        methods={},
        parent=None
    ):
        self.program = tree

        self.locals = local_args
        self.methods = methods
        self.name = name
        self.globals = global_args
        self.lexer = lexer
        self.parser = parser
        self.parent = parent

        self.defaults = {
            'VARIABLE_ASSIGNMENT': self.create_var,
            'PYTHON_CODE': self.python_eval,
            'PYTHON_CODE_EXEC': self.python_exec,
            'FUNCTION_DECLARATION': self.create_fun,
            'FUNCTION_CALL': self.run_function,
            'IMPORT': self.eval_import,
            'CLASS_DECLARATION': self.create_class,
            'GLOBALS':self.load_globals,
            'RETURN':self.returns,
            'WHILE': self.whiles,
            'BREAK': self.breaks,
            'SKIP': self.skips,
            'CONDITIONAL': self.proc_conditional,
            'DEFINE': self.define,
            'DEPENDS': self.depends,
            'NEWOBJECT': self.new_object
        }

    @numba.jit(target_backend='cuda',forceobj=True)
    def run(self, tree=None):
        self.create_defaults()
        pg = self.program
        if tree: pg = tree
        if pg:
            for line in pg:
                # print(f'Action: [{line[0]}]')
                if line[0] in self.defaults:
                    self.defaults[line[0]](line[1])
    
    def create_defaults(self):
        self.create_fun(
            { 'FUNCTION_ARGUMENTS': {'POSITIONAL_ARGS': (('ID', {'VALUE': 'x'}, 24),)},
            'ID': 'void',
            'PROGRAM': (('RETURN', {'EXPRESSION': ('ID', {'VALUE': 'x'}, 25)}, 25),),
            'RETURNS_TYPE': ('ID', {'VALUE': 'Any'}, 26)}
        )

            

    def eval_import(self, tree):
        path = self.eval_expr(tree['EXPRESSION'])
        lib = path.split('/')[1].strip('.ppy')

        with open(path, 'r') as f:
            if self.parser:
                tree = self.parser.parse(self.lexer.tokenize(f.read()))
            else:
                raise Exception(f"[{self.name}] Failed to read a library? Perhaps library '{path}' should be loaded in the main file.")

        args = [
            tree,
            f'_import_{lib}',
            {}, {}, {}
        ]

        ImportInstance = PyettyInterpreter.__new__(
            PyettyInterpreter,
            *args
        )
        ImportInstance.__init__(
            *args,
            parent=self
        )
        ImportInstance.run()

        self.locals[lib] = ImportInstance.locals

        # print(f'Loaded {lib}', ImportInstance.locals, ImportInstance.methods,sep='\n')

        # print(self.locals, i.locals, '\n', self.methods, i.methods)

    def create_class(self, tree):
        c = Class(
                tree['PROGRAM'],
                tree['ID'],
                {},
                {}, parent=self, lexer=self.lexer, parser=self.parser
            )
        c.run()
        self.locals[tree['ID']] = c.locals
        self.methods[tree['ID']] = c.methods
    
    def new_object(self, tree):
        if tree['FROM'] in self.locals:
            self.locals[tree['TO']] = self.locals[tree['FROM']].copy()[tree['FROM']]
            _vars = self.locals[tree['FROM']].copy()
            del _vars[tree['FROM']]
            self.locals[tree['TO']] = _vars
            self.locals[tree['TO']][tree['TO']] = self.locals[tree['FROM']].copy()[tree['FROM']]
        else:
            raise Exception(f"[NewObject] No such class '{tree['FROM']}'")

    def proc_conditional(self, tree):
        _if = tree['IF'][1] if tree['IF'][0] else None
        _else = tree['ELSE'][1] if tree['ELSE'][0] else None
        _elif = tree['ELSE_IF'] if tree['ELSE_IF'] else None
        elif_ran = False


        if _if:
            _code = _if['CODE']
            _condition = _if['CONDITION']
            if self.eval_expr(_condition):
                self.run(tree=_code)
            else:
                
                if _elif:
                    for _e in _elif[1:]:
                        if _e is None:
                            continue
                        if type(_e) == tuple:
                            _e = _e[0]
                        _code = _e['CODE']
                        _condition = _e['CONDITION']
                        if self.eval_expr(_condition):
                            self.run(tree=_code)
                            elif_ran = True
    
                if _else and not elif_ran:
                    _code = _else['CODE']
                    _condition = _if['CONDITION']
                    if not self.eval_expr(_condition):
                        self.run(tree=_code)

        
    
    def whiles(self, tree):
        _condition = tree['CONDITION']
        _currently = self.eval_expr(_condition)
        _program = tree['PROGRAM']

        if _currently and _program:
            self.create_fun( { 
                'FUNCTION_ARGUMENTS': { 
                    'POSITIONAL_ARGS': (
                        (
                            'ID',
                            {'VALUE': 'phase'},
                            0
                        ),
                    )
                    },
                'ID': '_while',
                'PROGRAM': ( _program )}
            )
            # Keep this for consistancy!
            self.locals['phase'] = ('INT',{'VALUE':0})
            self.breaking = False
            self.skip = False
            while self.eval_expr(_condition) and not self.breaking:
                if self.skip:
                    pass
                else:
                    self.run_function({ 
                        'FUNCTION_ARGUMENTS': {
                            'POSITIONAL_ARGS': (
                                ('ID',{'VALUE': 'phase'},0),
                            )
                        },
                        'ID': ('ID', {'VALUE': '_while'}, 0),
                        }
                    )
            self.skip = False
        
    def breaks(self, tree):
        if self.parent:
            self.parent.breaking = True
    
    def skips(self, tree):
        if self.parent:
            self.parent.skip = True

    def create_var(self, tree): 
        #print()
        #pprint(tree)
        if tree['EXPRESSION'][0] in ['INT', 'STRING', 'BOOL']:
            ticked = self.eval_expr(tree['EXPRESSION'])
            packed = {
                "TYPE": tree["EXPRESSION"][0],
                "VALUE": ticked
            }
        elif tree['EXPRESSION'][0] == 'ID':
            ticked = self.resolve_var(tree['EXPRESSION'])
            packed = {
                "TYPE": tree["EXPRESSION"][0],
                "VALUE": ticked
            }
        elif tree['EXPRESSION'][0] == 'FUNCTION_CALL':
            ticked = self.run_function(tree['EXPRESSION'][1])
            packed = {
                "TYPE": tree["EXPRESSION"][0],
                "VALUE": ticked
            }
        else:
            ticked = self.eval_expr(tree['EXPRESSION'])
            packed = {
                    "TYPE": tree["EXPRESSION"][0],
                    "VALUE": ticked
                }
        if self.parent is None:
            self.locals[tree["ID"]] = packed
        else:
            self.locals[tree["ID"]] = packed
            if '__globals' in self.locals:
                self.parent.locals[tree["ID"]] = packed
        #print(tree["ID"],{"TYPE": tree["EXPRESSION"][0],"VALUE": ticked})

    def create_fun(self, tree):

        if 'NAMESPACE' in tree:
            if tree['NAMESPACE'] not in self.locals:
                self.locals[tree['NAMESPACE']] = {}
            self.locals[tree['NAMESPACE']][tree['ID']] = Function(
                tree,
                tree['ID'],
                {},
                {}, parent=self, lexer=self.lexer, parser=self.parser
            )

        else:
            self.methods[tree['ID']] = Function(
                tree,
                tree['ID'],
                {},
                {}, parent=self, lexer=self.lexer, parser=self.parser
            )

        #print(self.name, self.locals)
    
    def load_globals(self, tree):
        self.locals = {**self.locals, **self.parent.locals}
        self.locals['__globals'] = {"VALUE":True}
    
    def returns(self, tree):
        # print(tree)
        if tree['EXPRESSION'][0] == 'ID':
            return self.resolve_var(tree['EXPRESSION'])
        elif tree['EXPRESSION'][0] in ['STRING','INT','BOOL']:
            return tree['EXPRESSION']
        elif tree['EXPRESSION'][0] in self.defaults:
            return self.defaults[tree['EXPRESSION'][0]](tree['EXPRESSION'][1])
        else:
            return self.eval_expr(tree['EXPRESSION'])
    
    def depends(self, tree):
        depends = tree['TO'][1]['VALUE']
        if depends not in self.parent.locals:
            raise Exception(f"[{self.name}] I depend on '{depends}'! Please include it in the main file!")

    def define(self, tree):
        _to = tree['TO']
        _from = tree['EXPRESSION']
        if _from[0] == 'CLASS_ATTRIBUTE':
            _func = _from[1]['ATTRIBUTE']
            _class = _from[1]['CLASS'][1]['VALUE']
            if _class not in self.locals:
                raise Exception(f'No class {_class} in', pprint(self.locals))

            if _func not in self.locals[_class][_class]:
                if _func not in self.locals[_class]:
                    raise Exception(f'!!! Unbound method {_func}, tree: {self.locals[_class]}')
                else:
                    self.methods[_to] = self.locals[_class][_func]
            else:
                self.methods[_to] = self.locals[_class][_class][_func]
        elif _from[0] == 'ID':
            self.methods[_to] = self.methods[_from[1]['VALUE']]

    
    def run_function(self, tree):
        returns = []

        if 'ONCOMPLETE' in tree:
            _oncomplete = list(tree['ONCOMPLETE'])
            _oncomplete.insert(0, ['GLOBALS', {'VALUE': ''}, 0] )
        else:
            _oncomplete = None

        if tree['ID'][0] == 'CLASS_ATTRIBUTE':
            _func = tree['ID'][1]['ATTRIBUTE']
            _class = tree['ID'][1]['CLASS'][1]['VALUE']


            _vars = []
            if 'POSITIONAL_ARGS' in tree['FUNCTION_ARGUMENTS']:
                for var in tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS']:
                    if var[0] not in self.defaults:
                        resed = self.resolve_var(var)
                    else:
                        resed = self.defaults[var[0]](var[1])
                    if not resed:
                        raise Exception(f'Variable "{var[0]}" "{self.defaults}" resolution failed. Does it exist?')
                    _vars.append(resed)

            if _class in self.locals:
                if _class in self.locals[_class]:
                    if _func in self.locals[_class][_class]:
                        returns = self.locals[_class][_class][_func].run(_vars, {})
                    else:
                        raise Exception(f'!!! Unbound method [{_func}], locals: {self.locals[_class]}')
                else:
                    if _func in self.locals[_class]:
                        self.locals[_class][_func].run(_vars, {})
                    else:
                        raise Exception(f'!!! Unbound method [{_func}], Class locals: {self.locals[_class]}\n Globals: {self.locals}')
            else:
                raise Exception(f'No class {_class} in {self.locals}!')

        else:
            if 'VALUE' in tree['ID'][1]:
                _func = tree['ID'][1]['VALUE']
            elif 'VALUE' in tree['ID'][1][1]:
                _func = tree['ID'][1][1]['VALUE']

            if _func not in self.methods:
                print('!!! Unbound method, tree:', tree)

            _vars = []
            if 'POSITIONAL_ARGS' not in tree['FUNCTION_ARGUMENTS']:
                _vars = []
            else:
                for var in tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS']:
                    if var[0] not in self.defaults:
                        resed = self.resolve_var(var)
                    else:
                        resed = self.defaults[var[0]](var[1])

                    if not resed:
                        raise Exception(
                            f'Variable resolution failed. Does it exist? {resed}')
                    _vars.append(resed)

            # Run it!
            returns = self.methods[_func].run(_vars, {})
        
        #print(f'Ran {_func} and got {returns}')

        if _oncomplete:
            #print('_oncomplete',_oncomplete)
            self.create_fun( { 
                'FUNCTION_ARGUMENTS': { 
                    'POSITIONAL_ARGS': (
                        (
                            'ID',
                            {'VALUE': 'phase'},
                            0
                        ),
                    )
                    },
                'ID': 'phased',
                'PROGRAM': ( _oncomplete )}
            )
            if not returns:
                self.locals['phase'] = ( 'STRING', {'VALUE':''}, 0 )
            else:
                self.locals['phase'] = returns if len(returns) > 1 else returns[0]

            self.run_function({ 
                'FUNCTION_ARGUMENTS': {
                    'POSITIONAL_ARGS': (
                        ('ID',{'VALUE': 'phase'},0),
                    )
                },
                'ID': ('ID', {'VALUE': 'phased'}, 0),
                }
            )
        else:
            #print(returns)
            #input()
            for res in returns:
                if type(res) == tuple and res[0] not in ['STRING','INT','BOOL']:
                    if res[0] == 'ID':
                        return self.resolve_var(res)
                    else:
                        return self.eval_expr(res)
                else:
                    return res


    def locals_watcher(self):
        return (self, self.locals)

    def resolve_var(self, tree):
        if tree[0] == 'ID':
            if tree[1]['VALUE'] in self.locals:
                return self.locals.get(tree[1]['VALUE'])
            else: 
                raise Exception(f'[Resolver] Couldnt resolve variable --> {tree[1]["VALUE"]} on line {tree[2]} \nLocals: {self.locals}\n' + str("Parent: " + str(self.parent.locals)) if self.parent else '')
        else:
            # Assume legacy variable
            return (tree[0], {'VALUE': self.eval_expr(tree)})

    def eval_str(self, tree):
        tree = tree[0]
        return tree['VALUE']

    def eval_int(self, tree):
        tree = tree[0]
        return int(tree['VALUE'])  # Brave to assume it's actually int ;P

    def eval_add(self, tree):
        one = self.eval_expr(tree[0])
        two = self.eval_expr(tree[1])
        while type(one) == dict:
            one = self.eval_expr(one)
            #print('one tried!')
        while type(two) == dict:
            two = self.eval_expr(two)
            #print('two tried!')
        return one + two

    def eval_sub(self, tree):
        return self.eval_expr(tree[0]) - self.eval_expr(tree[1])

    def eval_mul(self, tree):
        return self.eval_expr(tree[0]) * self.eval_expr(tree[1])

    def eval_div(self, tree):
        one = self.eval_expr(tree[0])
        two = self.eval_expr(tree[1])
        while type(one) == dict:
            one = self.eval_expr(one)
            #print('one tried!')
        while type(two) == dict:
            two = self.eval_expr(two)
            #print('two tried!')
        return one / two
    
    def eval_list(self, tree):
        _ = []
        for item in tree[0]['ITEMS']:
            _.append(self.eval_expr(item))
        return _
    
    def eval_null(self, tree):
        return ''
    
    def eval_get_index(self, tree):
        tree = tree[0]
        lister = self.eval_expr(tree['EXPRESSION'])
        while type(lister) == dict:
            lister = self.eval_expr(lister)
        index = self.eval_expr(tree['INDEX'])
        while type(index) == dict:
            index = self.eval_expr(index)
        return lister[index]

    def eval_not_equal(self, tree):
        one = self.eval_expr(tree[0])
        two = self.eval_expr(tree[1])
        while type(one) == dict:
            one = self.eval_expr(one)
            #print('one tried!')
        while type(two) == dict:
            two = self.eval_expr(two)
            #print('two tried!')
        return one != two
    
    def eval_equal(self, tree):
        one = self.eval_expr(tree[0])
        two = self.eval_expr(tree[1])
        while type(one) == dict:
            one = self.eval_expr(one)
            #print('one tried!')
        while type(two) == dict:
            two = self.eval_expr(two)
            #print('two tried!')
        return one == two
    
    def eval_grtr(self, tree):
        one = self.eval_expr(tree[0])
        two = self.eval_expr(tree[1])
        while type(one) == dict:
            one = self.eval_expr(one)
            #print('one tried!')
        while type(two) == dict:
            two = self.eval_expr(two)
            #print('two tried!')
        return one > two

    def create_dict(self, tree):
        _ = {}
        for item in tree[0]['ITEMS']:
            _name = item[0][1]['VALUE']
            # We now need to resolve the values
            if item[1][0] == 'LIST':
                _value = self.eval_expr(item[1])
            elif item[1][0] == 'ASSOC_ARRAY':
                _value = self.eval_expr(item[1])
            else:
                _value = item[1][1]["VALUE"]
            _[_name] = _value
        return _

    def get_assoc_index(self, tree):
        soc = self.eval_expr(tree[0]['EXPRESSION'])
        idx = tree[0]['INDEX'][1]['VALUE']\
             if tree[0]['INDEX'][1]['VALUE'] not in self.locals\
                 else self.eval_expr(self.locals.get(tree[0]['INDEX'][1]['VALUE']))
        while type(idx) == dict:
            idx = self.eval_expr(idx)
        if type(soc['VALUE']) == dict:
            return soc['VALUE'].get(idx)
        elif type(soc['VALUE']) == list:
            return soc['VALUE'][int(idx)] if int(idx) < len(soc['VALUE']) else []
    
    def get_class_attr(self, tree):
        line = tree[1]
        tree = tree[0]
        tree['CLASS'] = tree['CLASS'][1]['VALUE']
        if tree['CLASS'] == 'self':
            if self.parent.name in self.locals:
                return self.locals[self.parent.name][tree['ATTRIBUTE']]
            else:
                raise Exception(f"[GetClassAttr] {self.name}:{line} Class '{self.parent.name}' does not exist or has not yet been declared\nLocals: {self.locals}")
        if tree['CLASS'] not in self.locals:
            raise Exception(f"[GetClassAttr] {self.name}:{line} Class '{tree['CLASS']}' does not exist or has not yet been declared\nLocals: {self.locals}")
        if tree['ATTRIBUTE'] in self.locals[tree['CLASS']]:
            return self.locals[tree['CLASS']][tree['ATTRIBUTE']]
        else:
            raise Exception(f"[GetClassAttr] {self.name}:{line} Class '{tree['CLASS']}' has no attribute '{tree['ATTRIBUTE']}'\nLocals: {self.locals}")

    def convert_type(self, tree):
        var = list(tree[0]['VAR'])
        var[0] = tree[0]['TO']
        return var

    def eval_expr(self, tree, raw=False):
        # print(tree)
        # New variable type matching
        if type(tree) == dict:
            
            if 'TYPE' in tree:
                if tree['TYPE'] == 'ASSOC_ARRAY':
                    return tree['VALUE']

            # Below is only valid for legacy code,
            # With the addition of assoc arrays 
            # this is no longer valid

            # -- Ignore resolving or type matching!
            # This is because we fed in a raw type!
            # A raw type doesn't require type-matching! --
            if 'VALUE' in tree:
                return tree['VALUE']
            else:
                if raw:
                    # Failed to resolve, return whole tree
                    return tree
                else:
                    raise Exception("Failed to resolve a raw variable, perhaps this is a Pyetty call?")

        elif type(tree) == str or type(tree) == int:
            # Seems like the function bounced!
            # We will try to feed back the variable raw!
            return tree

        _act = tree[0]
        _body = tree[1:]
        _val = None

        self.procs = {
            # Types
            'STRING': self.eval_str,
            'INT': self.eval_int,
            'NULL': self.eval_null,
            'LIST': self.eval_list,
            'GET_INDEX': self.eval_get_index,
            'ASSOC_ARRAY': self.create_dict,
            'GET_ASSOC_INDEX': self.get_assoc_index,
            'CLASS_ATTRIBUTE': self.get_class_attr,
            # Math
            'ADD': self.eval_add,
            'SUB': self.eval_sub,
            'MUL': self.eval_mul,
            'DIV': self.eval_div,
            # ID
            'ID': self.resolve_var,
            'TYPECONVERT': self.convert_type,
            # Logik
            'NOT_EQEQ': self.eval_not_equal,
            'EQEQ': self.eval_equal,
            'GREATER': self.eval_grtr
        }

        if _act in self.procs:
            if _act == 'ID':
                _val = self.resolve_var(tree)
            else:
                _val = self.procs[_act](_body)

        if _val is not None:
            return _val
        else:
            raise Exception(
                f'Variable "{_body[0]}" was never declared or failed to resolve. Line: {_body[1]} Act:{_act} Body:{_body}')

    def python_eval(self, tree):
        code = tree["CODE"]
        x = {}
        for _ in self.locals:
            try:
                x[_] = self.eval_expr(self.locals[_])
            except:
                x[_] = self.locals[_]
        return eval(code, {'etty': Locale(x)})

    def python_exec(self, tree):
        code = tree["CODE"]
        x = {}
        for _ in self.locals:
            try:
                x[_] = self.eval_expr(self.locals[_])
            except:
                x[_] = self.locals[_]
        return exec(code, {'etty': Locale(x)})


class Function(PyettyInterpreter):
    def __init__(self, tree, name, local_args, global_args, parent=None,lexer=None,parser=None):
        PyettyInterpreter.__init__(self, tree, name, local_args, global_args, parent=parent,lexer=lexer,parser=parser)
        self.tree = tree
        if 'POSITIONAL_ARGS' in self.tree['FUNCTION_ARGUMENTS']:
            self.required_args = self.proc_args()
        else:
            self.required_args = []
        # self.run([('STRING','hello'),('INT',2),('INT',3)],{}) # for testing run after declaration

    def proc_args(self):
        proced_args = []
        for arg in self.tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS']:
            if type(arg) == tuple and arg[0] == 'TYPED':
                # We found a typed argument!
                proced_args.append(arg[1])
            else:
                proced_args.append(arg)
        return proced_args

    def run(self, args={}, kwargs={}, tree=None):
        if tree: 
            for line in tree:
                if line[0] in self.defaults:
                    self.defaults[line[0]](line[1])
            return 

        if len(args) < len(self.required_args):
            raise Exception(f'[{self.name}] Not enough arguments supplied!\nRequired: {len(self.required_args)} :: {self.required_args}\nGot: {len(args)} :: {args}')

        # Check if there are types!
        for i, arg in enumerate(self.required_args):
            #print('>',args[i], arg) # for debugging tuples AAAAAAAA

            if 'TYPE' in arg:
                # Check type
                # TYPES CAN ONLY BE CAPS!!! But they will be converted to CAPS AS WELL
                if type(args[i]) == dict:
                    if arg['TYPE'].upper() == args[i]['TYPE'].upper():
                        self.locals[arg['ID']] = {
                            'TYPE': arg['TYPE'].upper(),
                            'VALUE': args[i]['VALUE']
                        }
                    else:
                        raise Exception(f'Invalid non-typed object supplied [{args[i]["TYPE"]}] for type [{arg["TYPE"]}]')
                elif type(args[i]) == tuple:
                    if args[i][0] == 'TYPECONVERT':
                        # Manually convert type
                        args[i] = args[i][1]['VALUE']
                    if arg['TYPE'].upper() == args[i][0].upper():
                        self.locals[arg['ID']] = {
                            'TYPE': arg['TYPE'].upper(),
                            'VALUE': args[i][1]['VALUE']
                        }
                    else:
                        raise Exception(f'Invalid non-typed object supplied [{args[i][0]}] for type [{arg["TYPE"]}]')
                else:
                    raise Exception(f'Invalid typed object supplied {args[i]}')

            else:
                if type(args[i]) == dict:
                    if 'TYPE' in args[i]:
                        self.locals[arg[1]['VALUE']] = {
                            'TYPE': args[i]['TYPE'],
                            'VALUE': args[i]['VALUE']
                        }
                elif type(args[i]) == tuple:
                    if type(args[i][1]) == dict:
                        # We didnt resolve fully :(
                        # Let's manually resolve
                        self.locals[arg[1]['VALUE']] = {
                            'TYPE': args[i][0],
                            'VALUE': args[i][1]['VALUE']
                        }
                    else:
                        self.locals[arg[1]['VALUE']] = {
                            'TYPE': args[i][0],
                            'VALUE': args[i][1]
                        }
                elif type(args[i]) in [str, int, bool]:
                    self.locals[arg[1]['VALUE']] = {
                        'TYPE': 'ANY',
                        'VALUE': args[i]
                    }
                else:
                    raise Exception(
                        f'Invalid non-typed object supplied {args[i]}')

        returns = []
        self.parent.skip = False

        
        if self.tree:
            for line in self.tree['PROGRAM']:
                #print(f'Action: [{line[0]}] :: {line[1]}')
                if line[0] == 'SKIP':
                    self.parent.skip = True
                    pass

                elif line[0] == 'BREAK':
                    self.parent.breaking = True
                    break

                elif line[0] in self.defaults:
                    ret = self.defaults[line[0]](line[1])
                    #print(f'>> {ret}')
                    if ret:
                        returns.append(ret)

        return returns


class Class(PyettyInterpreter):
    def __init__(self, tree, name, local_args, global_args, parent=None,lexer=None,parser=None):
        self.tree = tree
        PyettyInterpreter.__init__(self, tree, name, local_args, global_args, parent=parent,lexer=lexer,parser=parser)
        self.create_defaults()

    def create_defaults(self):
        f = self.create_fun(
            { 'FUNCTION_ARGUMENTS': {'POSITIONAL_ARGS': (('ID', {'VALUE': 'object'}, 0),)},
            'ID': '_new',
            'PROGRAM': (
                ('GLOBALS', {'VALUE': ''}, 0),
                ('CLASS_DECLARATION', {'ID': self.name, 'PROGRAM': (self.tree), 'PARENT': self.parent}, 0),
            ),
            'RETURNS_TYPE': ('ID', {'VALUE': 'Object'}, 26),
            'NAMESPACE': self.name
            }
        )


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

i = PyettyInterpreter(tree, '_main', {}, {}, lexer=lexer, parser=parser)
if debug:
    print('== OUTPUT ==')
i.run()

if debug:
    print('\n== DEBUG ==')
    print(f'[_main]\nLocal:')
    pprint(i.locals)
    print('\nMethods:')
    pprint(i.methods)

    for method in i.methods:
        if type(i.methods[method]) == dict:
            try:
                print(f'\n[CLASS_{method}]\nLocal:')
                pprint(i.locals[method])
            except KeyError:
                print(f'\n[CLASS_{method}]\nLocal:\n')
        else:
            print(f'\n[_{method}]\nLocal:\n{i.methods[method].locals}\n')

    while True:
        code = input("⠕ ")
        try:
            tree = parser.parse(lexer.tokenize(code))
            i = PyettyInterpreter(tree, '_main', i.locals, {},
                              lexer=lexer, parser=parser, methods=i.methods)
            i.run()
        except Exception as e:
            print(e)