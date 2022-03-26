from sly import Lexer, Parser
import pprint
import logging
import sys


class PyettyLexer(Lexer):

    tokens = {
        ID,
        FLOAT,
        INT,
        FUNC,
        CLASS,
        NAMESPACE,
        STRING,
        EQ_GREATER,
        EQ_LESS,
        EQEQ,
        PYTHON_CODE,
        COLON_COLON,
        IF,
        ELSE,
        TRUE,
        FALSE,
        NOT_EQEQ,
        WHILE,
        BREAK,
        SKIP,
        FOR,
        IN,
        DEL,
        RETURN,
        NULL,
        EQ_ADD,
        EQ_SUB,
        EQ_MUL,
        EQ_DIV,
        EQ_MOD,
        IMPORT,
        LIMPORT,
        SANDBOX,
        FARROW,
        TARROW,
        LET,
        TELSE,
        PYTHON_CODE_EXEC,
        OF,
        GLOBAL
    }
    literals = {
        "+",
        "-",
        "*",
        "/",
        "%",
        "|",
        "&",
        "!",
        ">",
        "<",
        "=",
        "(",
        ")",
        "{",
        "}",
        ";",
        ",",
        ":",
        "[",
        "]",
        "\\",
        ".",
        "?"
    }

    ignore = " \t"
    ignore_comment_slash = r"//.*"

    FLOAT = r"\d*\.\d+"
    INT = r"\d+"

    PYTHON_CODE = r"\$`[.\W\w]*?`"
    PYTHON_CODE_EXEC = r"\$e`[.\W\w]*?`"
    STRING = r"(\".*?(?<!\\)(\\\\)*\"|'.*?(?<!\\)(\\\\)*')"
    ID = r"(--[a-zA-Z_]([a-zA-Z0-9_]|!)*--|[a-zA-Z_]([a-zA-Z0-9_]|!)*)"
    ID["func"] = FUNC
    ID["class"] = CLASS
    ID["namespace"] = NAMESPACE
    ID["break"] = BREAK
    ID["skip"] = SKIP
    ID["true"] = TRUE
    ID["false"] = FALSE
    ID["while"] = WHILE
    ID["for"] = FOR
    ID["in"] = IN
    ID["if"] = IF
    ID["else"] = ELSE
    ID["del"] = DEL
    ID["null"] = NULL
    ID["return"] = RETURN
    ID["import"] = IMPORT
    ID["limport"] = LIMPORT
    ID["sandbox"] = SANDBOX
    ID["let"] = LET
    ID["of"] = OF
    ID["globals"] = GLOBAL

    TARROW = r'->'
    FARROW = r'\=\=>'
    TELSE = r'\|\|'
    COLON_COLON = r"::"
    EQEQ = "=="
    NOT_EQEQ = r"!="
    EQ_GREATER = r"=>"
    EQ_LESS = r"=<"
    EQ_ADD = r"\+="
    EQ_SUB = r"-="
    EQ_MUL = r"\*="
    EQ_DIV = r"/="
    EQ_MOD = r"%="

    @_(r"\n+")
    def ignore_newline(self, t):
        self.lineno += len(t.value)


class PyettyParser(Parser):
    tokens = PyettyLexer.tokens
    debugfile = "parser.out"
    log = logging.getLogger()
    log.setLevel(logging.ERROR)
    #syntax_error_obj = syntax_error()

    precedence = (
        ("left", EMPTY),
        ("left", ","),
        ("right", "="),
        ("left", "|"),
        ("left", "&"),
        ("left", EQEQ, NOT_EQEQ),
        ("left", EQ_LESS, EQ_GREATER, "<", ">"),
        ("left", "+", "-"),
        ("left", "*", "/", "%"),
        ("right", UMINUS, UPLUS),
        ("right", "!"),
        ("left", COLON_COLON),
    )

    # Program START
    @_("program statement")
    def program(self, p):
        return p.program + (p.statement,)

    @_("statement")
    def program(self, p):
        return (p.statement,)

    @_("empty")
    def program(self, p):
        return ()

    # Program END
    ###########################################################################
    # Statements START

    @_("function_declaration")
    def statement(self, p):
        return p.function_declaration + ()

    @_("class_declaration")
    def statement(self, p):
        return p.class_declaration

    @_("function_call_statement")
    def statement(self, p):
        return p.function_call_statement

    @_("class_attribute_assignment")
    def statement(self, p):
        return p.class_attribute_assignment

    @_("conditional")
    def statement(self, p):
        return p.conditional

    @_("while_loop")
    def statement(self, p):
        return p.while_loop

    @_("python_code_statement")
    def statement(self, p):
        return p.python_code_statement

    @_("variable_assignment")
    def statement(self, p):
        return p.variable_assignment

    @_("break_statement")
    def statement(self, p):
        return p.break_statement

    @_("for_loop")
    def statement(self, p):
        return p.for_loop

    @_("delete_statement")
    def statement(self, p):
        return p.delete_statement

    @_("return_statement")
    def statement(self, p):
        return p.return_statement

    @_("variable_operation")
    def statement(self, p):
        return p.variable_operation

    @_("import_statement")
    def statement(self, p):
        return p.import_statement

    @_("sandbox")
    def statement(self, p):
        return p.sandbox

    # Statements END
    ###########################################################################
    # Statment syntax START

    @_("LIMPORT expression ';'")
    def sandbox(self, p):
        return ("LIMPORT", {"EXPRESSION": p.expression}, p.lineno)

    @_("SANDBOX '{' program '}'")
    def sandbox(self, p):
        return ("SANDBOX", {"PROGRAM": p.program}, p.lineno)

    @_("function_call ';'")
    def function_call_statement(self, p):
        return p.function_call

    @_("python_code ';'")
    def python_code_statement(self, p):
        return p.python_code

    @_("BREAK ';'")
    def break_statement(self, p):
        return ("BREAK", p.lineno)
    
    @_("SKIP ';'")
    def break_statement(self, p):
        return ("SKIP", p.lineno)

    @_("RETURN expression ';'")
    def return_statement(self, p):
        return ("RETURN", {"EXPRESSION": p.expression}, p.lineno)

    @_("expression '(' function_arguments ')'")
    def function_call(self, p):
        return (
            "FUNCTION_CALL",
            {"FUNCTION_ARGUMENTS": p.function_arguments, "ID": p.expression},
            p.lineno,
        )

    @_("expression '(' function_arguments ')' FARROW '{' program '}'")
    def function_call(self, p):
        return (
            "FUNCTION_CALL",
            {"FUNCTION_ARGUMENTS": p.function_arguments, "ID": p.expression,
             "ONCOMPLETE": p.program},
            p.lineno,
        )
    
    @_("'?' ';'")
    def debug_call(self, p):
        return (
            "DEBUG_CALL",
            {"VALUE": {}},
            p.lineno,
        )

    @_("expression '(' empty ')'")
    def function_call(self, p):
        return (
            "FUNCTION_CALL",
            {"FUNCTION_ARGUMENTS": {}, "ID": p.expression},
            p.lineno,
        )

    @_("expression '(' empty ')' FARROW '{' program '}'")
    def function_call(self, p):
        return (
            "FUNCTION_CALL",
            {
                "FUNCTION_ARGUMENTS": {},
                "ID": p.expression,
                "ONCOMPLETE": p.program
            },
            p.lineno,
        )

    @_("FUNC ID '(' function_arguments ')' '{' program '}' TARROW expression")
    def function_declaration(self, p):
        return (
            "FUNCTION_DECLARATION",
            {
                "FUNCTION_ARGUMENTS": p.function_arguments,
                "ID": p.ID,
                "PROGRAM": p.program,
                "RETURNS_TYPE": p.expression
            },
            p.lineno,
        )

    @_("FUNC ID COLON_COLON ID '(' function_arguments ')' '{' program '}' TARROW expression")
    def function_declaration(self, p):
        return (
            "FUNCTION_DECLARATION",
            {
                "FUNCTION_ARGUMENTS": p.function_arguments,
                "NAMESPACE": p.ID0,
                "ID": p.ID1,
                "PROGRAM": p.program,
                "RETURNS_TYPE": p.expression
            },
            p.lineno,
        )
    
    @_("FUNC ID COLON_COLON ID '(' empty ')' '{' program '}' TARROW expression")
    def function_declaration(self, p):
        return (
            "FUNCTION_DECLARATION",
            {"FUNCTION_ARGUMENTS": {}, "ID": p.ID1, "PROGRAM": p.program, "NAMESPACE": p.ID0,
                "RETURNS_TYPE": p.expression},
            p.lineno,
        )

    @_("FUNC ID '(' empty ')' '{' program '}' TARROW expression")
    def function_declaration(self, p):
        return (
            "FUNCTION_DECLARATION",
            {"FUNCTION_ARGUMENTS": {}, "ID": p.ID, "PROGRAM": p.program,
                "RETURNS_TYPE": p.expression},
            p.lineno,
        )

    @_("positional_args")
    def function_arguments(self, p):
        return {"POSITIONAL_ARGS": p.positional_args}

    @_("positional_args ',' kwargs")
    def function_arguments(self, p):
        return {"POSITIONAL_ARGS": p.positional_args, "KWARGS": p.kwargs}

    @_("kwargs")
    def function_arguments(self, p):
        return {"KWARGS": p.kwargs}

    @_("CLASS ID '{' program '}'")
    def class_declaration(self, p):
        return ("CLASS_DECLARATION", {"ID": p.ID, "PROGRAM": p.program}, p.lineno)

    @_("NAMESPACE ID '{' program '}'")
    def class_declaration(self, p):
        return ("CLASS_DECLARATION", {"ID": p.ID, "PROGRAM": p.program}, p.lineno)

    @_("FOR expression IN expression '{' program '}'")
    def for_loop(self, p):
        return (
            "FOR",
            {
                "PROGRAM": p.program,
                "VARIABLE": p.expression0,
                "ITERABLE": p.expression1,
            },
            p.lineno,
        )

    @_("WHILE '(' expression ')' '{' program '}'")
    def while_loop(self, p):
        return ("WHILE", {"PROGRAM": p.program, "CONDITION": p.expression}, p.lineno)

    @_("positional_args ',' expression")
    def positional_args(self, p):
        return p.positional_args + (p.expression,)

    @_("expression")
    def positional_args(self, p):
        return (p.expression,)

    @_("kwargs ',' id '=' expression")
    def kwargs(self, p):
        return p.kwargs + ({"ID": p.id, "EXPRESSION": p.expression},)

    @_("ID '=' expression")
    def kwargs(self, p):
        return ({"ID": p.ID, "EXPRESSION": p.expression},)

    @_("LET ID '=' expression ';'")
    def variable_assignment(self, p):
        return (
            "VARIABLE_ASSIGNMENT",
            {"ID": p.ID, "EXPRESSION": p.expression},
            p.lineno,
        )

    @_("LET get_index '=' expression ';'")
    def variable_assignment(self, p):
        return (
            "VARIABLE_ASSIGNMENT",
            {"ID": p.get_index, "EXPRESSION": p.expression},
            p.lineno,
        )

    @_("ID EQ_ADD expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.ID, "EXPRESSION": p.expression, "OPERATION": "ADD"},
            p.lineno,
        )

    @_("get_index EQ_ADD expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.get_index, "EXPRESSION": p.expression, "OPERATION": "ADD"},
            p.lineno,
        )

    @_("ID EQ_SUB expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.ID, "EXPRESSION": p.expression, "OPERATION": "SUB"},
            p.lineno,
        )

    @_("get_index EQ_SUB expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.get_index, "EXPRESSION": p.expression, "OPERATION": "SUB"},
            p.lineno,
        )

    @_("ID EQ_MUL expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.ID, "EXPRESSION": p.expression, "OPERATION": "MUL"},
            p.lineno,
        )

    @_("get_index EQ_MUL expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.get_index, "EXPRESSION": p.expression, "OPERATION": "MUL"},
            p.lineno,
        )

    @_("ID EQ_MOD expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.ID, "EXPRESSION": p.expression, "OPERATION": "MOD"},
            p.lineno,
        )

    @_("get_index EQ_MOD expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.get_index, "EXPRESSION": p.expression, "OPERATION": "MOD"},
            p.lineno,
        )

    @_("ID EQ_DIV expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.ID, "EXPRESSION": p.expression, "OPERATION": "DIV"},
            p.lineno,
        )

    @_("get_index EQ_DIV expression ';'")
    def variable_operation(self, p):
        return (
            "VARIABLE_OPERATION",
            {"ID": p.get_index, "EXPRESSION": p.expression, "OPERATION": "DIV"},
            p.lineno,
        )

    @_("class_attribute '=' expression ';'")
    def class_attribute_assignment(self, p):
        return (
            "CLASS_ATTRIBUTE_ASSIGNMENT",
            {"CLASS_ATTRIBUTE": p.class_attribute, "EXPRESSION": p.expression},
            p.lineno,
        )

    @_("if_statement")
    def conditional(self, p):
        return (
            "CONDITIONAL",
            {"IF": p.if_statement, "ELSE_IF": (
                None, None), "ELSE": (None, None)},
            p.if_statement[2],
        )

    @_("if_statement else_if_loop")
    def conditional(self, p):
        return (
            "CONDITIONAL",
            {"IF": p.if_statement, "ELSE_IF": p.else_if_loop,
                "ELSE": (None, None)},
            p.if_statement[2],
        )

    @_("if_statement else_if_loop else_statement")
    def conditional(self, p):
        return (
            "CONDITIONAL",
            {"IF": p.if_statement, "ELSE_IF": p.else_if_loop,
                "ELSE": p.else_statement},
            p.if_statement[2],
        )

    @_("if_statement else_statement")
    def conditional(self, p):
        return (
            "CONDITIONAL",
            {"IF": p.if_statement, "ELSE_IF": (
                None, None), "ELSE": p.else_statement},
            p.if_statement[2],
        )

    @_("IF '(' expression ')' '{' program '}'")
    def if_statement(self, p):
        return ("IF", {"CODE": p.program, "CONDITION": p.expression}, p.lineno)

    @_("else_if_loop else_if_statement")
    def else_if_loop(self, p):
        return p.else_if_loop + (p.else_if_statement,)

    @_("else_if_statement")
    def else_if_loop(self, p):
        return ("ELSE_IF", p.else_if_statement)

    @_("ELSE IF '(' expression ')' '{' program '}'")
    def else_if_statement(self, p):
        return ({"CODE": p.program, "CONDITION": p.expression}, p.lineno)

    @_("ELSE '{' program '}'")
    def else_statement(self, p):
        return ("ELSE", {"CODE": p.program}, p.lineno)

    @_("DEL ID ';'")
    def delete_statement(self, p):
        return ("DEL", {"ID": p.ID}, p.lineno)

    @_("IMPORT expression ';'")
    def import_statement(self, p):
        return ("IMPORT", {"EXPRESSION": p.expression}, p.lineno)

    @_("'.' GLOBAL ';'")
    def import_statement(self, p):
        return ("GLOBALS", {"VALUE":""}, p.lineno)

    # Statment syntax END
    ###########################################################################
    # Expression START

    @_("ID OF ID")
    def expression(self, p):
        return ("TYPED", {"ID": p.ID0, "TYPE": p.ID1}, p.lineno)

    @_("'-' expression %prec UMINUS")
    def expression(self, p):
        return ("NEG", p.expression)

    @_("'+' expression %prec UPLUS")
    def expression(self, p):
        return ("POS", p.expression)

    @_("expression '+' expression")
    def expression(self, p):
        return ("ADD", p[0], p[2])

    @_("expression '-' expression")
    def expression(self, p):
        return ("SUB", p[0], p[2])

    @_("expression '/' expression")
    def expression(self, p):
        return ("DIV", p[0], p[2])

    @_("expression '*' expression")
    def expression(self, p):
        return ("MUL", p[0], p[2])

    @_("expression '%' expression")
    def expression(self, p):
        return ("MOD", p[0], p[2])

    @_("expression EQEQ expression")
    def expression(self, p):
        return ("EQEQ", p[0], p[2])

    @_("expression NOT_EQEQ expression")
    def expression(self, p):
        return ("NOT_EQEQ", p[0], p[2])

    @_("expression EQ_LESS expression")
    def expression(self, p):
        return ("EQ_LESS", p[0], p[2])

    @_("expression EQ_GREATER expression")
    def expression(self, p):
        return ("EQ_GREATER", p[0], p[2])

    @_("expression '|' expression")
    def expression(self, p):
        return ("OR", p[0], p[2])

    @_("expression '&' expression")
    def expression(self, p):
        return ("AND", p[0], p[2])

    @_("'!' expression")
    def expression(self, p):
        return ("NOT", p.expression)

    @_("expression '<' expression")
    def expression(self, p):
        return ("LESS", p[0], p[2])

    @_("expression '>' expression")
    def expression(self, p):
        return ("GREATER", p[0], p[2])

    @_("'(' expression ')'")
    def expression(self, p):
        return p.expression

    @_("python_code")
    def expression(self, p):
        return p.python_code

    @_("function_call")
    def expression(self, p):
        return p.function_call

    @_("get_index")
    def expression(self, p):
        return p.get_index

    @_("null")
    def expression(self, p):
        return p.null

    @_("int")
    def expression(self, p):
        return p.int

    @_("float")
    def expression(self, p):
        return p.float

    @_("bool")
    def expression(self, p):
        return p.bool

    @_("string")
    def expression(self, p):
        return p.string

    @_("id")
    def expression(self, p):
        return p.id

    @_("class_attribute")
    def expression(self, p):
        return p.class_attribute

    @_("_tuple")
    def expression(self, p):
        return p._tuple

    @_("_list")
    def expression(self, p):
        return p._list

    @_("_numpy")
    def expression(self, p):
        return p._numpy

    @_("assoc_array")
    def expression(self, p):
        return p.assoc_array

    # Expression END
    ###########################################################################
    # Intermediate expression START

    @_("NULL")
    def null(self, p):
        return ("NULL", "NULL")

    @_("expression '[' expression ']'")
    def get_index(self, p):
        return ("GET_INDEX", {"EXPRESSION": p.expression0, "INDEX": p.expression1}, p.lineno)

    @_("'{' positional_args '}'")
    def _tuple(self, p):
        return ("TUPLE", {"ITEMS": p.positional_args})

    @_("'{' positional_args ',' '}'")
    def _tuple(self, p):
        return ("TUPLE", {"ITEMS": p.positional_args})

    @_("'[' positional_args ']'")
    def _list(self, p):
        return ("LIST", {"ITEMS": p.positional_args})

    @_("'[' positional_args ',' ']'")
    def _list(self, p):
        return ("LIST", {"ITEMS": p.positional_args})

    @_("'(' items ')'")
    def _numpy(self, p):
        return ("NUMPY", {"ITEMS": p.items})

    @_("'(' items ',' ')'")
    def _numpy(self, p):
        return ("NUMPY", {"ITEMS": p.items})

    @_("'(' expression ',' ')'")
    def _numpy(self, p):
        return ("NUMPY", {"ITEMS": (p.expression,)})

    @_("'(' ')'")
    def _numpy(self, p):
        return ("NUMPY", {"ITEMS": ()})

    @_("'(' ',' ')'")
    def _numpy(self, p):
        return ("NUMPY", {"ITEMS": ()})

    @_("items ',' expression")
    def items(self, p):
        return p.items + (p.expression,)

    @_("expression ',' expression")
    def items(self, p):
        return (p.expression,)

    @_("INT")
    def int(self, p):
        return ("INT", {"VALUE": p.INT})

    @_("STRING")
    def string(self, p):
        return ("STRING", {"VALUE": p.STRING[1:-1]})

    @_("FLOAT")
    def float(self, p):
        return ("FLOAT", {"VALUE": p.FLOAT})

    @_("TRUE")
    def bool(self, p):
        return ("BOOL", {"VALUE": p.TRUE})

    @_("FALSE")
    def bool(self, p):
        return ("BOOL", {"VALUE": p.FALSE})

    @_("expression COLON_COLON ID")
    def class_attribute(self, p):
        return ("CLASS_ATTRIBUTE", {"CLASS": p[0], "ATTRIBUTE": p[2]}, p.lineno)

    @_("ID")
    def id(self, p):
        return ("ID", {"VALUE": p.ID}, p.lineno)

    @_(r"'\' assoc_array_items '\'")
    def assoc_array(self, p):
        return ("ASSOC_ARRAY", {"ITEMS": p.assoc_array_items})

    @_("assoc_array_items ',' expression ':' expression")
    def assoc_array_items(self, p):
        return p.assoc_array_items + ((p.expression0, p.expression1),)

    @_("expression ':' expression")
    def assoc_array_items(self, p):
        return ((p.expression0, p.expression1),)

    @_("PYTHON_CODE")
    def python_code(self, p):
        return ("PYTHON_CODE", {"CODE": p.PYTHON_CODE[2:-1]})

    @_("PYTHON_CODE_EXEC")
    def python_code(self, p):
        return ("PYTHON_CODE_EXEC", {"CODE": p.PYTHON_CODE_EXEC[3:-1]})

    @_("%prec EMPTY")
    def empty(self, p):
        pass

    # Intermediate expression END
    ###########################################################################
    # Syntax error START


class Locale:
    def __init__(self, d):
        for _ in d:
            #print('Adding',_)
            if "VALUE" in d[_]:
                setattr(self, _, d[_]['VALUE'])
        #print(self.__dict__)


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
            'CONDITIONAL': self.proc_conditional
        }

    def run(self, tree=None):
        pg = self.program
        if tree: pg = tree
        if pg:
            for line in pg:
                # print(f'Action: [{line[0]}]')
                if line[0] in self.defaults:
                    self.defaults[line[0]](line[1])

    def eval_import(self, tree):
        path = self.eval_expr(tree['EXPRESSION'])
        lib = path.split('/')[1].strip('.ppy')

        with open(path, 'r') as f:
            tree = self.parser.parse(self.lexer.tokenize(f.read()))

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
            *args
        )
        ImportInstance.run()

        self.locals[lib] = ImportInstance.locals

        #print(f'Loaded {lib}', ImportInstance.locals, ImportInstance.methods,sep='\n')

        # print(self.locals, i.locals, '\n', self.methods, i.methods)

    def create_class(self, tree):
        # print(tree)
        self.locals[tree['ID']] = {
        }

    def proc_conditional(self, tree):
        # print(tree)
        _if = tree['IF'][1] if tree['IF'][0] else None
        _else = tree['ELSE'][1] if tree['ELSE'][0] else None
        _elif = tree['ELSE_IF'][1][0] if tree['ELSE_IF'][0] else None

        if _if:
            _code = _if['CODE']
            _condition = _if['CONDITION']
            if self.eval_expr(_condition):
                self.run(tree=_code)
            
            if _elif:
                _code = _elif['CODE']
                _condition = _elif['CONDITION']
                if self.eval_expr(_condition):
                    self.run(tree=_code)
            
            if _else:
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
            try:
                ticked = self.eval_expr(tree['EXPRESSION'])
                packed = {
                    "TYPE": tree["EXPRESSION"][0],
                    "VALUE": ticked
                }
            except:
                raise Exception(f'Unbound call {tree["EXPRESSION"][0]} when creating {tree["ID"]}')
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

    def run_function(self, tree):

        if 'ONCOMPLETE' in tree:
            _oncomplete = tree['ONCOMPLETE']
        else:
            _oncomplete = None

        if tree['ID'][0] == 'CLASS_ATTRIBUTE':
            _func = tree['ID'][1]['ATTRIBUTE']
            _class = tree['ID'][1]['CLASS'][1]['VALUE']


            if _class not in self.locals:
                raise Exception(f'No class {_class} in {self.locals}!')

            if _func not in self.locals[_class][_class]:
                raise Exception('!!! Unbound method, tree:',
                                tree, self.locals[_class])
            _vars = []
            if 'POSITIONAL_ARGS' in tree['FUNCTION_ARGUMENTS']:
                for var in tree['FUNCTION_ARGUMENTS']['POSITIONAL_ARGS']:
                    if var[0] not in self.defaults:
                        resed = self.resolve_var(var)
                    else:
                        resed = self.defaults[var[0]](var[1])
                    if not resed:
                        raise Exception(
                            f'Variable "{var[0]}" "{self.defaults}" resolution failed. Does it exist?')
                    _vars.append(resed)

            # Run it!
            returns = self.locals[_class][_class][_func].run(_vars, {})

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
                raise Exception(f'Couldnt resolve variable :( {tree[1]["VALUE"]} {self.locals}')
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

    def eval_expr(self, tree):
        print(tree)
        # New variable type matching
        if type(tree) == dict:
            # Ignore resolving or type matching!
            # This is because we fed in a raw type!
            # A raw type doesn't require type-matching!
            return tree['VALUE']

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
            'LIST': self.eval_list,
            'GET_INDEX': self.eval_get_index,
            # Math
            'ADD': self.eval_add,
            'SUB': self.eval_sub,
            'MUL': self.eval_mul,
            'DIV': self.eval_div,
            # ID
            'ID': self.resolve_var,
            # Logik
            'NOT_EQEQ': self.eval_not_equal,
            'EQEQ': self.eval_equal
        }
        #try:
        if _act in self.procs:
            if _act == 'ID':
                _val = self.resolve_var(tree)
            else:
                _val = self.procs[_act](_body)
        #except TypeError:
        #    raise Exception(f'Failed to hash _act? Act -> {_act} Tree -> {tree} Body -> {_body}')

        if _val is not None:
            return _val
        else:
            print(self.locals)
            raise Exception(
                f'Variable "{_body[0]}" was never declared or failed to resolve. Line: {_body[1]} Act:{_act}')

    def python_eval(self, tree):
        code = tree["CODE"]
        return eval(code, {'etty': Locale(self.locals)})

    def python_exec(self, tree):
        code = tree["CODE"]
        return exec(code, {'etty': Locale(self.locals)})


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
                # print(f'Action: [{line[0]}]')
                if line[0] in self.defaults:
                    self.defaults[line[0]](line[1])
            return 

        if len(args) < len(self.required_args):
            raise Exception('Not enough arguments supplied!')

        # Check if there are types!
        for i, arg in enumerate(self.required_args):
            # print('>',args[i]) # for debugging tuples AAAAAAAA

            if 'TYPE' in arg:
                # Check type
                # TYPES CAN ONLY BE CAPS!!! But they will be converted to CAPS AS WELL
                if type(args[i]) == dict:
                    if arg['TYPE'].upper() == args[i]['TYPE'].upper():
                        self.locals[arg['ID']] = {
                            'TYPE': arg['TYPE'].upper(),
                            'VALUE': args[i]['VALUE']
                        }
                elif type(args[i]) == tuple:
                    if arg['TYPE'].upper() == args[i][0].upper():
                        self.locals[arg['ID']] = {
                            'TYPE': arg['TYPE'].upper(),
                            'VALUE': args[i][1]['VALUE']
                        }
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


class Class(Function):

    def __init__(self, tree, name, local_args):
        super().__init__(tree, name, local_args, {})


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
    print(f'[_main]\nLocal:\n{i.locals}\nMethods:\n{i.methods}')

    for method in i.methods:
        if type(i.methods[method]) == dict:
            print(f'\n[CLASS_{method}]\nLocal:\n{i.locals[method]}\n')
        else:
            print(f'\n[_{method}]\nLocal:\n{i.methods[method].locals}\n')

    while True:
        code = input("⠕ ")
        tree = parser.parse(lexer.tokenize(code))
        i = PyettyInterpreter(tree, '_main', i.locals, {},
                              lexer=lexer, parser=parser, methods=i.methods)
        i.run()

"""program = neutron_interpreter.Process(tree, filename=path.abspath('example.ppy'))

program.objects.update(defult_functions[0])
program.global_items["OBJECTS"].update(defult_functions[1])

program.run()"""
