from __future__ import annotations
import re
import sys
from typing import List, Optional, Any, Dict, Tuple
from collections import deque
import graphviz

#Lexer
TOKEN_SPEC = [
    ('NUMBER',   r"\d+(?:\.\d+)?"),
    ('STRING',   r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\''),
    ('DEF',      r'\bdef\b'),
    ('RETURN',   r'\breturn\b'),
    ('IF',       r'\bif\b'),
    ('ELIF',     r'\belif\b'),
    ('ELSE',     r'\belse\b'),
    ('WHILE',    r'\bwhile\b'),
    ('FOR',      r'\bfor\b'),
    ('IN',       r'\bin\b'),
    ('NOT',      r'\bnot\b'),
    ('AND',      r'\band\b'),
    ('OR',       r'\bor\b'),
    ('IDENT',    r'[A-Za-z_][A-Za-z0-9_]*'),
    ('NEWLINE',  r'\n'),
    ('SKIP',     r'[ \t]+'),
    ('COMMENT',  r'\#.*'),
    ('OP',       r'==|!=|<=|>=|//|\+|-|\*|/|%|<|>|=|\(|\)|\[|\]|\{|\}|:|,|\.'),
]
TOKEN_RE = re.compile('|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC))

class Token:
    def __init__(self, type_, value, lineno, col):
        self.type = type_
        self.value = value
        self.lineno = lineno
        self.col = col
    def __repr__(self):
        return f"Token({self.type!r},{self.value!r},{self.lineno},{self.col})"


def lex(text: str) -> List[Token]:
    tokens: List[Token] = []
    lines = text.replace('\r\n','\n').split('\n')
    indent_stack = [0]
    lineno = 0
    for line in lines:
        lineno += 1
        if re.match(r'^\s*($|#)', line):
            tokens.append(Token('NEWLINE','\n',lineno,0))
            continue
        m = re.match(r'^(?P<indent>\s*)', line)
        indent = len(m.group('indent').replace('\t','    '))
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token('INDENT','INDENT',lineno,0))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token('DEDENT','DEDENT',lineno,0))
            if indent != indent_stack[-1]:
                raise IndentationError(f"Indentation error on line {lineno}")
        pos = 0
        while pos < len(line):
            m = TOKEN_RE.match(line,pos)
            if not m:
                raise SyntaxError(f"Unexpected character: {line[pos]!r} at line {lineno}")
            typ = m.lastgroup
            val = m.group(typ)
            if typ == 'SKIP' or typ == 'COMMENT':
                pass
            elif typ == 'NEWLINE':
                tokens.append(Token('NEWLINE','\n',lineno,pos))
            else:
                tokens.append(Token(typ,val,lineno,pos))
            pos = m.end()
        tokens.append(Token('NEWLINE','\n',lineno,len(line)))
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token('DEDENT','DEDENT',lineno,0))
    tokens.append(Token('EOF','EOF',lineno,0))
    return tokens

#nodos AST
class Node:
    def __init__(self):
        self.code: List[str] = []
        self.place: Optional[str] = None
        self.type: Optional[str] = None
        self.truelist: List[int] = []
        self.falselist: List[int] = []
        self.nextlist: List[int] = []

class Program(Node):
    def __init__(self, stmts):
        super().__init__()
        self.stmts = stmts

class StmtList(Node):
    def __init__(self, stmts: List[Node]):
        super().__init__()
        self.stmts = stmts

class Assign(Node):
    def __init__(self, target, expr):
        super().__init__()
        self.target = target
        self.expr = expr

class Return(Node):
    def __init__(self, expr: Optional[Node]):
        super().__init__()
        self.expr = expr

class If(Node):
    def __init__(self, cond, then_block, elifs: List[Tuple[Node,Node]], else_block: Optional[Node]):
        super().__init__()
        self.cond = cond
        self.then_block = then_block
        self.elifs = elifs
        self.else_block = else_block

class While(Node):
    def __init__(self, cond, block):
        super().__init__()
        self.cond = cond
        self.block = block

class For(Node):
    def __init__(self, varname, iterable, block):
        super().__init__()
        self.varname = varname
        self.iterable = iterable
        self.block = block

class FuncDef(Node):
    def __init__(self, name, params, block):
        super().__init__()
        self.name = name
        self.params = params
        self.block = block

class Call(Node):
    def __init__(self, name, args):
        super().__init__()
        self.name = name
        self.args = args

class Print(Node):
    def __init__(self, args):
        super().__init__()
        self.args = args

class BinOp(Node):
    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

class UnOp(Node):
    def __init__(self, op, operand):
        super().__init__()
        self.op = op
        self.operand = operand

class Literal(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

class Identifier(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name

class ListLiteral(Node):
    def __init__(self, items):
        super().__init__()
        self.items = items

class DictLiteral(Node):
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

#Parser
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current = self.tokens[self.pos]

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Token('EOF','EOF',-1,-1)

    def expect(self, typ):
        if self.current.type == typ:
            val = self.current.value
            self.advance()
            return val
        raise SyntaxError(f"Expected {typ} at {self.current.lineno}:{self.current.col}, got {self.current.type}")

    def maybe(self, typ):
        if self.current.type == typ:
            return self.expect(typ)
        return None

    def parse(self) -> Program:
        stmts = self.parse_stmt_list()
        if self.current.type != 'EOF':
            raise SyntaxError('Expected EOF')
        return Program(stmts)

    def parse_stmt_list(self):
        stmts = []
        while self.current.type not in ('DEDENT','EOF'):
            if self.current.type == 'NEWLINE':
                self.advance(); continue
            stmts.append(self.parse_stmt())
        return StmtList(stmts)

    def parse_stmt(self):
        if self.current.type == 'DEF':
            return self.parse_funcdef()
        if self.current.type == 'IF':
            return self.parse_if()
        if self.current.type == 'WHILE':
            return self.parse_while()
        if self.current.type == 'FOR':
            return self.parse_for()
        if self.current.type == 'RETURN':
            self.advance()
            if self.current.type != 'NEWLINE':
                expr = self.parse_expression()
            else:
                expr = None
            self.expect('NEWLINE')
            return Return(expr)
        node = self.parse_simple_stmt()
        self.expect('NEWLINE')
        return node

    def parse_simple_stmt(self):
        if self.current.type == 'IDENT':
            name = self.current.value
            saved_pos = self.pos
            self.advance()
            if self.current.type == 'OP' and self.current.value == '=':
                self.pos = saved_pos
                self.current = self.tokens[self.pos]
                return self.parse_assignment()
            else:
                self.pos = saved_pos
                self.current = self.tokens[self.pos]
        expr = self.parse_expression()
        if isinstance(expr, Call):
            return expr
        return expr

    def parse_assignment(self):
        target_name = self.expect('IDENT')
        self.expect('OP')  # should be '='
        expr = self.parse_expression()
        return Assign(Identifier(target_name), expr)

    def parse_funcdef(self):
        self.expect('DEF')
        name = self.expect('IDENT')
        self.expect('OP')  # (
        params = []
        if self.current.type != 'OP' or self.current.value != ')':
            while True:
                p = self.expect('IDENT')
                params.append(p)
                if self.current.type == 'OP' and self.current.value == ',':
                    self.advance(); continue
                break
        self.expect('OP')  # )
        self.expect('OP')  # :
        self.expect('NEWLINE')
        self.expect('INDENT')
        block = self.parse_stmt_list()
        self.expect('DEDENT')
        return FuncDef(name, params, block)

    def parse_if(self):
        self.expect('IF')
        cond = self.parse_expression()
        self.expect('OP')  # ':'
        self.expect('NEWLINE')
        self.expect('INDENT')
        then_block = self.parse_stmt_list()
        self.expect('DEDENT')
        elifs = []
        while self.current.type == 'ELIF':
            self.advance()
            c = self.parse_expression()
            self.expect('OP')  # ':'
            self.expect('NEWLINE')
            self.expect('INDENT')
            b = self.parse_stmt_list()
            self.expect('DEDENT')
            elifs.append((c,b))
        else_block = None
        if self.current.type == 'ELSE':
            self.advance()
            self.expect('OP')  # ':'
            self.expect('NEWLINE')
            self.expect('INDENT')
            else_block = self.parse_stmt_list()
            self.expect('DEDENT')
        return If(cond, then_block, elifs, else_block)

    def parse_while(self):
        self.expect('WHILE')
        cond = self.parse_expression()
        self.expect('OP')  # ':'
        self.expect('NEWLINE')
        self.expect('INDENT')
        block = self.parse_stmt_list()
        self.expect('DEDENT')
        return While(cond, block)

    def parse_for(self):
        self.expect('FOR')
        var = self.expect('IDENT')
        self.expect('IN')
        iterable = self.parse_expression()
        self.expect('OP')  # ':'
        self.expect('NEWLINE')
        self.expect('INDENT')
        block = self.parse_stmt_list()
        self.expect('DEDENT')
        return For(var, iterable, block)

    # Expresiones grammar
    def parse_expression(self):
        return self.parse_logic_or()

    def parse_logic_or(self):
        node = self.parse_logic_and()
        while self.current.type == 'OR':
            op = self.current.value; self.advance()
            right = self.parse_logic_and()
            node = BinOp('or', node, right)
        return node

    def parse_logic_and(self):
        node = self.parse_equality()
        while self.current.type == 'AND':
            op = self.current.value; self.advance()
            right = self.parse_equality()
            node = BinOp('and', node, right)
        return node

    def parse_equality(self):
        node = self.parse_comparison()
        while self.current.type == 'OP' and self.current.value in ('==','!='):
            op = self.current.value; self.advance()
            right = self.parse_comparison()
            node = BinOp(op, node, right)
        return node

    def parse_comparison(self):
        node = self.parse_term()
        while self.current.type == 'OP' and self.current.value in ('<','>','<=','>='):
            op = self.current.value; self.advance()
            right = self.parse_term()
            node = BinOp(op, node, right)
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.current.type == 'OP' and self.current.value in ('+','-'):
            op = self.current.value; self.advance()
            right = self.parse_factor()
            node = BinOp(op, node, right)
        return node

    def parse_factor(self):
        node = self.parse_unary()
        while self.current.type == 'OP' and self.current.value in ('*','/','%','//'):
            op = self.current.value; self.advance()
            right = self.parse_unary()
            node = BinOp(op, node, right)
        return node

    def parse_unary(self):
        if self.current.type == 'OP' and self.current.value in ('+','-'):
            op = self.current.value; self.advance()
            node = self.parse_unary()
            return UnOp(op,node)
        if self.current.type == 'NOT':
            self.advance()
            node = self.parse_unary()
            return UnOp('not', node)
        return self.parse_primary()

    def parse_primary(self):
        if self.current.type == 'NUMBER':
            val = self.current.value; self.advance();
            return Literal(val)
        if self.current.type == 'STRING':
            val = self.current.value; self.advance();
            return Literal(val)
        if self.current.type == 'IDENT':
            name = self.current.value; self.advance()
            # function call?
            if self.current.type == 'OP' and self.current.value == '(':
                self.advance()
                args = []
                if not (self.current.type == 'OP' and self.current.value == ')'):
                    while True:
                        args.append(self.parse_expression())
                        if self.current.type == 'OP' and self.current.value == ',':
                            self.advance(); continue
                        break
                self.expect('OP')  # )
                return Call(name, args)
            return Identifier(name)
        if self.current.type == 'OP' and self.current.value == '(':
            self.advance()
            node = self.parse_expression()
            self.expect('OP')  # )
            return node
        if self.current.type == 'OP' and self.current.value == '[':
            self.advance()
            items = []
            if not (self.current.type == 'OP' and self.current.value == ']'):
                while True:
                    items.append(self.parse_expression())
                    if self.current.type == 'OP' and self.current.value == ',':
                        self.advance(); continue
                    break
            self.expect('OP')  # ]
            return ListLiteral(items)
        if self.current.type == 'OP' and self.current.value == '{':
            self.advance()
            pairs = []
            if not (self.current.type == 'OP' and self.current.value == '}'):
                while True:
                    k = self.parse_expression()
                    self.expect('OP')  # :
                    v = self.parse_expression()
                    pairs.append((k,v))
                    if self.current.type == 'OP' and self.current.value == ',':
                        self.advance(); continue
                    break
            self.expect('OP')  # }
            return DictLiteral(pairs)
        raise SyntaxError(f"Unexpected token {self.current}")

#Tabla de Simbolos
class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str,Dict[str,Any]]] = [{}]

    def push(self):
        self.scopes.append({})

    def pop(self):
        return self.scopes.pop()

    def add(self, name, kind='var', typ=None, location=None, params=None):
        self.scopes[-1][name] = {'kind':kind,'type':typ,'location':location or name,'params':params}

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope: return scope[name]
        return None

class CodeGenState:
    def __init__(self):
        self.code: List[str] = []
        self.tempcount = 0
        self.labelcount = 0
        self.symtab = SymbolTable()

    def newtemp(self):
        self.tempcount += 1
        name = f"t{self.tempcount}"
        self.symtab.add(name, kind='temp', location=name)
        return name

    def newlabel(self):
        self.labelcount += 1
        return f"L{self.labelcount}"

    def emit(self, instr: str) -> int:
        self.code.append(instr)
        return len(self.code)-1

    def makelist(self, i:int) -> List[int]:
        return [i]

    def merge(self,a:List[int], b:List[int]) -> List[int]:
        return a + b

    def backpatch(self, lst:List[int], label:str):
        for i in lst:
            self.code[i] = self.code[i].replace('?', label)

#Generador AST

def generate(node: Node, state: CodeGenState):
    if isinstance(node, Program):
        for s in node.stmts.stmts:
            generate(s, state)
        node.code = state.code
        return
    if isinstance(node, StmtList):
        for s in node.stmts:
            generate(s, state)
        return
    if isinstance(node, Assign):
        generate(node.expr, state)
        if not state.symtab.lookup(node.target.name):
            state.symtab.add(node.target.name, kind='var', location=node.target.name)
        state.emit(f"{node.target.name} = {node.expr.place}")
        return
    if isinstance(node, Return):
        if node.expr:
            generate(node.expr, state)
            state.emit(f"return {node.expr.place}")
        else:
            state.emit("return")
        return
    if isinstance(node, Print):
        for a in node.args:
            generate(a, state)
            state.emit(f"print {a.place}")
        return
    if isinstance(node, If):
        cg = state
        generate(node.cond, cg)
        L_end = cg.newlabel()
        nextlist_branches: List[int] = []
        L_then = cg.newlabel(); cg.emit(f"label {L_then}:")
        cg.backpatch(node.cond.truelist, L_then)
        generate(node.then_block, cg)
        idx = cg.emit("goto ?"); nextlist_branches.append(idx)
        cur_falselist = node.cond.falselist[:]
        for (cnd, blk) in node.elifs:
            L_elif = cg.newlabel(); cg.emit(f"label {L_elif}:")
            cg.backpatch(cur_falselist, L_elif)
            generate(cnd, cg)
            L_then2 = cg.newlabel(); cg.emit(f"label {L_then2}:")
            cg.backpatch(cnd.truelist, L_then2)
            generate(blk, cg)
            idx = cg.emit("goto ?"); nextlist_branches.append(idx)
            cur_falselist = cnd.falselist[:]
        if node.else_block:
            L_else = cg.newlabel(); cg.emit(f"label {L_else}:")
            cg.backpatch(cur_falselist, L_else)
            generate(node.else_block, cg)
        else:
            pass
        for i in nextlist_branches:
            cg.backpatch([i], L_end)
        cg.emit(f"label {L_end}:")
        return
    if isinstance(node, While):
        cg = state
        L_begin = cg.newlabel(); cg.emit(f"label {L_begin}:")
        generate(node.cond, cg)
        L_body = cg.newlabel(); cg.emit(f"label {L_body}:")
        cg.backpatch(node.cond.truelist, L_body)
        generate(node.block, cg)
        cg.emit(f"goto {L_begin}")
        L_end = cg.newlabel()
        cg.backpatch(node.cond.falselist, L_end)
        cg.emit(f"label {L_end}:")
        return
    if isinstance(node, For):
        cg = state
        generate(node.iterable, cg)
        tlist = cg.newtemp(); cg.emit(f"{tlist} = {node.iterable.place}")
        i = cg.newtemp(); cg.emit(f"{i} = 0")
        L_check = cg.newlabel(); cg.emit(f"label {L_check}:")
        cmp = cg.newtemp(); cg.emit(f"{cmp} = {i} >= len({tlist})")
        j = cg.emit(f"if {cmp} goto ?")
        elem = cg.newtemp(); cg.emit(f"{elem} = {tlist}[{i}]")
        if not state.symtab.lookup(node.varname):
            state.symtab.add(node.varname, kind='var', location=node.varname)
        cg.emit(f"{node.varname} = {elem}")
        L_body = cg.newlabel(); cg.emit(f"label {L_body}:")
        cg.backpatch([j], L_body)
        generate(node.block, cg)
        cg.emit(f"{i} = {i} + 1")
        cg.emit(f"goto {L_check}")
        L_end = cg.newlabel(); cg.emit(f"label {L_end}:")
        return
    if isinstance(node, FuncDef):
        cg = state
        label = f"func_{node.name}"
        state.symtab.add(node.name, kind='function', location=label, params=node.params)
        cg.emit(f"label {label}:")
        cg.symtab.push()
        for p in node.params:
            cg.symtab.add(p, kind='param', location=p)
        generate(node.block, cg)
        cg.emit("return")
        cg.symtab.pop()
        return
    if isinstance(node, Call):
        cg = state
        for a in node.args:
            generate(a, cg)
            cg.emit(f"param {a.place}")
        ret = cg.newtemp()
        cg.emit(f"{ret} = call {node.name}, {len(node.args)}")
        node.place = ret
        return
    if isinstance(node, BinOp):
        cg = state
        if node.op in ('and','or'):
            generate(node.left, cg)
            generate(node.right, cg)
            if node.op == 'and':
                Lr = cg.newlabel(); cg.emit(f"label {Lr}:")
                cg.backpatch(node.left.truelist, Lr)
                node.truelist = node.right.truelist
                node.falselist = cg.merge(node.left.falselist, node.right.falselist)
                node.code = node.left.code + node.right.code
            else:
                Lr = cg.newlabel(); cg.emit(f"label {Lr}:")
                cg.backpatch(node.left.falselist, Lr)
                node.truelist = cg.merge(node.left.truelist, node.right.truelist)
                node.falselist = node.right.falselist
                node.code = node.left.code + node.right.code
            node.type = 'bool'
            return
        if node.op in ('==','!=','<','>','<=','>='):
            generate(node.left, cg); generate(node.right, cg)
            i = cg.emit(f"if {node.left.place} {node.op} {node.right.place} goto ?")
            j = cg.emit("goto ?")
            node.truelist = cg.makelist(i); node.falselist = cg.makelist(j)
            node.code = []
            node.type = 'bool'
            return
        generate(node.left, cg); generate(node.right, cg)
        t = cg.newtemp()
        cg.emit(f"{t} = {node.left.place} {node.op} {node.right.place}")
        node.place = t
        node.type = 'number'
        return
    if isinstance(node, UnOp):
        cg = state
        generate(node.operand, cg)
        if node.op == 'not':
            node.truelist = node.operand.falselist
            node.falselist = node.operand.truelist
            node.code = node.operand.code
            node.type = 'bool'
            return
        t = cg.newtemp()
        if node.op == '-':
            cg.emit(f"{t} = 0 - {node.operand.place}")
        else:
            cg.emit(f"{t} = {node.operand.place}")
        node.place = t
        node.type = 'number'
        return
    if isinstance(node, Literal):
        cg = state
        t = cg.newtemp()
        cg.emit(f"{t} = {node.value}")
        node.place = t
        node.type = 'literal'
        return
    if isinstance(node, Identifier):
        sym = state.symtab.lookup(node.name)
        if not sym:
            state.symtab.add(node.name, kind='var', location=node.name)
        node.place = node.name
        return
    if isinstance(node, ListLiteral):
        cg = state
        t = cg.newtemp(); cg.emit(f"{t} = make_list()")
        for it in node.items:
            generate(it, cg)
            cg.emit(f"append {t}, {it.place}")
        node.place = t
        return
    if isinstance(node, DictLiteral):
        cg = state
        t = cg.newtemp(); cg.emit(f"{t} = make_dict()")
        for (k,v) in node.pairs:
            generate(k, cg); generate(v, cg)
            cg.emit(f"dict_put {t}, {k.place}, {v.place}")
        node.place = t
        return
    return

#Visualizacion AST

def ast_to_graph(node: Node) -> graphviz.Digraph:
    dot = graphviz.Digraph(comment='AST')
    counter = [0]
    def add(n: Node):
        nid = f"n{counter[0]}"; counter[0]+=1
        label = n.__class__.__name__
        if isinstance(n, Identifier): label += '\n' + n.name
        if isinstance(n, Literal): label += '\n' + str(n.value)
        if isinstance(n, BinOp): label += '\n' + n.op
        dot.node(nid, label)
        children = []
        if isinstance(n, Program): children = n.stmts.stmts
        elif isinstance(n, StmtList): children = n.stmts
        elif isinstance(n, Assign): children = [n.target, n.expr]
        elif isinstance(n, Return): children = [n.expr] if n.expr else []
        elif isinstance(n, If):
            children = [n.cond, n.then_block] + [c for pair in n.elifs for c in pair] + ([n.else_block] if n.else_block else [])
        elif isinstance(n, While): children = [n.cond, n.block]
        elif isinstance(n, For): children = [Identifier(n.varname), n.iterable, n.block]
        elif isinstance(n, FuncDef): children = [Identifier(n.name)] + [Identifier(p) for p in n.params] + [n.block]
        elif isinstance(n, Call): children = [Identifier(n.name)] + n.args
        elif isinstance(n, Print): children = n.args
        elif isinstance(n, BinOp): children = [n.left, n.right]
        elif isinstance(n, UnOp): children = [n.operand]
        elif isinstance(n, Literal): children = []
        elif isinstance(n, Identifier): children = []
        elif isinstance(n, ListLiteral): children = n.items
        elif isinstance(n, DictLiteral): children = [item for pair in n.pairs for item in pair]
        for c in children:
            if c is None: continue
            cid = add(c)
            dot.edge(nid,cid)
        return nid
    add(node)
    return dot

#Main + Ejemplo
EXAMPLE = '''
def sum_and_print(a, b):
    c = a + b
    if c > 10:
        print(c)
    else:
        c = c * 2
    return c

x = sum_and_print(3, 9)
print(x)
'''

def main():
    src = EXAMPLE
    tokens = lex(src)
    parser = Parser(tokens)
    ast = parser.parse()
    state = CodeGenState()
    generate(ast, state)
    # write TAC
    with open('tac.txt','w') as f:
        for i,l in enumerate(state.code):
            f.write(f"{i}: {l}\n")
    print('TAC written to tac.txt')
    # render AST
    dot = ast_to_graph(ast)
    dot.render('ast', format='png', cleanup=True)
    print('AST written to ast.png')

if __name__ == '__main__':
    main()
