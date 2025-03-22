# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
An Instruction represents an executable unit of a Python program.

Almost all simple statements correspond to Instructions (except for statements
like pass, continue, and break, whose effects are represented in the control-flow graph).

In addition to simple statements, assignments occurring implicitly (e.g. within
function or class definitions) also correspond to Instructions.

The complete set of places where Instructions occur in source are listed here:

1. <Instruction>  (Any node in INSTRUCTION_AST_NODES used as a statement.)
2. if <Instruction>: ... (elif is similar.)
3+4. for <Instruction> in <Instruction>: ...
5. while <Instruction>: ...
6. try: ... except <Instruction>: ...
7. (TODO: Test for "with <Instruction>:")
8. Each decorator is an Instruction.
9. Each default is an Instruction.
10. The assignment of the function definition to the function name is an Instruction.
11. Within function definitions, assignments to arguments are Instructions,
    and the body consists of multiple Instructions.
12. For class definitions, decorators and the assignment of the class to its name
    are Instructions.

Note: In Python 3, print and exec are treated as ordinary function calls (wrapped
in an ast.Call within an ast.Expr), so they no longer appear as distinct AST node types.
"""

import ast

# Types of accesses:
READ = "read"
WRITE = "write"

# Context lists
WRITE_CONTEXTS = (ast.Store, ast.Del, ast.Param, ast.AugStore)
READ_CONTEXTS = (ast.Load, ast.AugLoad)

# Sources of implicit writes:
CLASS = "class"
FUNCTION = "function"
ARGS = "args"
KWARG = "kwarg"
KWONLYARGS = "kwonlyargs"
VARARG = "vararg"
ITERATOR = "iter"
EXCEPTION = "exception"

# In Python 3, print and exec are handled as ordinary function calls.
INSTRUCTION_AST_NODES = (
    ast.Expr,  # expression statement
    ast.Assert,  # assert statement
    ast.Assign,  # assignment statement
    ast.AugAssign,  # augmented assignment statement
    ast.Delete,  # delete statement
    ast.Return,  # return statement
    ast.Raise,  # raise statement
    ast.Import,  # import statement
    ast.ImportFrom,
    ast.Global,  # global statement
)

# https://docs.python.org/3/reference/simple_stmts.html
SIMPLE_STATEMENT_AST_NODES = INSTRUCTION_AST_NODES + (
    ast.Pass,  # pass statement
    ast.Break,  # break statement
    ast.Continue,  # continue statement
)


def _canonicalize(node):
    if isinstance(node, list) and len(node) == 1:
        return _canonicalize(node[0])
    if isinstance(node, ast.Module):
        return _canonicalize(node.body)
    if isinstance(node, ast.Expr):
        return _canonicalize(node.value)
    return node


def represent_same_program(node1, node2):
    """
    Determine whether AST nodes node1 and node2 represent the same program syntactically.

    Two programs are considered syntactically equivalent if their ASTs are equivalent,
    ignoring differences in the context field of Name nodes.
    """
    node1 = _canonicalize(node1)
    node2 = _canonicalize(node2)

    if type(node1) != type(node2):  # Use type equality
        return False
    if not isinstance(node1, ast.AST):
        return node1 == node2

    fields1 = list(ast.iter_fields(node1))
    fields2 = list(ast.iter_fields(node2))
    if len(fields1) != len(fields2):
        return False

    for (field1, value1), (field2, value2) in zip(fields1, fields2):
        if field1 == "ctx":
            continue
        if field1 != field2 or type(value1) is not type(value2):
            return False
        if isinstance(value1, list):
            for item1, item2 in zip(value1, value2):
                if not represent_same_program(item1, item2):
                    return False
        elif not represent_same_program(value1, value2):
            return False

    return True


class AccessVisitor(ast.NodeVisitor):
    """
    Visitor that computes an ordered list of accesses.

    Accesses are collected in depth-first order according to the AST field order.
    For assignment nodes, the right-hand side is processed before the left-hand side.
    """

    def __init__(self):
        self.accesses = []

    def visit_Name(self, node):
        self.accesses.append(node)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node):
        # Process value first for augmented assignments.
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            # Record a read access for the target.
            self.accesses.append(("read", node.target, node))
        self.visit(node.target)


def get_accesses_from_ast_node(node):
    """Return all accesses in the AST node in depth-first order."""
    visitor = AccessVisitor()
    visitor.visit(node)
    return visitor.accesses


def get_reads_from_ast_node(ast_node):
    """Return all read accesses from the AST node."""
    return [
        access
        for access in get_accesses_from_ast_node(ast_node)
        if access_is_read(access)
    ]


def get_writes_from_ast_node(ast_node):
    """Return all write accesses from the AST node."""
    return [
        access
        for access in get_accesses_from_ast_node(ast_node)
        if access_is_write(access)
    ]


def create_writes(node, parent=None):
    if isinstance(node, ast.AST):
        return [("write", n, parent) for n in ast.walk(node) if isinstance(n, ast.Name)]
    else:
        return [("write", node, parent)]


def access_is_read(access):
    if isinstance(access, ast.AST):
        # An ast.arg represents a function parameter, not a read.
        if isinstance(access, ast.arg):
            return False
        assert isinstance(access, ast.Name), access
        return isinstance(access.ctx, READ_CONTEXTS)
    else:
        return access[0] == "read"


def access_is_write(access):
    if isinstance(access, ast.AST):
        # An ast.arg represents a function parameter, not a read.
        if isinstance(access, ast.arg):
            return False
        assert isinstance(access, ast.Name), access
        return isinstance(access.ctx, WRITE_CONTEXTS)
    else:
        return access[0] == "write"


def access_name(access):
    if isinstance(access, ast.AST):
        return access.id
    elif isinstance(access, tuple):
        if isinstance(access[1], str):
            return access[1]
        elif isinstance(access[1], ast.Name):
            return access[1].id
    raise ValueError("Unexpected access type.", access)


def access_kind(access):
    if access_is_read(access):
        return "read"
    elif access_is_write(access):
        return "write"


def access_kind_and_name(access):
    return "{}-{}".format(access_kind(access), access_name(access))


def access_identifier(name, kind):
    return "{}-{}".format(kind, name)


class Instruction(object):
    """
    Represents an executable unit of a Python program.

    An Instruction corresponds to a simple statement or assignment in the AST,
    not related to control flow. It encapsulates an AST node (or a string, such as a variable name)
    and computes its accesses.

    In Python 3, constructs such as print and exec are handled as ordinary function calls.
    """

    def __init__(self, node, accesses=None, source=None):
        if not isinstance(node, ast.AST):
            raise TypeError("node must be an instance of ast.AST.", node)
        self.node = node
        if accesses is None:
            accesses = get_accesses_from_ast_node(node)
        self.accesses = accesses
        self.source = source

    def contains_subprogram(self, node):
        """
        Determine whether this Instruction contains the given AST as a subprogram.

        If self.source is not None, an exact match is required; otherwise, a subtree match is used.
        """
        if self.source is not None:
            return represent_same_program(node, self.node)
        for subtree in ast.walk(self.node):
            if represent_same_program(node, subtree):
                return True
        return False

    def get_reads(self):
        return {access for access in self.accesses if access_is_read(access)}

    def get_read_names(self):
        return {access_name(access) for access in self.get_reads()}

    def get_writes(self):
        return {access for access in self.accesses if access_is_write(access)}

    def get_write_names(self):
        return {access_name(access) for access in self.get_writes()}
