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

"""Creates ProgramGraphs from a program or function's AST.

A ProgramGraph represents a Python program or function. The nodes in a
ProgramGraph represent an Instruction (see instruction.py), an AST node, or a
piece of syntax from the program. The edges in a ProgramGraph represent the
relationships between these nodes.
"""

import codecs
import collections
import json
import os

from absl import logging
import astunparse
from astunparse import unparser
import ast
from python_graphs import control_flow
from python_graphs import data_flow
from python_graphs import instruction as instruction_module
from python_graphs import program_graph_dataclasses as pb
from python_graphs import program_utils
from python_graphs import unparser_patch  # pylint: disable=unused-import

# Legacy compatibility removed (e.g., six) – Python 3 only.

NEWLINE_TOKEN = "#NEWLINE#"
UNINDENT_TOKEN = "#UNINDENT#"
INDENT_TOKEN = "#INDENT#"


class ProgramGraph(object):
    """A ProgramGraph represents a Python program or function.

    Attributes:
      root_id: The id of the root ProgramGraphNode.
      nodes: Maps from node id to the ProgramGraphNode with that id.
      edges: A list of the edges (from_node.id, to_node.id, edge type) in the graph.
      child_map: Maps from node id to a list of that node's AST children node ids.
      parent_map: Maps from node id to that node's AST parent node id.
      neighbors_map: Maps from node id to a list of that node's neighboring edges.
      ast_id_to_program_graph_node: Maps from an AST node's object id to the
        corresponding AST program graph node, if it exists.
      root: The root ProgramGraphNode.
    """

    def __init__(self):
        """Constructs an empty ProgramGraph with no root."""
        self.root_id = None
        self.nodes = {}
        self.edges = []
        self.ast_id_to_program_graph_node = {}
        self.child_map = collections.defaultdict(list)
        self.parent_map = collections.defaultdict(lambda: None)
        self.neighbors_map = collections.defaultdict(list)

    # Accessors
    @property
    def root(self):
        if self.root_id not in self.nodes:
            raise ValueError("Graph has no root node.")
        return self.nodes[self.root_id]

    def all_nodes(self):
        return self.nodes.values()

    def get_node(self, obj):
        """Returns the node in the program graph corresponding to an object.

        Arguments:
           obj: Can be an integer, AST node, ProgramGraphNode, or program graph node
             protobuf.

        Raises:
           ValueError: no node exists in the program graph matching obj.
        """
        if isinstance(obj, int) and obj in self.nodes:
            return self.get_node_by_id(obj)
        elif isinstance(obj, ProgramGraphNode):
            return obj
        elif isinstance(obj, pb.Node):
            return self.get_node_by_id(obj.id)
        elif isinstance(obj, (ast.AST, list)):
            return self.get_node_by_ast_node(obj)
        else:
            raise ValueError("Unexpected value for obj.", obj)

    def get_node_by_id(self, obj):
        """Gets a ProgramGraph node for the given integer id."""
        return self.nodes[obj]

    def get_node_by_access(self, access):
        """Gets a ProgramGraph node for the given read or write."""
        if isinstance(access, ast.Name):
            return self.get_node(access)
        else:
            # Expecting a tuple; check the second element.
            if isinstance(access[1], ast.Name):
                return self.get_node(access[1])
            else:
                return self.get_node(access[2])
        raise ValueError("Could not find node for access.", access)

    def get_nodes_by_source(self, source):
        """Generates the nodes in the program graph containing the query source.

        Args:
          source: The query source.

        Returns:
          A generator of all nodes in the program graph with an Instruction with
          source that includes the query source.
        """
        module = ast.parse(source, mode="exec")
        node = module.body[0]
        if isinstance(node, ast.Expr):
            node = node.value

        def matches_source(pg_node):
            if pg_node.has_instruction():
                return pg_node.instruction.contains_subprogram(node)
            else:
                return instruction_module.represent_same_program(pg_node.ast_node, node)

        return filter(matches_source, self.nodes.values())

    def get_node_by_source(self, node):
        return min(self.get_nodes_by_source(node), key=lambda x: len(ast.dump(x.node)))

    def get_nodes_by_function_name(self, name):
        return filter(
            lambda n: n.has_instance_of(ast.FunctionDef) and n.node.name == name,
            self.nodes.values(),
        )

    def get_node_by_function_name(self, name):
        return next(self.get_nodes_by_function_name(name))

    def get_node_by_ast_node(self, ast_node):
        return self.ast_id_to_program_graph_node[id(ast_node)]

    def contains_ast_node(self, ast_node):
        return id(ast_node) in self.ast_id_to_program_graph_node

    def get_ast_nodes_of_type(self, ast_type):
        for node in self.nodes.values():
            if node.node_type == pb.NodeType.AST_NODE and node.ast_type == ast_type:
                yield node

    def get_nodes_by_source_and_identifier(self, source, name):
        for pg_node in self.get_nodes_by_source(source):
            for node in ast.walk(pg_node.node):
                if isinstance(node, ast.Name) and node.id == name:
                    if self.contains_ast_node(node):
                        yield self.get_node_by_ast_node(node)

    def get_node_by_source_and_identifier(self, source, name):
        return next(self.get_nodes_by_source_and_identifier(source, name))

    # Graph Construction Methods
    def add_node(self, node):
        """Adds a ProgramGraphNode to this graph.

        Args:
          node: The ProgramGraphNode that should be added.

        Returns:
          The node that was added.

        Raises:
          ValueError: the node has already been added to this graph.
        """
        assert isinstance(node, ProgramGraphNode), "Not a ProgramGraphNode"
        if node.id in self.nodes:
            raise ValueError("Already contains node", self.nodes[node.id], node.id)
        if node.ast_node is not None:
            if self.contains_ast_node(node.ast_node):
                raise ValueError("Already contains ast node", node.ast_node)
            self.ast_id_to_program_graph_node[id(node.ast_node)] = node
        self.nodes[node.id] = node
        return node

    def add_node_from_instruction(self, instruction):
        node = make_node_from_instruction(instruction)
        return self.add_node(node)

    def add_edge(self, edge):
        """Adds an edge between two nodes in the graph.

        Args:
          edge: The edge, a pb.Edge proto.
        """
        assert isinstance(edge, pb.Edge), "Not a pb.Edge"
        self.edges.append(edge)

        n1 = self.get_node_by_id(edge.id1)
        n2 = self.get_node_by_id(edge.id2)
        if edge.type == pb.EdgeType.FIELD:
            self.child_map[edge.id1].append(edge.id2)
            self.parent_map[n2.id] = edge.id1
        self.neighbors_map[n1.id].append((edge, edge.id2))
        self.neighbors_map[n2.id].append((edge, edge.id1))

    def remove_edge(self, edge):
        """Removes an edge from the graph."""
        self.edges.remove(edge)

        n1 = self.get_node_by_id(edge.id1)
        n2 = self.get_node_by_id(edge.id2)

        if edge.type == pb.EdgeType.FIELD:
            self.child_map[edge.id1].remove(edge.id2)
            del self.parent_map[n2.id]

        self.neighbors_map[n1.id].remove((edge, edge.id2))
        self.neighbors_map[n2.id].remove((edge, edge.id1))

    def add_new_edge(self, n1, n2, edge_type=None, field_name=None):
        n1 = self.get_node(n1)
        n2 = self.get_node(n2)
        new_edge = pb.Edge(id1=n1.id, id2=n2.id, type=edge_type, field_name=field_name)
        self.add_edge(new_edge)
        return new_edge

    # AST Methods
    def to_ast(self, node=None):
        """Convert the program graph to a Python AST."""
        if node is None:
            node = self.root
        return self._build_ast(node=node, update_references=False)

    def reconstruct_ast(self):
        """Reconstruct all internal ProgramGraphNode.ast_node references."""
        self.ast_id_to_program_graph_node.clear()
        self._build_ast(node=self.root, update_references=True)

    def _build_ast(self, node, update_references):
        """Helper method: builds an AST and optionally sets ast_node references."""
        if node.node_type == pb.NodeType.AST_NODE:
            ast_node = getattr(ast, node.ast_type)()
            adjacent_edges = self.neighbors_map[node.id]
            for edge, other_node_id in adjacent_edges:
                if other_node_id == edge.id1:  # incoming edge
                    continue
                if edge.type == pb.EdgeType.FIELD:
                    child_id = other_node_id
                    child = self.get_node_by_id(child_id)
                    try:
                        child_ast = self._build_ast(node=child, update_references=update_references)
                        setattr(ast_node, edge.field_name, child_ast)
                    except Exception as e:
                        error_info = {
                            "original_node_type": type(ast_node).__name__,
                            "field": edge.field_name,
                            "error_message": str(e)
                        }
                        logging.warning(json.dumps({
                            "level": "warning",
                            "message": "Error processing AST field",
                            "details": error_info,
                        }))
                        placeholder = make_placeholder_node(
                            original_node_type=type(ast_node).__name__,
                            error_message=str(e),
                            field_name=edge.field_name,
                        )
                        setattr(ast_node, edge.field_name, placeholder)
            if update_references:
                node.ast_node = ast_node
                self.ast_id_to_program_graph_node[id(ast_node)] = node
            return ast_node
        elif node.node_type == pb.NodeType.AST_LIST:
            list_items = {}
            adjacent_edges = self.neighbors_map[node.id]
            for edge, other_node_id in adjacent_edges:
                if other_node_id == edge.id1:
                    continue
                if edge.type == pb.EdgeType.FIELD:
                    child_id = other_node_id
                    child = self.get_node_by_id(child_id)
                    _, index = parse_list_field_name(edge.field_name)
                    list_items[index] = self._build_ast(node=child, update_references=update_references)
            ast_list = []
            for index in range(len(list_items)):
                ast_list.append(list_items[index])
            return ast_list
        elif node.node_type == pb.NodeType.AST_VALUE:
            return node.ast_value
        else:
            raise ValueError("This ProgramGraphNode does not correspond to a node in an AST.")

    def walk_ast_descendants(self, node=None):
        """Yields the nodes that correspond to the descendants of node in the AST."""
        if node is None:
            node = self.root
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for child_id in reversed(self.child_map[current.id]):
                frontier.append(self.get_node_by_id(child_id))
            yield current

    def parent(self, node):
        """Returns the AST parent of an AST program graph node."""
        parent_id = self.parent_map[node.id]
        if parent_id is None:
            return None
        else:
            return self.get_node_by_id(parent_id)

    def children(self, node):
        """Yields the (direct) AST children of an AST program graph node."""
        for child_id in self.child_map[node.id]:
            yield self.get_node_by_id(child_id)

    def neighbors(self, node, edge_type=None):
        """Returns the incoming and outgoing neighbors of a program graph node."""
        adj_edges = self.neighbors_map[node.id]
        if edge_type is None:
            ids = [tup[1] for tup in adj_edges]
        else:
            ids = [tup[1] for tup in adj_edges if tup[0].type == edge_type]
        return [self.get_node_by_id(id0) for id0 in ids]

    def incoming_neighbors(self, node, edge_type=None):
        """Returns the incoming neighbors of a program graph node."""
        adj_edges = self.neighbors_map[node.id]
        result = []
        for edge, neighbor_id in adj_edges:
            if edge.id2 == node.id:
                if edge_type is None or edge.type == edge_type:
                    result.append(self.get_node_by_id(neighbor_id))
        return result

    def outgoing_neighbors(self, node, edge_type=None):
        """Returns the outgoing neighbors of a program graph node."""
        adj_edges = self.neighbors_map[node.id]
        result = []
        for edge, neighbor_id in adj_edges:
            if edge.id1 == node.id:
                if edge_type is None or edge.type == edge_type:
                    result.append(self.get_node_by_id(neighbor_id))
        return result

    def dump_tree(self, start_node=None):
        """Returns a string representation for debugging."""
        def dump_tree_recurse(node, indent, all_lines):
            indent_str = " " + ("--" * indent)
            node_str = dump_node(node)
            line = " ".join([indent_str, node_str, "\n"])
            all_lines.append(line)
            # Output long distance edges.
            for edge, neighbor_id in self.neighbors_map[node.id]:
                if not is_ast_edge(edge) and not is_syntax_edge(edge) and node.id == edge.id1:
                    type_str = edge.type.name
                    line = " ".join([indent_str, "--((",
                                      type_str, "))-->", str(neighbor_id), "\n"])
                    all_lines.append(line)
            for child in self.children(node):
                dump_tree_recurse(child, indent + 1, all_lines)
            return all_lines

        if start_node is None:
            start_node = self.root
        return "".join(dump_tree_recurse(start_node, 0, []))

    def copy_with_placeholder(self, node):
        """Returns a new program graph in which the subtree of NODE is replaced by a placeholder."""
        descendant_ids = {n.id for n in self.walk_ast_descendants(node)}
        new_graph = ProgramGraph()
        new_graph.add_node(self.root)
        new_graph.root_id = self.root_id
        for edge in self.edges:
            v1 = self.nodes[edge.id1]
            v2 = self.nodes[edge.id2]
            adj_bad_subtree = (edge.id1 in descendant_ids) or (edge.id2 in descendant_ids)
            if adj_bad_subtree:
                if edge.id2 == node.id and is_ast_edge(edge):
                    placeholder = make_placeholder_node(
                        original_node_type="AST",
                        error_message="Subtree removed",
                        field_name="copy_with_placeholder"
                    )
                    placeholder.id = node.id
                    new_graph.add_node(placeholder)
                    new_graph.add_new_edge(v1, placeholder, edge_type=edge.type)
            else:
                if edge.id1 not in new_graph.nodes:
                    new_graph.add_node(v1)
                if edge.id2 not in new_graph.nodes:
                    new_graph.add_node(v2)
                new_graph.add_new_edge(v1, v2, edge_type=edge.type)
        return new_graph

    def copy_subgraph(self, node):
        """Returns a new program graph containing only the subtree rooted at NODE."""
        descendant_ids = {n.id for n in self.walk_ast_descendants(node)}
        new_graph = ProgramGraph()
        new_graph.add_node(node)
        new_graph.root_id = node.id
        for edge in self.edges:
            v1 = self.nodes[edge.id1]
            v2 = self.nodes[edge.id2]
            good_edge = (edge.id1 in descendant_ids) and (edge.id2 in descendant_ids)
            if good_edge:
                if edge.id1 not in new_graph.nodes:
                    new_graph.add_node(v1)
                if edge.id2 not in new_graph.nodes:
                    new_graph.add_node(v2)
                new_graph.add_new_edge(v1, v2, edge_type=edge.type)
        return new_graph


def is_ast_node(node):
    return node.node_type == pb.NodeType.AST_NODE


def is_ast_edge(edge):
    return edge.type == pb.EdgeType.FIELD


def is_syntax_edge(edge):
    return edge.type == pb.EdgeType.SYNTAX


def dump_node(node):
    type_str = "[" + node.node_type.name + "]"
    elements = [type_str, str(node.id), node.ast_type]
    if node.ast_value:
        elements.append(str(node.ast_value))
    if node.syntax:
        elements.append(str(node.syntax))
    return " ".join(elements)


def make_placeholder_node(original_node_type, error_message, field_name):
    node = ProgramGraphNode()
    node.node_type = pb.NodeType.PLACEHOLDER
    node.id = program_utils.unique_id()
    node.has_error = True
    node.error_info = {
        "original_node_type": original_node_type,
        "error_message": error_message,
        "field": field_name,
    }
    return node


def get_program_graph(program):
    """Constructs a program graph to represent the given program."""
    program_node = program_utils.program_to_ast(program)
    program_graph = ProgramGraph()

    # Perform control flow analysis.
    control_flow_graph = control_flow.get_control_flow_graph(program_node)

    for control_flow_node in control_flow_graph.get_control_flow_nodes():
        program_graph.add_node_from_instruction(control_flow_node.instruction)

    for ast_node in ast.walk(program_node):
        if not program_graph.contains_ast_node(ast_node):
            pg_node = make_node_from_ast_node(ast_node)
            program_graph.add_node(pg_node)

    root = program_graph.get_node_by_ast_node(program_node)
    program_graph.root_id = root.id

    # Add AST edges (FIELD), AST_LIST and AST_VALUE nodes.
    for ast_node in ast.walk(program_node):
        for field_name, value in ast.iter_fields(ast_node):
            if isinstance(value, list):
                pg_node = make_node_for_ast_list()
                program_graph.add_node(pg_node)
                program_graph.add_new_edge(ast_node, pg_node, pb.EdgeType.FIELD, field_name)
                for index, item in enumerate(value):
                    list_field_name = make_list_field_name(field_name, index)
                    if isinstance(item, ast.AST):
                        program_graph.add_new_edge(pg_node, item, pb.EdgeType.FIELD, list_field_name)
                    else:
                        item_node = make_node_from_ast_value(item)
                        program_graph.add_node(item_node)
                        program_graph.add_new_edge(pg_node, item_node, pb.EdgeType.FIELD, list_field_name)
            elif isinstance(value, ast.AST):
                program_graph.add_new_edge(ast_node, value, pb.EdgeType.FIELD, field_name)
            else:
                pg_node = make_node_from_ast_value(value)
                program_graph.add_node(pg_node)
                program_graph.add_new_edge(ast_node, pg_node, pb.EdgeType.FIELD, field_name)

    # Add SYNTAX_NODE nodes using the custom AST unparser.
    SyntaxNodeUnparser(program_node, program_graph)

    # Perform data flow analysis.
    analysis = data_flow.LastAccessAnalysis()
    for node in control_flow_graph.get_enter_control_flow_nodes():
        analysis.visit(node)

    # Add control flow edges (CFG_NEXT).
    for control_flow_node in control_flow_graph.get_control_flow_nodes():
        instruction = control_flow_node.instruction
        for next_control_flow_node in control_flow_node.next:
            next_instruction = next_control_flow_node.instruction
            program_graph.add_new_edge(instruction.node, next_instruction.node, edge_type=pb.EdgeType.CFG_NEXT)

    # Add data flow edges (LAST_READ and LAST_WRITE).
    for control_flow_node in control_flow_graph.get_control_flow_nodes():
        last_accesses = control_flow_node.get_label("last_access_in").copy()
        for access in control_flow_node.instruction.accesses:
            pg_node = program_graph.get_node_by_access(access)
            access_name = instruction_module.access_name(access)
            read_identifier = instruction_module.access_identifier(access_name, "read")
            write_identifier = instruction_module.access_identifier(access_name, "write")
            for read in last_accesses.get(read_identifier, []):
                read_pg_node = program_graph.get_node_by_access(read)
                program_graph.add_new_edge(pg_node, read_pg_node, edge_type=pb.EdgeType.LAST_READ)
            for write in last_accesses.get(write_identifier, []):
                write_pg_node = program_graph.get_node_by_access(write)
                program_graph.add_new_edge(pg_node, write_pg_node, edge_type=pb.EdgeType.LAST_WRITE)
            if instruction_module.access_is_read(access):
                last_accesses[read_identifier] = [access]
            elif instruction_module.access_is_write(access):
                last_accesses[write_identifier] = [access]

    # Add COMPUTED_FROM edges.
    for node in ast.walk(program_node):
        if isinstance(node, ast.Assign):
            for value_node in ast.walk(node.value):
                if isinstance(value_node, ast.Name):
                    for target in node.targets:
                        program_graph.add_new_edge(target, value_node, edge_type=pb.EdgeType.COMPUTED_FROM)

    # Add CALLS, FORMAL_ARG_NAME and RETURNS_TO edges.
    for node in ast.walk(program_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_defs = list(program_graph.get_nodes_by_function_name(node.func.id))
                if not func_defs:
                    if node.func.id in dir(__import__("builtins")):
                        message = "Function is builtin."
                    else:
                        message = "Cannot statically determine the function being called."
                    logging.debug("%s (%s)", message, node.func.id)
                for func_def in func_defs:
                    fn_node = func_def.node
                    program_graph.add_new_edge(node, fn_node, edge_type=pb.EdgeType.CALLS)
                    for inner_node in ast.walk(func_def.node):
                        if isinstance(inner_node, ast.Return):
                            program_graph.add_new_edge(inner_node, node, edge_type=pb.EdgeType.RETURNS_TO)
                    for index, arg in enumerate(node.args):
                        formal_arg = None
                        if index < len(fn_node.args.args):
                            formal_arg = fn_node.args.args[index]
                        elif fn_node.args.vararg:
                            formal_arg = fn_node.args
                        if formal_arg is not None:
                            program_graph.add_new_edge(arg, formal_arg, edge_type=pb.EdgeType.FORMAL_ARG_NAME)
                        else:
                            logging.debug("formal_arg is None")
                    for keyword in node.keywords:
                        name = keyword.arg
                        formal_arg = None
                        for arg in fn_node.args.args:
                            if isinstance(arg, ast.Name) and arg.id == name:
                                formal_arg = arg
                                break
                        else:
                            if fn_node.args.kwarg:
                                formal_arg = fn_node.args
                        if formal_arg is not None:
                            program_graph.add_new_edge(keyword.value, formal_arg, edge_type=pb.EdgeType.FORMAL_ARG_NAME)
                        else:
                            logging.debug("formal_arg is None")
            else:
                logging.debug("Cannot statically determine the function being called. (%s)", astunparse.unparse(node.func).strip())

    return program_graph


class SyntaxNodeUnparser(unparser.Unparser):
    """An Unparser class for creating Syntax Token nodes for function graphs."""

    def __init__(self, ast_node, graph):
        self.graph = graph
        self.current_ast_node = None  # The AST node currently being unparsed.
        self.last_syntax_node = None
        self.last_lexical_uses = {}
        self.last_indent = 0
        with codecs.open(os.devnull, "w", encoding="utf-8") as devnull:
            super().__init__(ast_node, file=devnull)

    def dispatch(self, ast_node):
        tmp_ast_node = self.current_ast_node
        self.current_ast_node = ast_node
        super().dispatch(ast_node)
        self.current_ast_node = tmp_ast_node

    def fill(self, text=""):
        try:
            text_with_whitespace = NEWLINE_TOKEN
            if self.last_indent > self._indent:
                text_with_whitespace += UNINDENT_TOKEN * (self.last_indent - self._indent)
            elif self.last_indent < self._indent:
                text_with_whitespace += INDENT_TOKEN * (self._indent - self.last_indent)
            self.last_indent = self._indent
            text_with_whitespace += text
            self._add_syntax_node(text_with_whitespace)
            super().fill(text)
        except Exception as e:
            error_info = {
                "operation": "fill",
                "input": text,
                "error_message": str(e)
            }
            logging.warning(json.dumps({
                "level": "warning",
                "message": "Error in SyntaxNodeUnparser.fill",
                "details": error_info,
            }))
            placeholder = make_placeholder_node("Syntax", str(e), "fill")
            self.graph.add_node(placeholder)

    def write(self, text):
        try:
            if isinstance(text, ast.AST):
                return self.dispatch(text)
            self._add_syntax_node(text)
            super().write(text)
        except Exception as e:
            error_info = {
                "operation": "write",
                "input": str(text),
                "error_message": str(e)
            }
            logging.warning(json.dumps({
                "level": "warning",
                "message": "Error in SyntaxNodeUnparser.write",
                "details": error_info,
            }))
            placeholder = make_placeholder_node("Syntax", str(e), "write")
            self.graph.add_node(placeholder)

    def _add_syntax_node(self, text):
        try:
            text = text.strip()
            if not text:
                return
            syntax_node = make_node_from_syntax(str(text))
            self.graph.add_node(syntax_node)
            self.graph.add_new_edge(self.current_ast_node, syntax_node, edge_type=pb.EdgeType.SYNTAX)
            if self.last_syntax_node:
                self.graph.add_new_edge(self.last_syntax_node, syntax_node, edge_type=pb.EdgeType.NEXT_SYNTAX)
            self.last_syntax_node = syntax_node
        except Exception as e:
            error_info = {
                "operation": "_add_syntax_node",
                "input": text,
                "error_message": str(e)
            }
            logging.warning(json.dumps({
                "level": "warning",
                "message": "Error in _add_syntax_node",
                "details": error_info,
            }))
            placeholder = make_placeholder_node("Syntax", str(e), "_add_syntax_node")
            self.graph.add_node(placeholder)

    def _Name(self, node):
        if node.id in self.last_lexical_uses:
            self.graph.add_new_edge(node, self.last_lexical_uses[node.id], edge_type=pb.EdgeType.LAST_LEXICAL_USE)
        self.last_lexical_uses[node.id] = node
        super()._Name(node)


class ProgramGraphNode(object):
    """A single node in a Program Graph.

    Corresponds to either a SyntaxNode or an Instruction.
    """
    def __init__(self):
        self.node_type = None
        self.id = None
        self.instruction = None
        self.ast_node = None
        self.ast_type = ""
        self.ast_value = ""
        self.syntax = ""
        # New fields for error handling.
        self.has_error = False
        self.error_info = None

    def has_instruction(self):
        return self.instruction is not None

    def has_instance_of(self, t):
        if self.instruction is None:
            return False
        return isinstance(self.instruction.node, t)

    @property
    def node(self):
        if self.ast_node is not None:
            return self.ast_node
        if self.instruction is None:
            return None
        return self.instruction.node

    def __repr__(self):
        return str(self.id) + " " + str(self.ast_type)


def make_node_from_syntax(text):
    node = ProgramGraphNode()
    node.node_type = pb.NodeType.SYNTAX_NODE
    node.id = program_utils.unique_id()
    node.syntax = text
    return node


def make_node_from_instruction(instruction):
    ast_node = instruction.node
    node = make_node_from_ast_node(ast_node)
    node.instruction = instruction
    return node


def make_node_from_ast_node(ast_node):
    node = ProgramGraphNode()
    node.node_type = pb.NodeType.AST_NODE
    node.id = program_utils.unique_id()
    node.ast_node = ast_node
    node.ast_type = type(ast_node).__name__
    return node


def make_node_for_ast_list():
    node = ProgramGraphNode()
    node.node_type = pb.NodeType.AST_LIST
    node.id = program_utils.unique_id()
    return node


def make_node_from_ast_value(value):
    node = ProgramGraphNode()
    node.node_type = pb.NodeType.AST_VALUE
    node.id = program_utils.unique_id()
    node.ast_value = value
    return node


def make_list_field_name(field_name, index):
    return "{}:{}".format(field_name, index)


def parse_list_field_name(list_field_name):
    field_name, index = list_field_name.split(":")
    index = int(index)
    return field_name, index
