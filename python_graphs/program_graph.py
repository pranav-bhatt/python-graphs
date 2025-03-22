# program_graph.py
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
import copy
import ast

from absl import logging
import astunparse
from astunparse import unparser

from python_graphs import control_flow
from python_graphs import data_flow
from python_graphs import instruction as instruction_module
from python_graphs import program_graph_dataclasses as pb
from python_graphs import program_utils
from python_graphs import unparser_patch  # pylint: disable=unused-import

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
      parent_map: Maps from node id to a set of that node's AST parent node ids.
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
        self.parent_map = collections.defaultdict(set)  # now a set for multiple parents
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
        return self.nodes[obj]

    def get_node_by_access(self, access):
        if isinstance(access, (ast.Name, ast.arg)):
            return self.get_node(access)
        else:
            if isinstance(access[1], ast.Name):
                return self.get_node(access[1])
            else:
                return self.get_node(access[2])

    def get_nodes_by_source(self, source):
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

    def get_all_function_nodes_by_name(self, name):
        return filter(
            lambda n: n.has_instance_of(ast.FunctionDef) and n.node.name == name,
            self.nodes.values(),
        )

    def get_node_by_function_name(self, name):
        return next(self.get_all_function_nodes_by_name(name))

    def get_node_by_ast_node(self, ast_node):
        key = id(ast_node)
        if key in self.ast_id_to_program_graph_node:
            return self.ast_id_to_program_graph_node[key]
        raise ValueError("AST node not found in the program graph.")

    def get_node_by_ast_dump(self, ast_node):
        """Search based on structural equality (using ast.dump)."""
        dump_val = ast.dump(ast_node)
        for pg_node in self.ast_id_to_program_graph_node.values():
            if ast.dump(pg_node.ast_node) == dump_val:
                return pg_node
        raise ValueError("AST node (by dump) not found in the program graph.")

    def contains_ast_node(self, ast_node):
        for pg_node in self.ast_id_to_program_graph_node.values():
            if pg_node.ast_node is ast_node:
                return True
        return False

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

    def update_edge(self, existing_edge, new_edge):
        existing_edge.has_back_edge = new_edge.has_back_edge
        return existing_edge

    def add_edge(self, edge):
        assert isinstance(edge, pb.Edge), "Not a pb.Edge"
        for existing_edge in self.edges:
            if (
                existing_edge.id1 == edge.id1
                and existing_edge.id2 == edge.id2
                and existing_edge.type == edge.type
                and existing_edge.field_name == edge.field_name
            ):
                return self.update_edge(existing_edge, edge)
        self.edges.append(edge)
        n1 = self.get_node_by_id(edge.id1)
        n2 = self.get_node_by_id(edge.id2)
        if edge.type == pb.EdgeType.FIELD:
            if edge.id2 not in self.child_map[edge.id1]:
                self.child_map[edge.id1].append(edge.id2)
            self.parent_map[n2.id].add(edge.id1)  # allow multiple parents
        self.neighbors_map[n1.id].append((edge, edge.id2))
        self.neighbors_map[n2.id].append((edge, edge.id1))
        return edge

    def remove_edge(self, edge):
        self.edges.remove(edge)
        n1 = self.get_node_by_id(edge.id1)
        n2 = self.get_node_by_id(edge.id2)
        if edge.type == pb.EdgeType.FIELD:
            if edge.id2 in self.child_map[edge.id1]:
                self.child_map[edge.id1].remove(edge.id2)
            if n2.id in self.parent_map and edge.id1 in self.parent_map[n2.id]:
                self.parent_map[n2.id].remove(edge.id1)
                if not self.parent_map[n2.id]:
                    del self.parent_map[n2.id]
        if (edge, edge.id2) in self.neighbors_map[n1.id]:
            self.neighbors_map[n1.id].remove((edge, edge.id2))
        if (edge, edge.id1) in self.neighbors_map[n2.id]:
            self.neighbors_map[n2.id].remove((edge, edge.id1))

    def add_new_edge(self, n1, n2, edge_type=None, field_name=None):
        n1 = self.get_node(n1)
        n2 = self.get_node(n2)
        new_edge = pb.Edge(id1=n1.id, id2=n2.id, type=edge_type, field_name=field_name)
        updated_edge = self.add_edge(new_edge)
        return updated_edge

    # AST Methods
    def to_ast(self, node=None):
        if node is None:
            node = self.root
        return self._build_ast(node=node, update_references=False, visited={})

    def reconstruct_ast(self):
        logging.info("Starting AST reconstruction...")
        self.ast_id_to_program_graph_node.clear()
        _ = self._build_ast(node=self.root, update_references=True, visited={})
        logging.info("Finished building AST from root (node id=%s)", self.root.id)
        unreachable_count = 0
        for node in self.all_nodes():
            # Do not overwrite if already built.
            if node.ast_node is not None:
                continue
            if node.node_type == pb.NodeType.AST_NODE and node.ast_node is None:
                unreachable_count += 1
                logging.info(
                    "Unreachable node detected: id=%s, ast_type=%s",
                    node.id,
                    node.ast_type,
                )
                if node.instruction is not None:
                    new_ast = copy.deepcopy(node.instruction.node)
                    if isinstance(new_ast, ast.Constant) and new_ast.value is None:
                        new_ast.value = node.ast_value
                    node.ast_node = new_ast
                    self.ast_id_to_program_graph_node[id(new_ast)] = node
                    logging.info(
                        "Updated unreachable node id=%s with new AST node id=%s (from instruction)",
                        node.id,
                        id(new_ast),
                    )
                else:
                    if node.ast_type == "Name":
                        default_id = node.ast_value if node.ast_value else str(node.id)
                        new_ast = ast.Name(id=default_id, ctx=ast.Load())
                        node.ast_node = new_ast
                        dump_val = ast.dump(new_ast)
                        keys_to_remove = [
                            key
                            for key, pg in self.ast_id_to_program_graph_node.items()
                            if ast.dump(pg.ast_node) == dump_val
                        ]
                        for key in keys_to_remove:
                            del self.ast_id_to_program_graph_node[key]
                        self.ast_id_to_program_graph_node[id(new_ast)] = node
                        logging.info(
                            "Updated unreachable Name node id=%s with new AST Name (ast_value=%s)",
                            node.id,
                            default_id,
                        )
                    elif node.ast_type == "arguments":
                        try:
                            new_ast = ast.arguments(
                                posonlyargs=[],
                                args=[],
                                vararg=None,
                                kwonlyargs=[],
                                kw_defaults=[],
                                kwarg=None,
                                defaults=[],
                            )
                        except TypeError:
                            new_ast = ast.arguments(
                                args=[],
                                vararg=None,
                                kwonlyargs=[],
                                kw_defaults=[],
                                kwarg=None,
                                defaults=[],
                            )
                        node.ast_node = new_ast
                        dump_val = ast.dump(new_ast)
                        keys_to_remove = [
                            key
                            for key, pg in self.ast_id_to_program_graph_node.items()
                            if ast.dump(pg.ast_node) == dump_val
                        ]
                        for key in keys_to_remove:
                            del self.ast_id_to_program_graph_node[key]
                        self.ast_id_to_program_graph_node[id(new_ast)] = node
                        logging.info(
                            "Updated unreachable arguments node id=%s with new AST arguments node",
                            node.id,
                        )
                    else:
                        logging.warning(
                            "Unreachable node id=%s (ast_type=%s) has no instruction and no fallback; its ast_node remains None",
                            node.id,
                            node.ast_type,
                        )
        logging.info(
            "AST reconstruction complete. Total unreachable nodes updated: %d",
            unreachable_count,
        )

    def _build_ast(self, node, update_references, visited):
        if node.ast_node is not None:
            visited[node.id] = node.ast_node
            return node.ast_node
        if node.id in visited:
            return visited[node.id]
        if node.node_type == pb.NodeType.AST_NODE:
            if not node.ast_type:
                return node.ast_value
            try:
                if node.ast_type == "Name":
                    ast_node = ast.Name(id="", ctx=ast.Load())
                elif node.ast_type == "Constant":
                    ast_node = ast.Constant(value=None)
                elif node.ast_type == "arguments":
                    try:
                        ast_node = ast.arguments(
                            posonlyargs=[],
                            args=[],
                            vararg=None,
                            kwonlyargs=[],
                            kw_defaults=[],
                            kwarg=None,
                            defaults=[],
                        )
                    except TypeError:
                        ast_node = ast.arguments(
                            args=[],
                            vararg=None,
                            kwonlyargs=[],
                            kw_defaults=[],
                            kwarg=None,
                            defaults=[],
                        )
                else:
                    ast_node = getattr(ast, node.ast_type)()
            except Exception as e:
                logging.error(
                    "Error constructing AST node for type %s: %s", node.ast_type, str(e)
                )
                raise ValueError(
                    f"Error constructing AST node for type {node.ast_type}: {e}"
                )
            visited[node.id] = ast_node
            if node.ast_value:
                if isinstance(ast_node, ast.Name):
                    ast_node.id = node.ast_value
                elif isinstance(ast_node, ast.Constant):
                    ast_node.value = node.ast_value
            for edge, other_node_id in self.neighbors_map[node.id]:
                if other_node_id == edge.id1:
                    continue
                if edge.type == pb.EdgeType.FIELD:
                    child = self.get_node_by_id(other_node_id)
                    try:
                        child_ast = self._build_ast(
                            node=child,
                            update_references=update_references,
                            visited=visited,
                        )
                        setattr(ast_node, edge.field_name, child_ast)
                    except Exception as e:
                        logging.warning(
                            "Error processing AST field %s: %s", edge.field_name, str(e)
                        )
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
            for edge, other_node_id in self.neighbors_map[node.id]:
                if other_node_id == edge.id1:
                    continue
                if edge.type == pb.EdgeType.FIELD:
                    child = self.get_node_by_id(other_node_id)
                    _, index = parse_list_field_name(edge.field_name)
                    list_items[index] = self._build_ast(
                        node=child, update_references=update_references, visited=visited
                    )
            return [list_items[index] for index in range(len(list_items))]
        elif node.node_type == pb.NodeType.AST_VALUE:
            return node.ast_value
        else:
            logging.error(
                "This ProgramGraphNode does not correspond to a node in an AST (node id=%s)",
                node.id,
            )
            raise ValueError(
                "This ProgramGraphNode does not correspond to a node in an AST."
            )

    def walk_ast_descendants(self, node=None):
        if node is None:
            node = self.root
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for child_id in reversed(self.child_map[current.id]):
                frontier.append(self.get_node_by_id(child_id))
            yield current

    def parent(self, node):
        # Now returns the set of parent nodes.
        parent_ids = self.parent_map[node.id]
        return {self.get_node_by_id(pid) for pid in parent_ids} if parent_ids else set()

    def children(self, node):
        return (self.get_node_by_id(cid) for cid in self.child_map[node.id])

    def neighbors(self, node, edge_type=None):
        adj_edges = self.neighbors_map[node.id]
        if edge_type is None:
            ids = [pair[1] for pair in adj_edges]
        else:
            ids = [pair[1] for pair in adj_edges if pair[0].type == edge_type]
        return [self.get_node_by_id(i) for i in ids]

    def incoming_neighbors(self, node, edge_type=None):
        result = []
        for edge, neighbor_id in self.neighbors_map[node.id]:
            if edge.id2 == node.id and (edge_type is None or edge.type == edge_type):
                result.append(self.get_node_by_id(neighbor_id))
        return result

    def outgoing_neighbors(self, node, edge_type=None):
        result = []
        for edge, neighbor_id in self.neighbors_map[node.id]:
            if edge.id1 == node.id and (edge_type is None or edge.type == edge_type):
                result.append(self.get_node_by_id(neighbor_id))
        return result

    def dump_tree(self, start_node=None):
        def dump_tree_recurse(node, indent, all_lines):
            indent_str = " " + ("--" * indent)
            node_str = dump_node(node)
            line = " ".join([indent_str, node_str, "\n"])
            all_lines.append(line)
            for edge, neighbor_id in self.neighbors_map[node.id]:
                if (
                    (not is_ast_edge(edge))
                    and (not is_syntax_edge(edge))
                    and node.id == edge.id1
                ):
                    type_str = edge.type.name
                    line = " ".join(
                        [indent_str, "--((", type_str, "))-->", str(neighbor_id), "\n"]
                    )
                    all_lines.append(line)
            for child in self.children(node):
                dump_tree_recurse(child, indent + 1, all_lines)
            return all_lines

        if start_node is None:
            start_node = self.root
        return "".join(dump_tree_recurse(start_node, 0, []))

    def copy_with_placeholder(self, node):
        descendant_ids = {n.id for n in self.walk_ast_descendants(node)}
        new_graph = ProgramGraph()
        new_graph.add_node(self.root)
        new_graph.root_id = self.root_id
        for edge in self.edges:
            v1 = self.nodes[edge.id1]
            v2 = self.nodes[edge.id2]
            adj_bad_subtree = (edge.id1 in descendant_ids) or (
                edge.id2 in descendant_ids
            )
            if adj_bad_subtree:
                if edge.id2 == node.id and is_ast_edge(edge):
                    placeholder = make_placeholder_node(
                        original_node_type="AST",
                        error_message="Subtree removed",
                        field_name="copy_with_placeholder",
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
    program_node = program_utils.program_to_ast(program)
    program_graph = ProgramGraph()

    control_flow_graph = control_flow.get_control_flow_graph(program_node)
    for control_flow_node in control_flow_graph.get_control_flow_nodes():
        program_graph.add_node_from_instruction(control_flow_node.instruction)
    for ast_node in ast.walk(program_node):
        if not program_graph.contains_ast_node(ast_node):
            pg_node = make_node_from_ast_node(ast_node)
            program_graph.add_node(pg_node)
    root = program_graph.get_node_by_ast_node(program_node)
    program_graph.root_id = root.id

    for ast_node in ast.walk(program_node):
        for field_name, value in ast.iter_fields(ast_node):
            if isinstance(value, list):
                pg_node = make_node_for_ast_list()
                program_graph.add_node(pg_node)
                program_graph.add_new_edge(
                    ast_node, pg_node, pb.EdgeType.FIELD, field_name
                )
                for index, item in enumerate(value):
                    list_field_name = "{}:{}".format(field_name, index)
                    if isinstance(item, ast.AST):
                        program_graph.add_new_edge(
                            pg_node, item, pb.EdgeType.FIELD, list_field_name
                        )
                    else:
                        item_node = make_node_from_ast_value(item)
                        program_graph.add_node(item_node)
                        program_graph.add_new_edge(
                            pg_node, item_node, pb.EdgeType.FIELD, list_field_name
                        )
            elif isinstance(value, ast.AST):
                program_graph.add_new_edge(
                    ast_node, value, pb.EdgeType.FIELD, field_name
                )
            else:
                pg_node = make_node_from_ast_value(value)
                program_graph.add_node(pg_node)
                program_graph.add_new_edge(
                    ast_node, pg_node, pb.EdgeType.FIELD, field_name
                )

    SyntaxNodeUnparser(program_node, program_graph)

    analysis = data_flow.LastAccessAnalysis()
    for node in control_flow_graph.get_enter_control_flow_nodes():
        analysis.visit(node)

    for control_flow_node in control_flow_graph.get_control_flow_nodes():
        instruction = control_flow_node.instruction
        for next_control_flow_node in control_flow_node.next:
            next_instruction = next_control_flow_node.instruction
            program_graph.add_new_edge(
                instruction.node, next_instruction.node, edge_type=pb.EdgeType.CFG_NEXT
            )

    for control_flow_node in control_flow_graph.get_control_flow_nodes():
        last_accesses = {
            k: list(v) for k, v in control_flow_node.get_label("last_access_in").items()
        }
        for access in control_flow_node.instruction.accesses:
            pg_node = program_graph.get_node_by_access(access)
            access_name = instruction_module.access_name(access)
            read_identifier = instruction_module.access_identifier(access_name, "read")
            write_identifier = instruction_module.access_identifier(
                access_name, "write"
            )
            if read_identifier in last_accesses and last_accesses[read_identifier]:
                last_read = last_accesses[read_identifier][0]
                program_graph.add_new_edge(
                    pg_node,
                    program_graph.get_node_by_access(last_read),
                    edge_type=pb.EdgeType.LAST_READ,
                )
            if write_identifier in last_accesses and last_accesses[write_identifier]:
                last_write = last_accesses[write_identifier][0]
                program_graph.add_new_edge(
                    pg_node,
                    program_graph.get_node_by_access(last_write),
                    edge_type=pb.EdgeType.LAST_WRITE,
                )
            if instruction_module.access_is_read(access):
                if read_identifier in last_accesses and last_accesses[read_identifier]:
                    last_read = last_accesses[read_identifier][0]
                    program_graph.add_new_edge(
                        pg_node,
                        program_graph.get_node_by_access(last_read),
                        edge_type=pb.EdgeType.LAST_READ,
                    )
                last_accesses[read_identifier] = [access]
            elif instruction_module.access_is_write(access):
                if read_identifier in last_accesses and last_accesses[read_identifier]:
                    last_read = last_accesses[read_identifier][0]
                    program_graph.add_new_edge(
                        pg_node,
                        program_graph.get_node_by_access(last_read),
                        edge_type=pb.EdgeType.LAST_READ,
                    )
                last_accesses[write_identifier] = [access]

    for node in ast.walk(program_node):
        if isinstance(node, ast.Assign):
            for value_node in ast.walk(node.value):
                if isinstance(value_node, ast.Name):
                    for target in node.targets:
                        program_graph.add_new_edge(
                            target, value_node, edge_type=pb.EdgeType.COMPUTED_FROM
                        )

    for node in ast.walk(program_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_defs = list(
                    program_graph.get_all_function_nodes_by_name(node.func.id)
                )
                if not func_defs:
                    if node.func.id in dir(__import__("builtins")):
                        message = "Function is builtin."
                    else:
                        message = (
                            "Cannot statically determine the function being called."
                        )
                    logging.debug("%s (%s)", message, node.func.id)
                for func_def in func_defs:
                    fn_node = func_def.node
                    program_graph.add_new_edge(
                        node, fn_node, edge_type=pb.EdgeType.CALLS
                    )
                    for inner_node in ast.walk(func_def.node):
                        if isinstance(inner_node, ast.Return):
                            program_graph.add_new_edge(
                                inner_node, node, edge_type=pb.EdgeType.RETURNS_TO
                            )
                    for index, arg in enumerate(node.args):
                        formal_arg = None
                        if index < len(fn_node.args.args):
                            formal_arg = fn_node.args.args[index]
                        elif fn_node.args.vararg:
                            formal_arg = fn_node.args
                        if formal_arg is not None:
                            program_graph.add_new_edge(
                                arg, formal_arg, edge_type=pb.EdgeType.FORMAL_ARG_NAME
                            )
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
                            program_graph.add_new_edge(
                                keyword.value,
                                formal_arg,
                                edge_type=pb.EdgeType.FORMAL_ARG_NAME,
                            )
                        else:
                            logging.debug("formal_arg is None")
            else:
                logging.debug(
                    "Cannot statically determine the function being called. (%s)",
                    astunparse.unparse(node.func).strip(),
                )
    return program_graph


class SyntaxNodeUnparser(unparser.Unparser):
    """An Unparser class for creating Syntax Token nodes for function graphs."""

    def __init__(self, ast_node, graph):
        self.graph = graph
        self.current_ast_node = None
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
                text_with_whitespace += UNINDENT_TOKEN * (
                    self.last_indent - self._indent
                )
            elif self.last_indent < self._indent:
                text_with_whitespace += INDENT_TOKEN * (self._indent - self.last_indent)
            self.last_indent = self._indent
            text_with_whitespace += text
            self._add_syntax_node(text_with_whitespace)
            super().fill(text)
        except Exception as e:
            logging.warning(
                json.dumps(
                    {
                        "level": "warning",
                        "message": "Error in SyntaxNodeUnparser.fill",
                        "details": {
                            "operation": "fill",
                            "input": text,
                            "error_message": str(e),
                        },
                    }
                )
            )
            placeholder = make_placeholder_node("Syntax", str(e), "fill")
            self.graph.add_node(placeholder)

    def write(self, text):
        try:
            if isinstance(text, ast.AST):
                return self.dispatch(text)
            self._add_syntax_node(text)
            super().write(text)
        except Exception as e:
            logging.warning(
                json.dumps(
                    {
                        "level": "warning",
                        "message": "Error in SyntaxNodeUnparser.write",
                        "details": {
                            "operation": "write",
                            "input": str(text),
                            "error_message": str(e),
                        },
                    }
                )
            )
            placeholder = make_placeholder_node("Syntax", str(e), "write")
            self.graph.add_node(placeholder)

    def _add_syntax_node(self, text):
        try:
            text = text.strip()
            if not text:
                return
            syntax_node = make_node_from_syntax(str(text))
            self.graph.add_node(syntax_node)
            self.graph.add_new_edge(
                self.current_ast_node, syntax_node, edge_type=pb.EdgeType.SYNTAX
            )
            if self.last_syntax_node:
                self.graph.add_new_edge(
                    self.last_syntax_node,
                    syntax_node,
                    edge_type=pb.EdgeType.NEXT_SYNTAX,
                )
            self.last_syntax_node = syntax_node
        except Exception as e:
            logging.warning(
                json.dumps(
                    {
                        "level": "warning",
                        "message": "Error in _add_syntax_node",
                        "details": {
                            "operation": "_add_syntax_node",
                            "input": text,
                            "error_message": str(e),
                        },
                    }
                )
            )
            placeholder = make_placeholder_node("Syntax", str(e), "_add_syntax_node")
            self.graph.add_node(placeholder)

    def _Name(self, node):
        if node.id in self.last_lexical_uses:
            self.graph.add_new_edge(
                node,
                self.last_lexical_uses[node.id],
                edge_type=pb.EdgeType.LAST_LEXICAL_USE,
            )
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
        if self.instruction is not None:
            return self.instruction.node
        return None

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
    return field_name, int(index)
