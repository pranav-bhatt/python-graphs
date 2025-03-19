# test_program_graph.py

import collections
import inspect
import time
import ast
import json
import pytest

from python_graphs import control_flow_test_components as cftc
from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb
from python_graphs import program_graph_test_components as pgtc
from python_graphs import program_utils


def get_test_components():
    """Generates functions from two sets of test components."""
    for name, fn in inspect.getmembers(pgtc, predicate=inspect.isfunction):
        yield fn
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        yield fn


# Helper assertion functions.
def assert_edge(graph, n1, n2, edge_type):
    edge = pb.Edge(id1=n1.id, id2=n2.id, type=edge_type)
    assert edge in graph.edges, f"Expected edge {edge} not found in graph.edges."


def assert_no_edge(graph, n1, n2, edge_type):
    edge = pb.Edge(id1=n1.id, id2=n2.id, type=edge_type)
    assert edge not in graph.edges, f"Unexpected edge {edge} found in graph.edges."


def analyze_get_program_graph(program_generator, start=0):
    num_edges = 0
    num_edges_by_type = collections.defaultdict(int)
    num_nodes = 0
    num_graphs = 0
    times = {}
    for index, program in enumerate(program_generator):
        if index < start:
            continue
        start_time = time.time()
        graph = program_graph.get_program_graph(program)
        end_time = time.time()
        times[index] = end_time - start_time
        num_edges += len(graph.edges)
        for edge in graph.edges:
            num_edges_by_type[edge.type] += 1
        num_nodes += len(graph.nodes)
        num_graphs += 1
    print(f"Total edges: {num_edges}, nodes: {num_nodes}, graphs: {num_graphs}")
    if num_graphs:
        print(
            f"Avg edges per graph: {num_edges/num_graphs}, Avg nodes per graph: {num_nodes/num_graphs}"
        )
    for edge_type, count in num_edges_by_type.items():
        if num_graphs:
            print(f"Edge type {edge_type}: Avg {count/num_graphs}")
    print("Timing info:", times)
    sorted_times = sorted(times.items(), key=lambda kv: -kv[1])
    print("Top timings:", sorted_times[:10])


def test_get_program_graph_test_components():
    analyze_get_program_graph(get_test_components(), start=0)


def test_last_lexical_use_edges_function_call():
    graph = program_graph.get_program_graph(pgtc.function_call)
    read = graph.get_node_by_source_and_identifier("return z", "z")
    write = graph.get_node_by_source_and_identifier(
        "z = function_call_helper(x, y)", "z"
    )
    assert_edge(graph, read, write, pb.EdgeType.LAST_LEXICAL_USE)


def test_last_write_edges_function_call():
    graph = program_graph.get_program_graph(pgtc.function_call)
    write_z = graph.get_node_by_source_and_identifier(
        "z = function_call_helper(x, y)", "z"
    )
    read_z = graph.get_node_by_source_and_identifier("return z", "z")
    assert_edge(graph, read_z, write_z, pb.EdgeType.LAST_WRITE)
    write_y = graph.get_node_by_source_and_identifier("y = 2", "y")
    read_y = graph.get_node_by_source_and_identifier(
        "z = function_call_helper(x, y)", "y"
    )
    assert_edge(graph, read_y, write_y, pb.EdgeType.LAST_WRITE)


def test_last_read_edges_assignments():
    graph = program_graph.get_program_graph(pgtc.assignments)
    write_a0 = graph.get_node_by_source_and_identifier("a, b = 0, 0", "a")
    read_a0 = graph.get_node_by_source_and_identifier("c = 2 * a + 1", "a")
    write_a1 = graph.get_node_by_source_and_identifier("a = c + 3", "a")
    assert_edge(graph, write_a1, read_a0, pb.EdgeType.LAST_READ)
    assert_no_edge(graph, write_a0, read_a0, pb.EdgeType.LAST_READ)
    read_a1 = graph.get_node_by_source_and_identifier("return a, b, c, d", "a")
    assert_edge(graph, read_a1, read_a0, pb.EdgeType.LAST_READ)


def test_last_read_last_write_edges_repeated_identifier():
    graph = program_graph.get_program_graph(pgtc.repeated_identifier)
    write_x0 = graph.get_node_by_source_and_identifier("x = 0", "x")
    stmt1 = graph.get_node_by_source("x = x + 1").ast_node
    read_x0 = graph.get_node_by_ast_node(stmt1.value.left)
    write_x1 = graph.get_node_by_ast_node(stmt1.targets[0])
    stmt2 = graph.get_node_by_source("x = (x + (x + x)) + x").ast_node
    read_x1 = graph.get_node_by_ast_node(stmt2.value.left.left)
    read_x2 = graph.get_node_by_ast_node(stmt2.value.left.right.left)
    read_x3 = graph.get_node_by_ast_node(stmt2.value.left.right.right)
    read_x4 = graph.get_node_by_ast_node(stmt2.value.right)
    write_x2 = graph.get_node_by_ast_node(stmt2.targets[0])
    read_x5 = graph.get_node_by_source_and_identifier("return x", "x")

    assert_edge(graph, write_x1, read_x0, pb.EdgeType.LAST_READ)
    assert_edge(graph, read_x1, read_x0, pb.EdgeType.LAST_READ)
    assert_edge(graph, read_x2, read_x1, pb.EdgeType.LAST_READ)
    assert_edge(graph, read_x3, read_x2, pb.EdgeType.LAST_READ)
    assert_edge(graph, read_x4, read_x3, pb.EdgeType.LAST_READ)
    assert_edge(graph, write_x2, read_x4, pb.EdgeType.LAST_READ)
    assert_edge(graph, read_x5, read_x4, pb.EdgeType.LAST_READ)

    assert_edge(graph, read_x0, write_x0, pb.EdgeType.LAST_WRITE)
    assert_edge(graph, write_x1, write_x0, pb.EdgeType.LAST_WRITE)
    assert_edge(graph, read_x2, write_x1, pb.EdgeType.LAST_WRITE)
    assert_edge(graph, read_x3, write_x1, pb.EdgeType.LAST_WRITE)
    assert_edge(graph, read_x4, write_x1, pb.EdgeType.LAST_WRITE)
    assert_edge(graph, write_x2, write_x1, pb.EdgeType.LAST_WRITE)
    assert_edge(graph, read_x5, write_x2, pb.EdgeType.LAST_WRITE)


def test_computed_from_edges():
    graph = program_graph.get_program_graph(pgtc.assignments)
    target_c = graph.get_node_by_source_and_identifier("c = 2 * a + 1", "c")
    from_a = graph.get_node_by_source_and_identifier("c = 2 * a + 1", "a")
    assert_edge(graph, target_c, from_a, pb.EdgeType.COMPUTED_FROM)
    target_d = graph.get_node_by_source_and_identifier("d = b - c + 2", "d")
    from_b = graph.get_node_by_source_and_identifier("d = b - c + 2", "b")
    from_c = graph.get_node_by_source_and_identifier("d = b - c + 2", "c")
    assert_edge(graph, target_d, from_b, pb.EdgeType.COMPUTED_FROM)
    assert_edge(graph, target_d, from_c, pb.EdgeType.COMPUTED_FROM)


def test_calls_edges():
    graph = program_graph.get_program_graph(pgtc)
    call = graph.get_node_by_source("function_call_helper(x, y)")
    assert isinstance(call.node, ast.Call)
    function_call_helper_def = graph.get_node_by_function_name("function_call_helper")
    assignments_def = graph.get_node_by_function_name("assignments")
    assert_edge(graph, call, function_call_helper_def, pb.EdgeType.CALLS)
    assert_no_edge(graph, call, assignments_def, pb.EdgeType.CALLS)


def test_formal_arg_name_edges():
    graph = program_graph.get_program_graph(pgtc)
    x = graph.get_node_by_source_and_identifier("function_call_helper(x, y)", "x")
    y = graph.get_node_by_source_and_identifier("function_call_helper(x, y)", "y")
    function_call_helper_def = graph.get_node_by_function_name("function_call_helper")
    arg0_ast_node = function_call_helper_def.node.args.args[0]
    arg0 = graph.get_node_by_ast_node(arg0_ast_node)
    arg1_ast_node = function_call_helper_def.node.args.args[1]
    arg1 = graph.get_node_by_ast_node(arg1_ast_node)
    assert_edge(graph, x, arg0, pb.EdgeType.FORMAL_ARG_NAME)
    assert_edge(graph, y, arg1, pb.EdgeType.FORMAL_ARG_NAME)
    assert_no_edge(graph, x, arg1, pb.EdgeType.FORMAL_ARG_NAME)
    assert_no_edge(graph, y, arg0, pb.EdgeType.FORMAL_ARG_NAME)


def test_returns_to_edges():
    graph = program_graph.get_program_graph(pgtc)
    call = graph.get_node_by_source("function_call_helper(x, y)")
    return_stmt = graph.get_node_by_source("return arg0 + arg1")
    assert_edge(graph, return_stmt, call, pb.EdgeType.RETURNS_TO)


def test_syntax_information():
    # Placeholder for future tests on syntax representation.
    pass


def test_ast_acyclic():
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        ast_nodes = set()
        worklist = [graph.root]
        while worklist:
            current = worklist.pop()
            assert (
                current not in ast_nodes
            ), f"ProgramGraph AST cyclic in function {name}\nAST: {graph.dump_tree()}"
            ast_nodes.add(current)
            worklist.extend(list(graph.children(current)))


def test_neighbors_children_consistent():
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        for node in graph.all_nodes():
            if node.node_type == pb.NodeType.AST_NODE:
                children0 = set(graph.outgoing_neighbors(node, pb.EdgeType.FIELD))
                children1 = set(graph.children(node))
                assert children0 == children1, f"Mismatch in children for node {node}"


def test_walk_ast_descendants():
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        all_nodes = set(graph.all_nodes())
        for node in graph.walk_ast_descendants():
            assert (
                node in all_nodes
            ), f"Node {node} from walk_ast_descendants not in graph.all_nodes()"


def test_roundtrip_ast():
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        ast_representation = program_utils.program_to_ast(fn)
        graph = program_graph.get_program_graph(fn)
        ast_reproduction = graph.to_ast()
        assert ast.dump(ast_representation) == ast.dump(
            ast_reproduction
        ), f"AST roundtrip failed for function {name}"


def test_reconstruct_missing_ast():
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        ast_original = graph.root.ast_node
        for node in graph.all_nodes():
            node.ast_node = None
        graph.reconstruct_ast()
        ast_reproduction = graph.root.ast_node
        assert ast.dump(ast_original) == ast.dump(
            ast_reproduction
        ), f"AST reconstruction failed for function {name}"
        for node in graph.all_nodes():
            if node.node_type == pb.NodeType.AST_NODE:
                assert isinstance(
                    node.ast_node, ast.AST
                ), f"Node {node} ast_node not an instance of ast.AST"
                assert (
                    graph.get_node_by_ast_node(node.ast_node) == node
                ), f"Mismatch for node {node}"
        assert not graph.contains_ast_node(
            ast_original
        ), "Old AST nodes still referenced in graph"


def test_remove():
    graph = program_graph.get_program_graph(pgtc.assignments)
    for edge in list(graph.edges)[:]:
        graph.remove_edge(edge)
        assert edge not in graph.edges, "Edge not removed from graph.edges"
        assert (edge, edge.id2) not in graph.neighbors_map[
            edge.id1
        ], "Edge still in neighbors_map for node"
        assert (edge, edge.id1) not in graph.neighbors_map[
            edge.id2
        ], "Edge still in neighbors_map for node"
        if edge.type == pb.EdgeType.FIELD:
            assert (
                edge.id2 not in graph.child_map[edge.id1]
            ), "Child edge still in child_map"
            assert edge.id2 not in graph.parent_map, "Parent edge still in parent_map"
        graph.add_edge(edge)
        assert edge in graph.edges, "Edge not re-added to graph.edges"
        assert (edge, edge.id2) in graph.neighbors_map[
            edge.id1
        ], "Edge not in neighbors_map after re-add"
        assert (edge, edge.id1) in graph.neighbors_map[
            edge.id2
        ], "Edge not in neighbors_map after re-add"
        if edge.type == pb.EdgeType.FIELD:
            assert (
                edge.id2 in graph.child_map[edge.id1]
            ), "Child edge not in child_map after re-add"
            assert (
                edge.id2 in graph.parent_map
            ), "Parent edge not in parent_map after re-add"
