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

# --- Helper Functions for Testing ---


def create_ast_node(ast_type, ast_value=""):
    """Creates a ProgramGraphNode of type AST_NODE with a given ast_type.
    Note: We explicitly set ast_node to a default instance so that identity lookups succeed.
    For a Name node, we create an ast.Name with the given value.
    """
    node = program_graph.ProgramGraphNode()
    node.node_type = pb.NodeType.AST_NODE
    node.id = program_utils.unique_id()
    node.ast_type = ast_type
    node.ast_value = ast_value
    # Pre-set ast_node so that identity lookup works (if desired).
    if ast_type == "Name":
        node.ast_node = ast.Name(
            id=ast_value if ast_value else str(node.id), ctx=ast.Load()
        )
    elif ast_type == "Constant":
        node.ast_node = ast.Constant(value=ast_value if ast_value else None)
    elif ast_type == "arguments":
        try:
            node.ast_node = ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            )
        except TypeError:
            node.ast_node = ast.arguments(
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            )
    else:
        # Leave ast_node as None; it will be filled in by reconstruct_ast.
        node.ast_node = None
    return node


def create_ast_list_node():
    """Creates a ProgramGraphNode for an AST_LIST."""
    node = program_graph.ProgramGraphNode()
    node.node_type = pb.NodeType.AST_LIST
    node.id = program_utils.unique_id()
    return node


def create_ast_value_node(value):
    """Creates a ProgramGraphNode for an AST_VALUE."""
    node = program_graph.ProgramGraphNode()
    node.node_type = pb.NodeType.AST_VALUE
    node.id = program_utils.unique_id()
    node.ast_value = value
    return node


# Dummy instruction for testing unreachable nodes.
class DummyInstruction:
    def __init__(self, node):
        self.node = node


def create_instruction_node(ast_node):
    instr = DummyInstruction(ast_node)
    node = program_graph.make_node_from_instruction(instr)
    return node


def add_field_edge(graph, parent, child, field_name):
    # A helper that manually creates a FIELD edge and updates the graph.
    edge = pb.Edge(
        id1=parent.id, id2=child.id, type=pb.EdgeType.FIELD, field_name=field_name
    )
    graph.edges.append(edge)
    graph.neighbors_map[parent.id].append((edge, child.id))
    graph.neighbors_map[child.id].append((edge, parent.id))
    if child.id not in graph.child_map[parent.id]:
        graph.child_map[parent.id].append(child.id)
    graph.parent_map[child.id].add(parent.id)


# --- Tests for reconstruct_ast ---


def test_reconstruct_when_all_nodes_have_ast_node():
    """If nodes already have an ast_node, reconstruct_ast should not change them."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("Constant", ast_value=42)
    # Pre-set the ast_node.
    const = ast.Constant(value=42)
    node.ast_node = const
    graph.add_node(node)
    graph.root_id = node.id

    graph.reconstruct_ast()

    # The already-set ast_node should remain the same.
    assert node.ast_node is const, "Existing ast_node should not be overwritten"


def test_reconstruct_unreachable_node_with_instruction():
    """A node with an associated instruction should get its ast_node set to a deepcopy
    of the instruction’s node."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("Constant")
    instr_node = ast.Constant(value=99)
    node.instruction = DummyInstruction(instr_node)
    # Ensure ast_node is None initially.
    node.ast_node = None
    graph.add_node(node)
    graph.root_id = node.id

    graph.reconstruct_ast()

    assert node.ast_node is not None, "Node with instruction should be updated"
    assert isinstance(node.ast_node, ast.Constant)
    assert node.ast_node.value == 99


def test_reconstruct_unreachable_name_node_no_instruction():
    """A node of type 'Name' with no instruction should be updated using a default AST Name."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("Name", ast_value="foo")
    # Remove any preset so that reconstruct_ast takes fallback.
    node.ast_node = None
    graph.add_node(node)
    graph.root_id = node.id

    graph.reconstruct_ast()

    assert node.ast_node is not None, "Unreachable Name node should be updated"
    assert isinstance(node.ast_node, ast.Name)
    assert node.ast_node.id == "foo"


def test_reconstruct_unreachable_arguments_node_no_instruction():
    """A node of type 'arguments' with no instruction should be updated with a default
    AST arguments node (all fields empty)."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("arguments")
    node.ast_node = None
    graph.add_node(node)
    graph.root_id = node.id

    graph.reconstruct_ast()

    assert node.ast_node is not None, "Unreachable arguments node should be updated"
    assert isinstance(node.ast_node, ast.arguments)
    args_node = node.ast_node
    # Depending on Python version, posonlyargs may not be defined.
    if hasattr(args_node, "posonlyargs"):
        assert args_node.posonlyargs == []
    assert args_node.args == []
    assert args_node.vararg is None
    assert args_node.kwonlyargs == []
    assert args_node.kw_defaults == []
    assert args_node.kwarg is None
    assert args_node.defaults == []


def test_reconstruct_unreachable_node_no_fallback():
    """A node with an unsupported type (i.e. no instruction and not 'Name' or 'arguments')
    should remain with ast_node == None."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("BinOp")
    node.ast_node = None
    graph.add_node(node)
    graph.root_id = node.id

    graph.reconstruct_ast()

    assert node.ast_node is None, "Unreachable node with no fallback should remain None"


def test_ast_id_mapping_updated():
    """After reconstruction, each node that has an ast_node should be recorded in
    ast_id_to_program_graph_node by identity."""
    graph = program_graph.ProgramGraph()
    node1 = create_ast_node("Name", ast_value="x")
    node2 = create_ast_node("Constant", ast_value=123)
    # Pre-set ast_node for node1 (via create_ast_node) and leave node2's fallback to be generated.
    graph.add_node(node1)
    graph.add_node(node2)
    graph.root_id = node1.id

    graph.reconstruct_ast()

    mapping = graph.ast_id_to_program_graph_node
    assert id(node1.ast_node) in mapping
    assert mapping[id(node1.ast_node)] is node1
    if node2.ast_node is not None:
        assert id(node2.ast_node) in mapping
    else:
        with pytest.raises(ValueError):
            graph.get_node_by_ast_node(ast.Constant(value=1))  # dummy lookup


def test_multiple_parents():
    """Ensure that a single child node can have multiple parents.
    The test creates two parent nodes that both point to the same child node and verifies
    that after reconstruction, both parents refer to the same AST child."""
    graph = program_graph.ProgramGraph()
    child = create_ast_node("Name", ast_value="y")
    # Set child.ast_node to None so that reconstruct_ast uses fallback.
    child.ast_node = None
    parent1 = create_ast_node("Expr")
    parent1.ast_node = None
    parent2 = create_ast_node("Expr")
    parent2.ast_node = None
    graph.add_node(child)
    graph.add_node(parent1)
    graph.add_node(parent2)
    graph.root_id = parent1.id
    # Add edges: parent1 -> child and parent2 -> child.
    graph.add_new_edge(parent1, child, edge_type=pb.EdgeType.FIELD, field_name="child")
    graph.add_new_edge(parent2, child, edge_type=pb.EdgeType.FIELD, field_name="child")

    graph.reconstruct_ast()

    # Verify that both parent's AST nodes have an attribute "child" equal to child's AST node.
    for parent in [parent1, parent2]:
        assert hasattr(
            parent.ast_node, "child"
        ), "Parent AST node should have attribute 'child'"
        assert (
            parent.ast_node.child is child.ast_node
        ), "Child AST node must be identical for both parents"


def test_structural_lookup_functionality():
    """Verify that get_node_by_ast_node uses identity only by default and that a structural lookup
    (using get_node_by_ast_dump) returns a match for structurally equal AST nodes."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("Name", ast_value="z")
    # Ensure that ast_node is preset (from create_ast_node).
    graph.add_node(node)
    graph.root_id = node.id

    # Since ast_node was preset, identity lookup works.
    result = graph.get_node_by_ast_node(node.ast_node)
    assert result is node, "Identity lookup did not return the original node"

    # Structural lookup: use a different but structurally equal AST node.
    identical = ast.Name(id="z", ctx=ast.Load())
    result_dump = graph.get_node_by_ast_dump(identical)
    assert result_dump is node, "Structural lookup did not return the original node"


# --- Tests for _build_ast ---


def test_build_ast_constant():
    """Test that a simple AST_NODE with type 'Constant' is built correctly."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("Constant", ast_value=42)
    # Remove preset to force _build_ast to create a new Constant.
    node.ast_node = None
    graph.add_node(node)
    graph.root_id = node.id

    ast_obj = graph._build_ast(node, update_references=True, visited={})
    assert isinstance(ast_obj, ast.Constant)
    # Since ast_value was 42, the built AST should have value 42.
    assert ast_obj.value == 42
    assert node.ast_node is ast_obj
    assert graph.ast_id_to_program_graph_node.get(id(ast_obj)) is node


def test_build_ast_cycle_prevention():
    """Test that a cycle in the graph is prevented using the visited dict."""
    graph = program_graph.ProgramGraph()
    node1 = create_ast_node("Name", ast_value="x")
    node1.ast_node = None
    node2 = create_ast_node("Load")
    node2.ast_node = None
    graph.add_node(node1)
    graph.add_node(node2)
    graph.root_id = node1.id

    # Create a cycle: node1 -> node2 and node2 -> node1.
    add_field_edge(graph, node1, node2, "ctx")
    add_field_edge(graph, node2, node1, "dummy")
    visited = {}
    ast_obj1 = graph._build_ast(node1, update_references=True, visited=visited)
    assert node1.id in visited
    assert node2.id in visited
    assert ast_obj1 is visited[node1.id]


def test_build_ast_field_edge():
    """Test that a parent node with a FIELD edge gets its attribute set to the built child."""
    graph = program_graph.ProgramGraph()
    parent = create_ast_node("Expr")
    parent.ast_node = None
    child = create_ast_node("Name", ast_value="y")
    child.ast_node = None
    graph.add_node(parent)
    graph.add_node(child)
    graph.root_id = parent.id
    add_field_edge(graph, parent, child, "value")
    visited = {}
    ast_parent = graph._build_ast(parent, update_references=True, visited=visited)
    assert hasattr(
        ast_parent, "value"
    ), "Parent AST node should have a 'value' attribute"
    child_ast = graph._build_ast(child, update_references=True, visited=visited)
    assert ast_parent.value is child_ast


def test_build_ast_list():
    """Test that an AST_LIST node returns a list of child AST nodes in the correct order."""
    graph = program_graph.ProgramGraph()
    list_node = create_ast_list_node()
    child0 = create_ast_node("Name", ast_value="a")
    child1 = create_ast_node("Name", ast_value="b")
    list_node.ast_node = None
    child0.ast_node = None
    child1.ast_node = None
    graph.add_node(list_node)
    graph.add_node(child0)
    graph.add_node(child1)
    graph.root_id = list_node.id
    add_field_edge(graph, list_node, child0, "list:0")
    add_field_edge(graph, list_node, child1, "list:1")
    ast_list = graph._build_ast(list_node, update_references=True, visited={})
    assert isinstance(ast_list, list)
    assert len(ast_list) == 2
    a0 = ast_list[0]
    a1 = ast_list[1]
    assert isinstance(a0, ast.Name)
    assert a0.id == child0.ast_value
    assert isinstance(a1, ast.Name)
    assert a1.id == child1.ast_value


def test_build_ast_value():
    """Test that an AST_VALUE node returns its ast_value."""
    graph = program_graph.ProgramGraph()
    node = create_ast_value_node("hello")
    graph.add_node(node)
    graph.root_id = node.id
    result = graph._build_ast(node, update_references=True, visited={})
    assert result == "hello"


def test_build_ast_invalid_type():
    """Test that if a node’s ast_type is invalid, an error is raised."""
    graph = program_graph.ProgramGraph()
    node = create_ast_node("NonExistentType")
    graph.add_node(node)
    graph.root_id = node.id
    with pytest.raises(ValueError):
        graph._build_ast(node, update_references=True, visited={})


def test_build_ast_field_error():
    """Simulate an error in processing a FIELD edge: if the child raises an error,
    the parent’s field should be set to a placeholder."""
    graph = program_graph.ProgramGraph()
    parent = create_ast_node("Expr")
    parent.ast_node = None
    bad_child = create_ast_node("NonExistentType")
    bad_child.ast_node = None
    graph.add_node(parent)
    graph.add_node(bad_child)
    graph.root_id = parent.id
    add_field_edge(graph, parent, bad_child, "value")
    ast_parent = graph._build_ast(parent, update_references=True, visited={})
    placeholder = getattr(ast_parent, "value")
    assert not isinstance(
        placeholder, ast.AST
    ), "Field error should yield a placeholder (not an ast.AST)"


# --- Other Tests (for neighbors, tree walk, roundtrip, remove/re-add) ---


def test_neighbors_children_consistent():
    """Test that for every AST_NODE, the set of outgoing FIELD neighbors is the same as children()."""
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        for node in graph.all_nodes():
            if node.node_type == pb.NodeType.AST_NODE:
                children0 = set(graph.outgoing_neighbors(node, pb.EdgeType.FIELD))
                children1 = set(graph.children(node))
                assert children0 == children1, f"Mismatch in children for node {node}"


def test_walk_ast_descendants():
    """Test that every node returned by walk_ast_descendants is in all_nodes()."""
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        all_nodes = set(graph.all_nodes())
        for node in graph.walk_ast_descendants():
            assert (
                node in all_nodes
            ), f"Node {node} from walk_ast_descendants not in graph.all_nodes()"


def test_roundtrip_ast():
    """Test that to_ast() reproduces the original AST (by comparing dumps)."""
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        ast_original = program_utils.program_to_ast(fn)
        graph = program_graph.get_program_graph(fn)
        ast_reproduction = graph.to_ast()
        assert ast.dump(ast_original) == ast.dump(
            ast_reproduction
        ), f"AST roundtrip failed for function {name}"


def test_reconstruct_missing_ast():
    """Test that if all nodes’ ast_node fields are cleared, reconstruct_ast rebuilds the AST correctly."""
    for name, fn in inspect.getmembers(cftc, predicate=inspect.isfunction):
        graph = program_graph.get_program_graph(fn)
        ast_original = graph.root.ast_node
        # Clear all ast_node references.
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


def test_remove_and_readd_edge():
    """Test that removing and re‑adding an edge updates the graph correctly.
    Note: With multiple parents allowed, the parent_map now stores a set of parent IDs.
    """
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
            assert edge.id1 not in graph.parent_map.get(
                edge.id2, set()
            ), "Parent edge still in parent_map"
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
            assert edge.id1 in graph.parent_map.get(
                edge.id2, set()
            ), "Parent edge not in parent_map after re-add"


# --- End of Test Suite ---

if __name__ == "__main__":
    pytest.main()
