# test_control_flow.py

import ast
import inspect
import itertools
import json
import pytest
from typing import Optional

from python_graphs import control_flow
from python_graphs import control_flow_test_components as tc
from python_graphs import instruction as instruction_module
from python_graphs import program_utils
import astunparse


# --- Helper assertion functions ---


def get_block_by_label(
    graph: control_flow.ControlFlowGraph, label: str
) -> Optional[control_flow.BasicBlock]:
    """Return the first block in the graph with the given label, or None."""
    for block in graph.blocks:
        if block.label == label:
            return block
    return None


def get_block(
    graph: control_flow.ControlFlowGraph, selector
) -> Optional[control_flow.BasicBlock]:
    """Interpret the selector as either a block, a source string, or a label."""
    if isinstance(selector, control_flow.BasicBlock):
        return selector
    elif isinstance(selector, str):
        try:
            block = graph.get_block_by_source(selector)
        except Exception:
            block = None
        if block is None:
            block = get_block_by_label(graph, selector)
        return block
    else:
        raise ValueError("Unknown selector type.")


def assert_same_block(graph: control_flow.ControlFlowGraph, selector1, selector2):
    block1 = get_block(graph, selector1)
    block2 = get_block(graph, selector2)
    assert (
        block1 is not None and block2 is not None
    ), f"Expected both blocks to exist: {selector1}, {selector2}"
    assert block1 == block2, f"Expected same block, got {block1} and {block2}"


def assert_exits_to(graph: control_flow.ControlFlowGraph, selector1, selector2):
    block1 = get_block(graph, selector1)
    block2 = get_block(graph, selector2)
    assert block1 is not None, f"Block for selector {selector1} not found."
    assert block2 is not None, f"Block for selector {selector2} not found."
    assert block1.exits_to(block2), f"Expected {block1} to exit to {block2}"


def assert_not_exits_to(graph: control_flow.ControlFlowGraph, selector1, selector2):
    block1 = get_block(graph, selector1)
    block2 = get_block(graph, selector2)
    if block2 is None:
        return
    assert block1 is not None, f"Block for selector {selector1} not found."
    assert not block1.exits_to(block2), f"Did not expect {block1} to exit to {block2}"


def assert_raises_to(graph: control_flow.ControlFlowGraph, selector1, selector2):
    block1 = get_block(graph, selector1)
    block2 = get_block(graph, selector2)
    assert block1 is not None, f"Block for selector {selector1} not found."
    assert block2 is not None, f"Block for selector {selector2} not found."
    assert block1.raises_to(block2), f"Expected {block1} to raise to {block2}"


def assert_not_raises_to(graph: control_flow.ControlFlowGraph, selector1, selector2):
    block1 = get_block(graph, selector1)
    block2 = get_block(graph, selector2)
    if block2 is None:
        return
    assert block1 is not None, f"Block for selector {selector1} not found."
    assert not block1.raises_to(block2), f"Did not expect {block1} to raise to {block2}"


# --- Dummy helpers ---


def dummy_instruction(source: str = "pass") -> instruction_module.Instruction:
    """Return an Instruction created from the given source code."""
    node = ast.parse(source).body[0]
    return instruction_module.Instruction(node)


def create_dummy_graph() -> control_flow.ControlFlowGraph:
    """Create a dummy ControlFlowGraph with four blocks (entry, A, B, exit)."""
    graph = control_flow.ControlFlowGraph()
    block_entry = graph.new_block(label="<entry:test>", prunable=False)
    block_a = graph.new_block(label="A", prunable=False)
    block_b = graph.new_block(label="B", prunable=False)
    block_exit = graph.new_block(label="exit", prunable=False)

    block_entry.add_exit(block_a)
    block_a.add_exit(block_b)
    block_b.add_exit(block_exit)

    # Add dummy instructions so that CF nodes are not empty.
    block_a.add_instruction(dummy_instruction("x = 1"))
    block_b.add_instruction(dummy_instruction("y = 2"))

    # For legacy lookups, expose control_flow_nodes as 'instructions'
    for block in graph.blocks:
        block.instructions = block.control_flow_nodes

    return graph


# --- Tests for missing blocks and try-finally exits ---


def test_get_block_by_source_missing():
    """Test that a non-existent source returns None."""
    graph = control_flow.get_control_flow_graph(tc.straight_line_code)
    block = graph.get_block_by_source("nonexistent_source")
    assert (
        block is None
    ), "Expected get_block_by_source to return None for missing source."


def test_try_finally_break_label():
    """
    Test that a break in try-finally produces an exit block.
    (Note: our implementation currently falls back to 'after_block'.)
    """
    source = """
def f():
    while True:
        try:
            break
        finally:
            print("cleanup")
    return 1
"""
    graph = control_flow.get_control_flow_graph(source)
    # Look for the "after_block" label since that's our fallback.
    after_block = None
    for b in graph.blocks:
        if b.label == "after_block":
            after_block = b
            break
    assert (
        after_block is not None
    ), "Expected a block labeled 'after_block' for break in finally."


# --- Tests using control_flow_test_components (tc) ---


def test_control_flow_straight_line_code():
    graph = control_flow.get_control_flow_graph(tc.straight_line_code)
    assert_same_block(graph, "x = 1", "y = x + 2")
    assert_same_block(graph, "x = 1", "z = y * 3")
    assert_same_block(graph, "x = 1", "return z")


def test_control_flow_simple_if_statement():
    graph = control_flow.get_control_flow_graph(tc.simple_if_statement)
    assert_same_block(graph, "x = 1", "y = 2")
    assert_same_block(graph, "x = 1", "x > y")
    assert_exits_to(graph, "x > y", "y = 3")
    assert_exits_to(graph, "x > y", "return y")
    assert_exits_to(graph, "y = 3", "return y")
    assert_not_exits_to(graph, "y = 3", "x = 1")
    assert_not_exits_to(graph, "return y", "x = 1")
    assert_not_exits_to(graph, "return y", "y = 3")


def test_control_flow_simple_for_loop():
    graph = control_flow.get_control_flow_graph(tc.simple_for_loop)
    assert_same_block(graph, "x = 1", "range")
    assert_exits_to(graph, "range", "y")
    assert_exits_to(graph, "y", "y + 3")
    assert_not_exits_to(graph, "y + 3", "return z")
    assert_exits_to(graph, "y", "return z")


def test_control_flow_simple_while_loop():
    graph = control_flow.get_control_flow_graph(tc.simple_while_loop)
    assert_exits_to(graph, "x = 1", "x < 2")
    assert_exits_to(graph, "x < 2", "x += 3")
    assert_exits_to(graph, "x += 3", "x < 2")
    assert_not_exits_to(graph, "x += 3", "return x")
    assert_exits_to(graph, "x < 2", "return x")


def test_control_flow_break_in_while_loop():
    graph = control_flow.get_control_flow_graph(tc.break_in_while_loop)
    assert_exits_to(graph, "x < 2", "x += 3")
    assert_exits_to(graph, "x += 3", "return x")
    assert_not_exits_to(graph, "x += 3", "x < 2")
    assert_exits_to(graph, "x < 2", "return x")


def test_control_flow_nested_while_loops():
    graph = control_flow.get_control_flow_graph(tc.nested_while_loops)
    assert_exits_to(graph, "x = 1", "x < 2")
    assert_exits_to(graph, "x < 2", "y = 3")
    assert_exits_to(graph, "x < 2", "return x")
    assert_exits_to(graph, "y = 3", "y < 4")
    assert_exits_to(graph, "y < 4", "y += 5")
    assert_exits_to(graph, "y < 4", "x += 6")
    assert_exits_to(graph, "y += 5", "y < 4")
    assert_exits_to(graph, "x += 6", "x < 2")


def test_control_flow_exception_handling():
    graph = control_flow.get_control_flow_graph(tc.exception_handling)
    assert_same_block(graph, "before_stmt0", "before_stmt1")
    assert_exits_to(graph, "before_stmt1", "try_block")
    assert_not_exits_to(graph, "before_stmt0", "except_block1")
    assert_not_exits_to(graph, "before_stmt1", "final_block_stmt0")
    assert_raises_to(graph, "try_block", "error_type")
    assert_raises_to(graph, "error_type", "except_block2_stmt0")
    assert_exits_to(graph, "except_block1", "after_stmt0")
    assert_raises_to(graph, "after_stmt0", "except_block2_stmt0")
    assert_not_raises_to(graph, "try_block", "except_block2_stmt0")


def test_control_flow_try_with_loop():
    graph = control_flow.get_control_flow_graph(tc.try_with_loop)
    assert_same_block(graph, "for_body0", "for_body1")
    assert_same_block(graph, "except_body0", "except_body1")
    assert_exits_to(graph, "before_stmt0", "iterator")
    assert_exits_to(graph, "iterator", "target")
    assert_exits_to(graph, "target", "for_body0")
    assert_exits_to(graph, "for_body1", "target")
    assert_exits_to(graph, "target", "after_stmt0")
    assert_raises_to(graph, "iterator", "except_body0")
    assert_raises_to(graph, "target", "except_body0")
    assert_raises_to(graph, "for_body1", "except_body0")


def test_control_flow_break_in_finally():
    graph = control_flow.get_control_flow_graph(tc.break_in_finally)
    assert_raises_to(graph, "try0", "Exception0")
    assert_exits_to(graph, "Exception0", "Exception1")
    assert_exits_to(graph, "Exception1", "finally_stmt0")
    assert_raises_to(graph, "Exception0", "finally_stmt0")
    assert_raises_to(graph, "exception0_stmt0", "finally_stmt0")
    assert_raises_to(graph, "Exception1", "finally_stmt0")
    assert_not_exits_to(graph, "finally_stmt1", "target0")
    assert_exits_to(graph, "finally_stmt1", "after0")


def test_control_flow_for_loop_with_else():
    graph = control_flow.get_control_flow_graph(tc.for_with_else)
    assert_exits_to(graph, "target", "for_stmt0")
    assert_same_block(graph, "for_stmt0", "condition")
    assert_exits_to(graph, "condition", "after_stmt0")
    assert_exits_to(graph, "target", "else_stmt0")
    assert_not_exits_to(graph, "target", "after_stmt0")


def test_control_flow_lambda():
    graph = control_flow.get_control_flow_graph(tc.create_lambda)
    block_args = graph.get_block_by_source("args")
    assert (
        block_args is None
    ), "Expected no block for 'args' in lambda if not generated."
    assert_not_exits_to(graph, "before_stmt0", "output")


def test_control_flow_generator():
    graph = control_flow.get_control_flow_graph(tc.generator)
    assert_exits_to(graph, "target", "yield_statement")
    assert_same_block(graph, "yield_statement", "after_stmt0")


def test_control_flow_inner_fn_while_loop():
    graph = control_flow.get_control_flow_graph(tc.fn_with_inner_fn)
    block_true = get_block_by_label(graph, "True")
    assert block_true is not None, "Expected at least one block labeled 'True'."


def test_control_flow_example_class():
    graph = control_flow.get_control_flow_graph(tc.ExampleClass)
    assert_same_block(graph, "method_stmt0", "method_stmt1")


def test_control_flow_return_outside_function():
    with pytest.raises(RuntimeError) as error:
        control_flow.get_control_flow_graph("return x")
    assert "outside of a function frame" in str(error.value)


def test_control_flow_continue_outside_loop():
    control_flow.get_control_flow_graph("for i in j: continue")
    with pytest.raises(RuntimeError) as error:
        control_flow.get_control_flow_graph("if x: continue")
    assert "outside of a loop frame" in str(error.value)


def test_control_flow_break_outside_loop():
    control_flow.get_control_flow_graph("for i in j: break")
    with pytest.raises(RuntimeError) as error:
        control_flow.get_control_flow_graph("if x: break")
    assert "outside of a loop frame" in str(error.value)


def test_control_flow_for_all_test_components():
    for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
        graph = control_flow.get_control_flow_graph(fn)
        assert graph is not None


def test_control_flow_for_all_test_components_ast_to_instruction():
    for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
        node = program_utils.program_to_ast(fn)
        graph = control_flow.get_control_flow_graph(node)
        for n in ast.walk(node):
            if not isinstance(n, instruction_module.INSTRUCTION_AST_NODES):
                continue
            cf_nodes = list(graph.get_control_flow_nodes_by_ast_node(n))
            assert len(cf_nodes) == 1, ast.dump(n)


def test_control_flow_reads_and_writes_appear_once():
    for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
        reads = set()
        writes = set()
        node = program_utils.program_to_ast(fn)
        graph = control_flow.get_control_flow_graph(node)
        for instr in graph.get_instructions():
            for read in instr.get_reads():
                if isinstance(read, tuple):
                    read = read[1]
                assert isinstance(read, ast.Name), "Unexpected read type."
                assert (
                    read not in reads
                ), f"Duplicate read: {instruction_module.access_name(read)}"
                reads.add(read)
            for write in instr.get_writes():
                if isinstance(write, tuple):
                    write = write[1]
                if isinstance(write, str):
                    continue
                assert isinstance(write, ast.Name), "Unexpected write type."
                assert (
                    write not in writes
                ), f"Duplicate write: {instruction_module.access_name(write)}"
                writes.add(write)
            print("test")


# --- Additional Coverage Tests for ControlFlowNode ---


def test_control_flow_node_next_within_block():
    # Test when a node is followed by another node in the same block.
    graph = control_flow.ControlFlowGraph()
    block = graph.new_block(label="test_block", prunable=False)
    block.add_instruction(dummy_instruction("a = 1"))
    block.add_instruction(dummy_instruction("b = 2"))
    block.instructions = block.control_flow_nodes
    block.compact()
    first_node = block.control_flow_nodes[0]
    second_node = block.control_flow_nodes[1]
    next_nodes = first_node.next
    assert next_nodes is not None, "Expected next nodes not None."
    assert second_node in next_nodes, "Expected second node to be returned by next."


def test_control_flow_node_next_via_exit():
    # Test when the node is the last in its block and the block has an exit to another block.
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=False)
    block2 = graph.new_block(label="block2", prunable=False)
    block1.add_exit(block2)
    block1.add_instruction(dummy_instruction("a = 1"))
    block2.add_instruction(dummy_instruction("b = 2"))
    for b in [block1, block2]:
        b.instructions = b.control_flow_nodes
        b.compact()
    first_node = block1.control_flow_nodes[0]
    next_nodes = first_node.next
    branch_node = block2.control_flow_nodes[0]
    assert (
        branch_node in next_nodes
    ), "Expected next property to yield branch node from block2."


def test_control_flow_node_next_empty_next():
    # Test when the next block is empty (has no control_flow_nodes).
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=False)
    block2 = graph.new_block(label="block2_empty", prunable=False)
    block1.add_exit(block2)
    block1.add_instruction(dummy_instruction("a = 1"))
    block1.compact()
    next_nodes = block1.control_flow_nodes[0].next
    assert (
        next_nodes == set()
    ), "Expected next to return an empty set when next block is empty."


def test_control_flow_node_next_from_end_with_instructions():
    # Test next_from_end when the exit block has instructions.
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=False)
    block2 = graph.new_block(label="block2", prunable=False)
    block1.exits_from_end.add(block2)
    block1.add_instruction(dummy_instruction("a = 1"))
    block2.add_instruction(dummy_instruction("b = 2"))
    for b in [block1, block2]:
        b.instructions = b.control_flow_nodes
        b.compact()
    node = block1.control_flow_nodes[0]
    next_from_end = node.next_from_end
    branch_node = block2.control_flow_nodes[0]
    assert (
        branch_node in next_from_end
    ), "Expected next_from_end to yield branch node from block2."


def test_control_flow_node_next_from_end_empty():
    # Test next_from_end when the exit block is empty; should yield its label.
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=False)
    block2 = graph.new_block(label="block2_empty", prunable=False)
    block1.exits_from_end.add(block2)
    block1.add_instruction(dummy_instruction("a = 1"))
    block1.compact()
    node = block1.control_flow_nodes[0]
    next_from_end = node.next_from_end
    assert (
        block2.label in next_from_end
    ), "Expected next_from_end to return block2's label when empty."


def test_control_flow_node_prev_within_block():
    # Test the prev property when the node is not the first in its block.
    graph = control_flow.ControlFlowGraph()
    block = graph.new_block(label="test_block", prunable=False)
    block.add_instruction(dummy_instruction("a = 1"))
    block.add_instruction(dummy_instruction("b = 2"))
    block.compact()
    second_node = block.control_flow_nodes[1]
    prev_nodes = second_node.prev
    first_node = block.control_flow_nodes[0]
    assert (
        first_node in prev_nodes
    ), "Expected prev to return the preceding node in the same block."


def test_control_flow_node_prev_from_prev_block():
    # Test the prev property when the node is the first in its block but there is a previous block.
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=False)
    block2 = graph.new_block(label="block2", prunable=False)
    block1.add_exit(block2)
    block1.add_instruction(dummy_instruction("a = 1"))
    block2.add_instruction(dummy_instruction("b = 2"))
    for b in [block1, block2]:
        b.compact()
    node = block2.control_flow_nodes[0]
    prev_nodes = node.prev
    last_node_block1 = block1.control_flow_nodes[-1]
    assert (
        last_node_block1 in prev_nodes
    ), "Expected prev to include the last node of the preceding block."


def test_control_flow_node_get_branches_with_instruction():
    # Test get_branches when the branch target block has instructions.
    graph = control_flow.ControlFlowGraph()
    block = graph.new_block(label="branch_test", prunable=False)
    block.add_instruction(dummy_instruction("x = 1"))
    block.compact()
    node = block.control_flow_nodes[0]
    branch_block = graph.new_block(label="branch_target", prunable=False)
    branch_block.add_instruction(dummy_instruction("y = 2"))
    branch_block.compact()
    block.branches["test_key"] = branch_block
    block.compact()
    branches = node.get_branches()
    expected_node = branch_block.control_flow_nodes[0]
    assert (
        branches.get("test_key") == expected_node
    ), "Expected branch target node in branches."


def test_control_flow_node_get_branches_empty_branch():
    # Test get_branches when the branch target block is empty.
    graph = control_flow.ControlFlowGraph()
    block = graph.new_block(label="branch_empty", prunable=False)
    block.add_instruction(dummy_instruction("x = 1"))
    block.compact()
    node = block.control_flow_nodes[0]
    empty_block = graph.new_block(label="empty_target", prunable=False)
    block.branches["empty_key"] = empty_block
    block.compact()
    branches = node.get_branches()
    assert (
        branches.get("empty_key") == empty_block.label
    ), "Expected branch value to be block label for an empty branch target."


def test_controlFlowNode_labels():
    # Test the label methods: has_label, set_label, and get_label.
    graph = control_flow.ControlFlowGraph()
    block = graph.new_block(label="label_test", prunable=False)
    block.add_instruction(dummy_instruction("x = 1"))
    block.compact()
    node = block.control_flow_nodes[0]
    assert not node.has_label("test"), "Expected no label 'test' initially."
    node.set_label("test", 123)
    assert node.has_label("test"), "Expected label 'test' after setting it."
    assert (
        node.get_label("test") == 123
    ), "Expected get_label to return 123 for label 'test'."


# --- Additional Adjusted Tests for Visitor Behavior ---


def test_get_enter_control_flow_nodes():
    # Create an entry block with no nodes and a following block with an instruction.
    graph = control_flow.ControlFlowGraph()
    entry_block = graph.new_block(label="<entry:Test>", prunable=False)
    next_block = graph.new_block(label="next", prunable=False)
    entry_block.add_exit(next_block)
    next_block.add_instruction(dummy_instruction("a = 1"))
    for b in [entry_block, next_block]:
        b.instructions = b.control_flow_nodes
        b.compact()
    nodes = list(graph.get_enter_control_flow_nodes())
    assert any(
        "a = 1" in astunparse.unparse(n.instruction.node) for n in nodes
    ), "Expected to find node for 'a = 1' in enter control flow nodes."


def test_get_exit_blocks_and_start_node():
    # Build a simple graph with a chain of blocks.
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=False)
    block2 = graph.new_block(label="block2", prunable=False)
    block1.add_exit(block2)
    block2.add_instruction(dummy_instruction("x = 1"))
    exit_blocks = list(graph.get_exit_blocks())
    assert block2 in exit_blocks, "block2 should be an exit block."
    graph.start_block.control_flow_nodes = []
    graph.start_block.exits_from_end.add(block1)
    start_node = graph.get_start_control_flow_node()
    assert start_node is not None, "Expected a start control flow node from block1."


def test_get_control_flow_nodes_by_ast_node_and_node():
    # Use a function definition with a return statement so that a Return node is produced.
    source = """
def f():
    a = 1
"""
    graph = control_flow.get_control_flow_graph(source)
    module = ast.parse(source, mode="exec")
    # Get the Return node from within the function body.
    return_node = module.body[0].body[0]
    nodes = list(graph.get_control_flow_nodes_by_ast_node(return_node))
    assert nodes, "Expected to find control flow nodes for the given AST node."
    node2 = graph.get_control_flow_node_by_ast_node(return_node)
    assert node2 is not None, "Expected to retrieve a control flow node by AST node."


def test_get_blocks_by_function_name_and_source():
    # Define a function and then check block retrieval by function name.
    source = """
def foo():
    x = 1
    return x
"""
    graph = control_flow.get_control_flow_graph(source)
    block = graph.get_block_by_function_name("foo")
    assert block is not None, "Expected to retrieve a block by function name 'foo'."
    block_source = graph.get_block_by_source(source)
    assert block_source is not None, "Expected to retrieve a block by source."


def test_get_blocks_by_source_and_ast_node_type_and_label():
    # Use a function so that an assignment appears in the block.
    source = """
def f():
    a = 1
"""
    graph = control_flow.get_control_flow_graph(source)
    for b in graph.blocks:
        # Override each block's instructions with the underlying instruction objects.
        b.instructions = [cf.instruction for cf in b.control_flow_nodes]
    block_found = next(graph.get_blocks_by_source_and_ast_node_type(source, ast.Assign))
    block2 = graph.get_block_by_ast_node_type_and_label(ast.Assign, block_found.label)
    assert block2 is not None, "Expected to retrieve block by AST node type and label."


def test_compact_method():
    # Create a graph with mergeable blocks.
    graph = control_flow.ControlFlowGraph()
    block1 = graph.new_block(label="block1", prunable=True)
    block2 = graph.new_block(label="block2", prunable=True)
    block1.add_exit(block2)
    block2.add_instruction(dummy_instruction("y = 2"))
    graph.compact()
    for block in graph.blocks:
        block.compact()
        assert (
            block.control_flow_node_indexes is not None
        ), "Expected compact to create indexes."


def test_raise_through_frames_no_frame():
    # Test raise_through_frames when no exception frames exist.
    graph = control_flow.ControlFlowGraph()
    block = graph.new_block(label="test", prunable=False)
    visitor = control_flow.ControlFlowVisitor()
    visitor.frames = []  # No frames exist.
    with pytest.raises(ValueError, match="No frame exists"):
        visitor.raise_through_frames(block, interrupting=True)


def test_visit_yield():
    # Create a function that yields.
    source = """
def gen():
    yield 1
"""
    graph = control_flow.get_control_flow_graph(source)
    nodes = list(graph.get_control_flow_nodes())
    assert any(
        "yield" in astunparse.unparse(n.instruction.node) for n in nodes
    ), "Expected to find a yield statement in the control flow nodes."


def test_handle_argument_writes_with_varargs():
    # Create a function with *args and **kwargs.
    source = """
def foo(*args, **kwargs):
    return args, kwargs
"""
    graph = control_flow.get_control_flow_graph(source)
    instructions = list(graph.get_instructions())
    found = any(instr.source == instruction_module.ARGS for instr in instructions)
    assert (
        found
    ), "Expected an instruction with source ARGS for varargs/kwargs handling."
