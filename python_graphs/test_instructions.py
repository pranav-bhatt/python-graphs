import ast
import pytest
from python_graphs import instruction as instruction_module


# --- Helper function for creating instructions ---
def create_instruction(source):
    node = ast.parse(source)
    node = instruction_module._canonicalize(node)
    return instruction_module.Instruction(node)


# --- Tests for basic Instruction existence and represent_same_program ---


def test_instruction_exists():
    assert instruction_module.Instruction is not None


def test_represent_same_program_basic_positive_case():
    program1 = ast.parse("x + 1")
    program2 = ast.parse("x + 1")
    assert instruction_module.represent_same_program(program1, program2)


def test_represent_same_program_basic_negative_case():
    program1 = ast.parse("x + 1")
    program2 = ast.parse("x + 2")
    assert not instruction_module.represent_same_program(program1, program2)


def test_represent_same_program_different_contexts():
    full_program1 = ast.parse("y = x + 1")  # y is a write
    program1 = full_program1.body[0].targets[0]  # 'y'
    program2 = ast.parse("y")  # y is a read
    assert instruction_module.represent_same_program(program1, program2)


# --- Additional tests for represent_same_program branches ---


def test_represent_same_program_diff_types():
    # Create two nodes of different types.
    n1 = ast.Name(id="x", ctx=ast.Load())
    n2 = ast.Constant(value=1)
    assert not instruction_module.represent_same_program(n1, n2)


def test_represent_same_program_non_ast():
    # Test with non-AST values.
    assert instruction_module.represent_same_program("abc", "abc")
    assert not instruction_module.represent_same_program("abc", "def")


def test_represent_same_program_list_mismatch():
    # Compare two ast.List nodes with different elements.
    list1 = ast.List(elts=[ast.Constant(value=1)], ctx=ast.Load())
    list2 = ast.List(elts=[ast.Constant(value=2)], ctx=ast.Load())
    assert not instruction_module.represent_same_program(list1, list2)


# --- Tests for get_accesses, get_reads, get_writes, and create_writes ---


def test_get_accesses_simple_expr():
    instr = create_instruction("x + 1")
    # Expect one read: x.
    assert instr.get_read_names() == {"x"}
    assert instr.get_write_names() == set()


def test_get_accesses_return():
    instr = create_instruction("return x + y + z")
    assert instr.get_read_names() == {"x", "y", "z"}
    assert instr.get_write_names() == set()


def test_get_accesses_function_call():
    instr = create_instruction("fn(a, b, c)")
    assert instr.get_read_names() == {"fn", "a", "b", "c"}
    assert instr.get_write_names() == set()


def test_get_accesses_assignment():
    instr = create_instruction("c = fn(a, b, c)")
    assert instr.get_read_names() == {"fn", "a", "b", "c"}
    assert instr.get_write_names() == {"c"}


def test_get_accesses_augassign():
    instr = create_instruction("x += 1")
    assert instr.get_read_names() == {"x"}
    assert instr.get_write_names() == {"x"}
    instr = create_instruction("x *= y")
    assert instr.get_read_names() == {"x", "y"}
    assert instr.get_write_names() == {"x"}


def test_get_accesses_augassign_subscript():
    instr = create_instruction("x[0] *= y")
    # Not recognized as a write of x.
    assert instr.get_read_names() == {"x", "y"}
    assert instr.get_write_names() == set()


def test_get_accesses_augassign_attribute():
    instr = create_instruction("x.attribute *= y")
    # Not recognized as a write of x.
    assert instr.get_read_names() == {"x", "y"}
    assert instr.get_write_names() == set()


def test_get_accesses_subscript():
    instr = create_instruction("x[0] = y")
    # Not recognized as a write of x.
    assert instr.get_read_names() == {"x", "y"}
    assert instr.get_write_names() == set()


def test_get_accesses_attribute():
    instr = create_instruction("x.attribute = y")
    # Not recognized as a write of x.
    assert instr.get_read_names() == {"x", "y"}
    assert instr.get_write_names() == set()


# --- Tests for access helper functions ---


def test_create_writes_non_ast():
    # When node is not an AST, create_writes should return a list with one tuple.
    writes = instruction_module.create_writes("not an AST", parent="parent")
    assert writes == [("write", "not an AST", "parent")]


def test_access_name_with_tuple_str():
    # When access is a tuple with a string at index 1.
    access = ("read", "foo", None)
    assert instruction_module.access_name(access) == "foo"


def test_access_name_with_tuple_ast():
    # When access is a tuple with an ast.Name at index 1.
    name_node = ast.Name(id="bar", ctx=ast.Load())
    access = ("read", name_node, None)
    assert instruction_module.access_name(access) == "bar"


def test_access_kind_and_name():
    # Test access_kind_and_name for an ast.Name access.
    name_node = ast.Name(id="baz", ctx=ast.Load())
    # Direct ast.Name case.
    result = instruction_module.access_kind_and_name(name_node)
    assert result == "read-baz"  # Since ast.Load gives a read.
    # For a tuple access.
    access = ("write", name_node, None)
    result2 = instruction_module.access_kind_and_name(access)
    assert result2 == "write-baz"


def test_access_identifier():
    assert instruction_module.access_identifier("x", "read") == "read-x"


# --- Test error branch in Instruction constructor ---
def test_instruction_invalid_node():
    with pytest.raises(TypeError):
        instruction_module.Instruction(123)  # not an AST instance


# --- Tests for Instruction.contains_subprogram ---


def test_contains_subprogram_exact_match():
    # With source set, an exact match is required.
    instr = create_instruction("x = 1")
    # Set the source so that exact match is required.
    instr.source = "exact"
    # Should match if the canonicalized node is exactly equal.
    assert instr.contains_subprogram(instr.node)
    # A different node (e.g. a parsed constant) should not match.
    different_node = ast.parse("2").body[0]
    assert not instr.contains_subprogram(different_node)


def test_contains_subprogram_subtree():
    # Without source, a subtree match is used.
    instr = create_instruction("x = 1")
    # Extract a subtree (e.g. the target 'x').
    target = instr.node.targets[0] if hasattr(instr.node, "targets") else None
    if target is not None:
        assert instr.contains_subprogram(target)
    else:
        pytest.skip("No targets found in the instruction node.")


# --- Additional tests for AccessVisitor behavior ---
def test_access_visitor_augassign():
    # Test the AccessVisitor on an augmented assignment.
    node = ast.parse("x += 1").body[0]
    visitor = instruction_module.AccessVisitor()
    visitor.visit(node)
    # For "x += 1", we expect the following accesses:
    # - The value "1" from node.value should be visited (but since it's a Constant, not a Name, it might be skipped)
    # - A tuple ("read", x, node) is added and then a visit of node.target.
    # So at least one read for x and then the x in node.target.
    accesses = visitor.accesses
    # Filter for ast.Name accesses.
    names = [
        instruction_module.access_name(a) if isinstance(a, tuple) else a.id
        for a in accesses
    ]
    # We expect "x" to appear at least once.
    assert "x" in names


# --- Test get_reads_from_ast_node and get_writes_from_ast_node directly ---
def test_get_reads_from_ast_node():
    node = ast.parse("x + y").body[0].value
    reads = instruction_module.get_reads_from_ast_node(node)
    names = {
        instruction_module.access_name(a) if not isinstance(a, ast.AST) else a.id
        for a in reads
    }
    assert names == {"x", "y"}


def test_get_writes_from_ast_node():
    node = ast.parse("x = y").body[0]
    writes = instruction_module.get_writes_from_ast_node(node)
    # For assignment "x = y", expect "x" as a write.
    names = {
        instruction_module.access_name(a) if not isinstance(a, ast.AST) else a.id
        for a in writes
    }
    assert names == {"x"}
