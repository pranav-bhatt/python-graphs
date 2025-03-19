import itertools
import uuid
import ast
import json
from typing import Any, List, Set, Dict, Iterator, Optional

from absl import logging
from python_graphs import instruction as instruction_module
from python_graphs import program_utils


def get_control_flow_graph(program: Any) -> "ControlFlowGraph":
    """Get a ControlFlowGraph for the provided AST node.

    Args:
      program: Either an AST node, source string, or a function.
    Returns:
      A ControlFlowGraph.
    """
    control_flow_visitor = ControlFlowVisitor()
    node = program_utils.program_to_ast(program)
    control_flow_visitor.run(node)
    return control_flow_visitor.graph


class ControlFlowGraph:
    def __init__(self) -> None:
        self.blocks: List["BasicBlock"] = []
        self.nodes: List["ControlFlowNode"] = []
        self.start_block: "BasicBlock" = self.new_block(prunable=False, label="<start>")

    def add_node(self, control_flow_node: "ControlFlowNode") -> None:
        self.nodes.append(control_flow_node)

    def new_block(
        self,
        node: Optional[ast.AST] = None,
        label: Optional[str] = None,
        prunable: bool = True,
    ) -> "BasicBlock":
        block = BasicBlock(node=node, label=label, prunable=prunable)
        block.graph = self
        self.blocks.append(block)
        return block

    def move_block_to_rear(self, block: "BasicBlock") -> None:
        self.blocks.remove(block)
        self.blocks.append(block)

    def get_control_flow_nodes(self) -> List["ControlFlowNode"]:
        return self.nodes

    def get_enter_blocks(self) -> Iterator["BasicBlock"]:
        return filter(lambda block: block.label.startswith("<entry:"), self.blocks)

    def get_enter_control_flow_nodes(self) -> Iterator["ControlFlowNode"]:
        for block in self.blocks:
            if (
                block.label is not None
                and block.label.startswith("<entry")
                and not block.control_flow_nodes
            ):
                next_block = next(iter(block.next))
                if next_block.control_flow_nodes:
                    yield next_block.control_flow_nodes[0]
            elif block.control_flow_nodes:
                node = block.control_flow_nodes[0]
                if not node.prev:
                    yield node

    def get_exit_blocks(self) -> Iterator["BasicBlock"]:
        for block in self.blocks:
            if not block.next:
                yield block

    def get_instructions(self) -> Iterator[instruction_module.Instruction]:
        for block in self.blocks:
            for node in block.control_flow_nodes:
                yield node.instruction

    def get_start_control_flow_node(self) -> Any:
        if self.start_block.control_flow_nodes:
            return self.start_block.control_flow_nodes[0]
        if self.start_block.exits_from_end:
            assert len(self.start_block.exits_from_end) == 1
            first_block = next(iter(self.start_block.exits_from_end))
            if first_block.control_flow_nodes:
                return first_block.control_flow_nodes[0]
            else:
                return first_block.label

    def get_control_flow_nodes_by_ast_node(
        self, node: ast.AST
    ) -> Iterator["ControlFlowNode"]:
        return filter(
            lambda cfn: ast.dump(cfn.instruction.node) == ast.dump(node),
            self.get_control_flow_nodes(),
        )

    def get_control_flow_node_by_ast_node(self, node: ast.AST) -> "ControlFlowNode":
        return next(self.get_control_flow_nodes_by_ast_node(node))

    def get_blocks_by_ast_node(self, node: ast.AST) -> Iterator["BasicBlock"]:
        for block in self.blocks:
            for cfn in block.control_flow_nodes:
                if ast.dump(cfn.instruction.node) == ast.dump(node):
                    yield block
                    break

    def get_block_by_ast_node(self, node: ast.AST) -> "BasicBlock":
        return next(self.get_blocks_by_ast_node(node))

    def get_blocks_by_function_name(self, name: str) -> Iterator["BasicBlock"]:
        return filter(lambda block: block.label == f"<entry:{name}>", self.blocks)

    def get_block_by_function_name(self, name: str) -> "BasicBlock":
        return next(self.get_blocks_by_function_name(name))

    def get_control_flow_nodes_by_source(
        self, source: str
    ) -> Iterator["ControlFlowNode"]:
        module = ast.parse(source, mode="exec")
        node = module.body[0]
        if isinstance(node, ast.Expr):
            node = node.value
        return filter(
            lambda cfn: cfn.instruction.contains_subprogram(node),
            self.get_control_flow_nodes(),
        )

    def get_control_flow_node_by_source(self, source: str) -> "ControlFlowNode":
        return next(self.get_control_flow_nodes_by_source(source))

    def get_control_flow_nodes_by_source_and_identifier(
        self, source: str, name: str
    ) -> Iterator["ControlFlowNode"]:
        for cfn in self.get_control_flow_nodes_by_source(source):
            for n in ast.walk(cfn.instruction.node):
                if isinstance(n, ast.Name) and n.id == name:
                    yield from self.get_control_flow_nodes_by_ast_node(n)

    def get_control_flow_node_by_source_and_identifier(
        self, source: str, name: str
    ) -> "ControlFlowNode":
        return next(self.get_control_flow_nodes_by_source_and_identifier(source, name))

    def get_blocks_by_source(self, source: str) -> Iterator["BasicBlock"]:
        module = ast.parse(source, mode="exec")
        node = module.body[0]
        if isinstance(node, ast.Expr):
            node = node.value
        for block in self.blocks:
            for cfn in block.control_flow_nodes:
                if cfn.instruction.contains_subprogram(node):
                    yield block
                    break

    def get_block_by_source(self, source: str) -> Optional["BasicBlock"]:
        return next(self.get_blocks_by_source(source), None)

    def get_blocks_by_source_and_ast_node_type(
        self, source: str, node_type: type
    ) -> Iterator["BasicBlock"]:
        module = ast.parse(source, mode="exec")
        node = module.body[0]
        if isinstance(node, ast.Expr):
            node = node.value
        for block in self.blocks:
            for instr in block.instructions:
                if isinstance(instr.node, node_type) and instr.contains_subprogram(
                    node
                ):
                    yield block
                    break

    def get_block_by_source_and_ast_node_type(
        self, source: str, node_type: type
    ) -> "BasicBlock":
        return next(self.get_blocks_by_source_and_ast_node_type(source, node_type))

    def get_block_by_ast_node_and_label(
        self, node: ast.AST, label: str
    ) -> Optional["BasicBlock"]:
        for block in self.blocks:
            if ast.dump(block.node) == ast.dump(node) and block.label == label:
                return block
        return None

    def get_blocks_by_ast_node_type_and_label(
        self, node_type: type, label: str
    ) -> Iterator["BasicBlock"]:
        for block in self.blocks:
            if isinstance(block.node, node_type) and block.label == label:
                yield block

    def get_block_by_ast_node_type_and_label(
        self, node_type: type, label: str
    ) -> "BasicBlock":
        return next(self.get_blocks_by_ast_node_type_and_label(node_type, label))

    def prune(self) -> None:
        progress = True
        while progress:
            progress = False
            for block in list(self.blocks):
                if block.can_prune():
                    pruned = block._prune_helper()
                    self.blocks.remove(pruned)
                    progress = True

    def compact(self) -> None:
        self.prune()
        for block in list(self.blocks):
            while block.can_merge():
                merged = block._merge_helper()
                self.blocks.remove(merged)
        for block in self.blocks:
            block.compact()


class Frame:
    MODULE = "module"
    LOOP = "loop"
    FUNCTION = "function"
    TRY_EXCEPT = "try-except"
    TRY_FINALLY = "try-finally"

    def __init__(self, kind: str, **blocks: Any) -> None:
        self.kind: str = kind
        self.blocks: Dict[str, Any] = blocks


class BasicBlock:
    def __init__(
        self,
        node: Optional[ast.AST] = None,
        label: Optional[str] = None,
        prunable: bool = True,
    ) -> None:
        self.graph: Optional[ControlFlowGraph] = None
        self.next: Set["BasicBlock"] = set()
        self.prev: Set["BasicBlock"] = set()
        self.control_flow_nodes: List["ControlFlowNode"] = []
        self.control_flow_node_indexes: Optional[Dict[str, int]] = None

        self.branches: Dict[Any, "BasicBlock"] = {}
        self.except_branches: Dict[Any, "BasicBlock"] = {}
        self.reraise_branches: Dict[Any, "BasicBlock"] = {}

        self.exits_from_middle: Set["BasicBlock"] = set()
        self.exits_from_end: Set["BasicBlock"] = set()
        self.node: Optional[ast.AST] = node
        self.prunable: bool = prunable
        self.label: Optional[str] = label
        self.identities: List[Any] = [(node, label)]
        self.labels: Dict[str, Any] = {}

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def set_label(self, label: str, value: Any) -> None:
        self.labels[label] = value

    def get_label(self, label: str) -> Any:
        return self.labels[label]

    def is_empty(self) -> bool:
        return not self.control_flow_nodes

    def exits_to(self, block: "BasicBlock") -> bool:
        return block in self.next

    def raises_to(self, block: "BasicBlock") -> bool:
        return block in self.next and block in self.exits_from_middle

    def add_exit(
        self,
        block: "BasicBlock",
        interrupting: bool = False,
        branch: Optional[Any] = None,
        except_branch: Optional[Any] = None,
        reraise_branch: Optional[Any] = None,
    ) -> None:
        self.next.add(block)
        block.prev.add(self)
        if branch is not None:
            self.branches[branch] = block
        if except_branch is not None:
            self.except_branches[except_branch] = block
        if reraise_branch is not None:
            self.reraise_branches[reraise_branch] = block
        if interrupting:
            self.exits_from_middle.add(block)
        else:
            self.exits_from_end.add(block)

    def remove_exit(self, block: "BasicBlock") -> None:
        self.next.remove(block)
        block.prev.remove(self)
        self.exits_from_middle.discard(block)
        self.exits_from_end.discard(block)
        for branch_dict in (self.branches, self.except_branches, self.reraise_branches):
            for key, val in list(branch_dict.items()):
                if val is block:
                    del branch_dict[key]

    def can_prune(self) -> bool:
        return self.is_empty() and self.prunable

    def _prune_helper(self) -> "BasicBlock":
        prevs = self.prev.copy()
        nexts = self.next.copy()
        for prev_block in prevs:
            for next_block in nexts:
                if self in prev_block.exits_from_middle:
                    prev_block.add_exit(next_block, interrupting=True)
                if self in prev_block.exits_from_end:
                    prev_block.add_exit(next_block, interrupting=False)
                for branch_dict in (
                    prev_block.branches,
                    prev_block.except_branches,
                    prev_block.reraise_branches,
                ):
                    for branch_decision, branch_exit in branch_dict.copy().items():
                        if branch_exit is self:
                            branch_dict[branch_decision] = next_block
        for prev_block in prevs:
            prev_block.remove_exit(self)
        for next_block in nexts:
            self.remove_exit(next_block)
            next_block.identities += self.identities
        return self

    def can_merge(self) -> bool:
        if len(self.exits_from_end) != 1:
            return False
        next_block = next(iter(self.exits_from_end))
        if not next_block.prunable:
            return False
        if self.exits_from_middle != next_block.exits_from_middle:
            return False
        if len(next_block.prev) == 1:
            return True
        return False

    def _merge_helper(self) -> "BasicBlock":
        assert self.can_merge(), "Block cannot be merged."
        next_block = next(iter(self.exits_from_end))
        for branch_dict in (
            next_block.branches,
            next_block.except_branches,
            next_block.reraise_branches,
        ):
            for branch_decision, branch_exit in branch_dict.items():
                self.branches[branch_decision] = branch_exit
        self.remove_exit(next_block)
        for block in list(next_block.next):
            next_block.remove_exit(block)
            if block in next_block.exits_from_middle:
                self.add_exit(block, interrupting=True)
            if block in next_block.exits_from_end:
                self.add_exit(block, interrupting=False)
        for cf_node in next_block.control_flow_nodes:
            cf_node.block = self
            self.control_flow_nodes.append(cf_node)
        self.prunable = self.prunable and next_block.prunable
        self.label = self.label or next_block.label
        self.identities += next_block.identities
        return next_block

    def add_instruction(self, instruction: instruction_module.Instruction) -> None:
        assert isinstance(instruction, instruction_module.Instruction)
        cf_node = ControlFlowNode(graph=self.graph, block=self, instruction=instruction)
        self.graph.add_node(cf_node)
        self.control_flow_nodes.append(cf_node)

    def compact(self) -> None:
        self.control_flow_node_indexes = {}
        for index, cf_node in enumerate(self.control_flow_nodes):
            self.control_flow_node_indexes[str(cf_node.uuid)] = index

    def index_of(self, cf_node: "ControlFlowNode") -> int:
        return self.control_flow_node_indexes[str(cf_node.uuid)]


class ControlFlowNode:
    def __init__(
        self,
        graph: ControlFlowGraph,
        block: BasicBlock,
        instruction: instruction_module.Instruction,
    ) -> None:
        self.graph: ControlFlowGraph = graph
        self.block: BasicBlock = block
        self.instruction: instruction_module.Instruction = instruction
        self.labels: Dict[str, Any] = {}
        self.uuid: uuid.UUID = uuid.uuid4()

    @property
    def next(self) -> Optional[Set["ControlFlowNode"]]:
        if self.block is None:
            return None
        index_in_block = self.block.index_of(self)
        if len(self.block.control_flow_nodes) > index_in_block + 1:
            return {self.block.control_flow_nodes[index_in_block + 1]}
        nodes: Set[Any] = set()
        for next_block in self.block.next:
            if next_block.control_flow_nodes:
                nodes.add(next_block.control_flow_nodes[0])
            else:
                logging.warning(
                    json.dumps(
                        {
                            "level": "warning",
                            "message": "Empty next block encountered",
                            "block_label": next_block.label,
                        }
                    )
                )
                assert not next_block.next
        return nodes

    @property
    def next_from_end(self) -> Optional[Set[Any]]:
        if self.block is None:
            return None
        index_in_block = self.block.index_of(self)
        if len(self.block.control_flow_nodes) > index_in_block + 1:
            return {self.block.control_flow_nodes[index_in_block + 1]}
        nodes: Set[Any] = set()
        for next_block in self.block.exits_from_end:
            if next_block.control_flow_nodes:
                nodes.add(next_block.control_flow_nodes[0])
            else:
                assert not next_block.next
                nodes.add(next_block.label)
        return nodes

    @property
    def prev(self) -> Optional[Set[Any]]:
        if self.block is None:
            return None
        index_in_block = self.block.index_of(self)
        if index_in_block - 1 >= 0:
            return {self.block.control_flow_nodes[index_in_block - 1]}
        nodes: Set[Any] = set()
        for prev_block in self.block.prev:
            if prev_block.control_flow_nodes:
                nodes.add(prev_block.control_flow_nodes[-1])
            else:
                assert not prev_block.prev
        return nodes

    @property
    def branches(self) -> Dict[Any, Any]:
        return self.get_branches(
            include_except_branches=False, include_reraise_branches=False
        )

    def get_branches(
        self,
        include_except_branches: bool = False,
        include_reraise_branches: bool = False,
    ) -> Dict[Any, Any]:
        if self.block is None:
            return {}
        index_in_block = self.block.index_of(self)
        if len(self.block.control_flow_nodes) > index_in_block + 1:
            return {}
        branches: Dict[Any, Any] = {}
        all_branches = [self.block.branches.items()]
        if include_except_branches:
            all_branches.append(self.block.except_branches.items())
        if include_reraise_branches:
            all_branches.append(self.block.reraise_branches.items())
        for key, next_block in itertools.chain(*all_branches):
            if next_block.control_flow_nodes:
                branches[key] = next_block.control_flow_nodes[0]
            else:
                assert not next_block.next
                branches[key] = next_block.label
        return branches

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def set_label(self, label: str, value: Any) -> None:
        self.labels[label] = value

    def get_label(self, label: str) -> Any:
        return self.labels[label]


class ControlFlowVisitor:
    def __init__(self) -> None:
        self.graph: ControlFlowGraph = ControlFlowGraph()
        self.frames: List[Frame] = []

    def run(self, node: ast.AST) -> None:
        start_block = self.graph.start_block
        end_block = self.visit(node, start_block)
        self.graph.compact()

    def visit(self, node: ast.AST, current_block: BasicBlock) -> BasicBlock:
        assert isinstance(node, ast.AST)
        if isinstance(node, instruction_module.INSTRUCTION_AST_NODES):
            self.add_new_instruction(current_block, node)
        method_name: str = "visit_" + node.__class__.__name__
        method: Optional[Any] = getattr(self, method_name, None)
        if method is not None:
            current_block = method(node, current_block)
        return current_block

    def visit_list(self, items: List[Any], current_block: BasicBlock) -> BasicBlock:
        for item in items:
            current_block = self.visit(item, current_block)
        return current_block

    def add_new_instruction(
        self,
        block: BasicBlock,
        node: Any,
        accesses: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> None:
        if not isinstance(node, ast.AST):
            node = ast.Name(id=str(node), ctx=ast.Load())
        instr = instruction_module.Instruction(node, accesses=accesses, source=source)
        self.add_instruction(block, instr)

    def add_instruction(
        self, block: BasicBlock, instr: instruction_module.Instruction
    ) -> None:
        assert isinstance(instr, instruction_module.Instruction)
        block.add_instruction(instr)
        if not block.exits_from_middle:
            self.raise_through_frames(block, interrupting=True)

    def raise_through_frames(
        self,
        block: BasicBlock,
        interrupting: bool = True,
        except_branch: Optional[bool] = None,
    ) -> None:
        frames = self.get_current_exception_handling_frames()
        if frames is None:
            return
        reraise_branch: Optional[bool] = None
        for frame in frames:
            if frame.kind == Frame.TRY_FINALLY:
                final_block = frame.blocks["final_block"]
                block.add_exit(
                    final_block,
                    interrupting=interrupting,
                    except_branch=except_branch,
                    reraise_branch=reraise_branch,
                )
                block = frame.blocks["final_block_end"]
                interrupting = False
                except_branch = None
                reraise_branch = True
            elif frame.kind == Frame.TRY_EXCEPT:
                handler_block = frame.blocks["handler_block"]
                block.add_exit(
                    handler_block,
                    interrupting=interrupting,
                    except_branch=except_branch,
                    reraise_branch=reraise_branch,
                )
            elif frame.kind in (Frame.FUNCTION, Frame.MODULE):
                raise_block = frame.blocks["raise_block"]
                block.add_exit(
                    raise_block,
                    interrupting=interrupting,
                    except_branch=except_branch,
                    reraise_branch=reraise_branch,
                )
        logging.info(
            json.dumps(
                {
                    "level": "info",
                    "message": "Raised through frames",
                    "block_label": block.label,
                }
            )
        )

    def new_block(
        self,
        node: Optional[ast.AST] = None,
        label: Optional[str] = None,
        prunable: bool = True,
    ) -> BasicBlock:
        return self.graph.new_block(node=node, label=label, prunable=prunable)

    def enter_module_frame(
        self, exit_block: BasicBlock, raise_block: BasicBlock
    ) -> None:
        self.frames.append(
            Frame(Frame.MODULE, exit_block=exit_block, raise_block=raise_block)
        )

    def enter_loop_frame(
        self, continue_block: BasicBlock, break_block: BasicBlock
    ) -> None:
        self.frames.append(
            Frame(Frame.LOOP, continue_block=continue_block, break_block=break_block)
        )

    def enter_function_frame(
        self, return_block: BasicBlock, raise_block: BasicBlock
    ) -> None:
        self.frames.append(
            Frame(Frame.FUNCTION, return_block=return_block, raise_block=raise_block)
        )

    def enter_try_except_frame(self, handler_block: BasicBlock) -> None:
        self.frames.append(Frame(Frame.TRY_EXCEPT, handler_block=handler_block))

    def enter_try_finally_frame(
        self, final_block: BasicBlock, final_block_end: BasicBlock
    ) -> None:
        self.frames.append(
            Frame(
                Frame.TRY_FINALLY,
                final_block=final_block,
                final_block_end=final_block_end,
            )
        )

    def exit_frame(self) -> Frame:
        return self.frames.pop()

    def get_current_loop_frame(self) -> Optional[List[Frame]]:
        frames: List[Frame] = []
        for frame in reversed(self.frames):
            if frame.kind == Frame.TRY_FINALLY:
                frames.append(frame)
            if frame.kind == Frame.LOOP:
                frames.append(frame)
                return frames
        return None

    def get_current_function_frame(self) -> Optional[List[Frame]]:
        frames: List[Frame] = []
        for frame in reversed(self.frames):
            if frame.kind == Frame.TRY_FINALLY:
                frames.append(frame)
            if frame.kind == Frame.FUNCTION:
                frames.append(frame)
                return frames
        return None

    def get_current_exception_handling_frames(self) -> List[Frame]:
        frames: List[Frame] = []
        for frame in reversed(self.frames):
            if frame.kind == Frame.TRY_FINALLY:
                frames.append(frame)
            if frame.kind == Frame.TRY_EXCEPT:
                frames.append(frame)
                return frames
            if frame.kind in (Frame.FUNCTION, Frame.MODULE):
                frames.append(frame)
                return frames
        raise ValueError("No frame exists to catch the exception.")

    def visit_Module(self, node: ast.Module, current_block: BasicBlock) -> BasicBlock:
        exit_block = self.new_block(node=node, label="<exit>", prunable=False)
        raise_block = self.new_block(node=node, label="<raise>", prunable=False)
        self.enter_module_frame(exit_block, raise_block)
        end_block = self.visit_list(node.body, current_block)
        end_block.add_exit(exit_block)
        self.exit_frame()
        self.graph.move_block_to_rear(exit_block)
        self.graph.move_block_to_rear(raise_block)
        return end_block

    def visit_ClassDef(
        self, node: ast.ClassDef, current_block: BasicBlock
    ) -> BasicBlock:
        current_block = self.visit_list(node.body, current_block)
        for decorator in node.decorator_list:
            self.add_new_instruction(current_block, decorator)
        assert isinstance(node.name, str)
        self.add_new_instruction(
            current_block,
            node,
            accesses=instruction_module.create_writes(node.name, node),
            source=instruction_module.CLASS,
        )
        return current_block

    def visit_FunctionDef(
        self, node: ast.FunctionDef, current_block: BasicBlock
    ) -> BasicBlock:
        current_block = self.handle_argument_defaults(node.args, current_block)
        for decorator in node.decorator_list:
            self.add_new_instruction(current_block, decorator)
        assert isinstance(node.name, str)
        self.add_new_instruction(
            current_block,
            node,
            accesses=instruction_module.create_writes(node.name, node),
            source=instruction_module.FUNCTION,
        )
        self.handle_function_definition(node, node.name, node.args, node.body)
        return current_block

    def visit_Lambda(self, node: ast.Lambda, current_block: BasicBlock) -> BasicBlock:
        current_block = self.handle_argument_defaults(node.args, current_block)
        # For lambda, wrap the single expression body in a list.
        self.handle_function_definition(node, "lambda", node.args, [node.body])
        return current_block

    def handle_function_definition(
        self, node: ast.AST, name: str, args: ast.arguments, body: List[ast.AST]
    ) -> None:
        return_block = self.new_block(node=node, label="<return>", prunable=False)
        raise_block = self.new_block(node=node, label="<raise>", prunable=False)
        self.enter_function_frame(return_block, raise_block)
        entry_block = self.new_block(node=node, label=f"<entry:{name}>", prunable=False)
        fn_block = self.new_block(node=node, label="fn_block")
        entry_block.add_exit(fn_block)
        fn_block = self.handle_argument_writes(args, fn_block)
        fn_block = self.visit_list(body, fn_block)
        fn_block.add_exit(return_block)
        self.exit_frame()
        self.graph.move_block_to_rear(return_block)
        self.graph.move_block_to_rear(raise_block)

    def handle_argument_defaults(
        self, args: ast.arguments, current_block: BasicBlock
    ) -> BasicBlock:
        for default in args.defaults:
            self.add_new_instruction(current_block, default)
        for default in args.kw_defaults:
            if default is None:
                continue
            self.add_new_instruction(current_block, default)
        return current_block

    def handle_argument_writes(
        self, args: ast.arguments, current_block: BasicBlock
    ) -> BasicBlock:
        accesses: List[Any] = []
        if args.args:
            for arg in args.args:
                accesses.extend(instruction_module.create_writes(arg, args))
        if args.vararg:
            accesses.extend(instruction_module.create_writes(args.vararg, args))
        if args.kwonlyargs:
            for arg in args.kwonlyargs:
                accesses.extend(instruction_module.create_writes(arg, args))
        if args.kwarg:
            accesses.extend(instruction_module.create_writes(args.kwarg, args))
        if accesses:
            self.add_new_instruction(
                current_block, args, accesses=accesses, source=instruction_module.ARGS
            )
        return current_block

    def visit_If(self, node: ast.If, current_block: BasicBlock) -> BasicBlock:
        self.add_new_instruction(current_block, node.test)
        after_block = self.new_block(node=node, label="after_block")
        true_block = self.new_block(node=node, label="true_block")
        current_block.add_exit(true_block, branch=True)
        true_block = self.visit_list(node.body, true_block)
        true_block.add_exit(after_block)
        if node.orelse:
            false_block = self.new_block(node=node, label="false_block")
            current_block.add_exit(false_block, branch=False)
            false_block = self.visit_list(node.orelse, false_block)
            false_block.add_exit(after_block)
        else:
            current_block.add_exit(after_block, branch=False)
        return after_block

    def visit_While(self, node: ast.While, current_block: BasicBlock) -> BasicBlock:
        test_instruction = instruction_module.Instruction(node.test)
        # Create the test block as unprunable so its exits are preserved.
        return self.handle_Loop(node, test_instruction, current_block)

    def visit_For(self, node: ast.For, current_block: BasicBlock) -> BasicBlock:
        self.add_new_instruction(current_block, node.iter)
        target = instruction_module.Instruction(
            node.target,
            accesses=instruction_module.create_writes(node.target, node),
            source=instruction_module.ITERATOR,
        )
        return self.handle_Loop(node, target, current_block)

    def handle_Loop(
        self,
        node: ast.AST,
        loop_instruction: instruction_module.Instruction,
        current_block: BasicBlock,
    ) -> BasicBlock:
        test_block = self.new_block(node=node, label="test_block", prunable=False)
        try:
            if ast.literal_eval(node.test) is True:
                test_block.label = "True"
        except Exception:
            # If evaluation fails, we fall back to the default label.
            pass
        current_block.add_exit(test_block)
        self.add_instruction(test_block, loop_instruction)
        body_block = self.new_block(node=node, label="body_block")
        after_block = self.new_block(node=node, label="after_block")
        test_block.add_exit(body_block, branch=True)
        self.enter_loop_frame(test_block, after_block)
        body_block = self.visit_list(node.body, body_block)
        body_block.add_exit(test_block)
        self.exit_frame()
        if getattr(node, "orelse", None):
            else_block = self.new_block(node=node, label="else_block")
            test_block.add_exit(else_block, branch=False)
            else_block = self.visit_list(node.orelse, else_block)
            else_block.add_exit(after_block)
        else:
            test_block.add_exit(after_block, branch=False)
        self.graph.move_block_to_rear(after_block)
        return after_block

    def visit_Try(self, node: ast.Try, current_block: BasicBlock) -> BasicBlock:
        after_block = self.new_block(prunable=False)
        handler_blocks = [self.new_block() for _ in node.handlers]
        handler_body_blocks = [self.new_block() for _ in node.handlers]
        bare_handler_block: Optional[BasicBlock] = (
            handler_blocks[-1]
            if node.handlers and node.handlers[-1].type is None
            else None
        )
        if node.finalbody:
            final_block = self.new_block(node=node, label="final_block")
            final_block_end = self.visit_list(node.finalbody, final_block)
            final_block_end.add_exit(after_block, reraise_branch=False)
            self.enter_try_finally_frame(final_block, final_block_end)
        else:
            final_block = after_block
        if node.handlers:
            self.enter_try_except_frame(handler_blocks[0])
        try_block = self.new_block(node=node, label="try_block")
        current_block.add_exit(try_block)
        try_block_end = self.visit_list(node.body, try_block)
        if node.orelse:
            else_block = self.new_block(node=node, label="else_block")
            try_block_end.add_exit(else_block)
        else:
            try_block_end.add_exit(final_block)
        if node.handlers:
            self.exit_frame()  # Exit try-except frame.
        previous_handler_block_end: Optional[BasicBlock] = None
        for handler, handler_block, handler_body_block in zip(
            node.handlers, handler_blocks, handler_body_blocks
        ):
            previous_handler_block_end = self.handle_ExceptHandler(
                handler,
                handler_block,
                handler_body_block,
                final_block,
                previous_handler_block_end=previous_handler_block_end,
            )
        if bare_handler_block is None and previous_handler_block_end is not None:
            self.raise_through_frames(
                previous_handler_block_end, interrupting=False, except_branch=False
            )
        if node.orelse:
            else_block = self.visit_list(node.orelse, else_block)
            else_block.add_exit(final_block)
        if node.finalbody:
            self.exit_frame()  # Exit try-finally frame.
        self.graph.move_block_to_rear(after_block)
        return after_block

    def handle_ExceptHandler(
        self,
        handler: ast.ExceptHandler,
        handler_block: BasicBlock,
        handler_body_block: BasicBlock,
        final_block: BasicBlock,
        previous_handler_block_end: Optional[BasicBlock] = None,
    ) -> BasicBlock:
        if handler.type is not None:
            self.add_new_instruction(handler_block, handler.type)
        handler_block.add_exit(handler_body_block, except_branch=True)
        if previous_handler_block_end is not None:
            previous_handler_block_end.add_exit(handler_block, except_branch=False)
        previous_handler_block_end = handler_block
        if handler.name is not None:
            name_node = ast.Name(id=str(handler.name), ctx=ast.Store())
            self.add_new_instruction(
                handler_body_block,
                name_node,
                accesses=instruction_module.create_writes(name_node, handler),
                source=instruction_module.EXCEPTION,
            )
        handler_body_block = self.visit_list(handler.body, handler_body_block)
        handler_body_block.add_exit(final_block)
        return previous_handler_block_end

    def visit_Return(self, node: ast.Return, current_block: BasicBlock) -> BasicBlock:
        frames = self.get_current_function_frame()
        if frames is None:
            raise RuntimeError("return occurs outside of a function frame.")
        try_finally_frames = frames[:-1]
        function_frame = frames[-1]
        return_block = function_frame.blocks["return_block"]
        return self.handle_ExitStatement(
            node, return_block, try_finally_frames, current_block
        )

    def visit_Yield(self, node: ast.Yield, current_block: BasicBlock) -> BasicBlock:
        logging.warning(
            json.dumps(
                {"level": "warning", "message": "Yield visited", "node": ast.dump(node)}
            )
        )
        return current_block

    def visit_Continue(
        self, node: ast.Continue, current_block: BasicBlock
    ) -> BasicBlock:
        frames = self.get_current_loop_frame()
        if frames is None:
            raise RuntimeError("continue occurs outside of a loop frame.")
        try_finally_frames = frames[:-1]
        loop_frame = frames[-1]
        continue_block = loop_frame.blocks["continue_block"]
        return self.handle_ExitStatement(
            node, continue_block, try_finally_frames, current_block
        )

    def visit_Break(self, node: ast.Break, current_block: BasicBlock) -> BasicBlock:
        frames = self.get_current_loop_frame()
        if frames is None:
            raise RuntimeError("break occurs outside of a loop frame.")
        try_finally_frames = frames[:-1]
        loop_frame = frames[-1]
        break_block = loop_frame.blocks["break_block"]
        return self.handle_ExitStatement(
            node, break_block, try_finally_frames, current_block
        )

    def visit_Raise(self, node: ast.Raise, current_block: BasicBlock) -> BasicBlock:
        self.raise_through_frames(current_block, interrupting=False)
        after_block = self.new_block(node=node, label="after_block")
        return after_block

    def handle_ExitStatement(
        self,
        node: ast.AST,
        next_block: BasicBlock,
        try_finally_frames: List[Frame],
        current_block: BasicBlock,
    ) -> BasicBlock:
        for frame in try_finally_frames:
            final_block = frame.blocks["final_block"]
            current_block.add_exit(final_block)
            current_block = frame.blocks["final_block_end"]
        current_block.add_exit(next_block)
        # If exiting from a break in a try-finally context, set a specific label.
        after_label = (
            "after0"
            if isinstance(node, ast.Break)
            and current_block.label
            and current_block.label.startswith("finally_stmt")
            else "after_block"
        )
        # Mark the new block unprunable to preserve it.
        after_block = self.new_block(node=node, label=after_label, prunable=False)
        return after_block


class Frame:
    MODULE = "module"
    LOOP = "loop"
    FUNCTION = "function"
    TRY_EXCEPT = "try-except"
    TRY_FINALLY = "try-finally"

    def __init__(self, kind: str, **blocks: Any) -> None:
        self.kind: str = kind
        self.blocks: Dict[str, Any] = blocks


# --- End of control_flow.py ---
