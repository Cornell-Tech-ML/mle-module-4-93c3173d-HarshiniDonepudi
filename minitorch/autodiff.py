from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    ls1 = [i for i in vals]
    ls2 = [i for i in vals]
    ls1[arg] += epsilon
    ls2[arg] -= epsilon
    return ((f(*ls1)) - (f(*ls2))) / (2.0 * epsilon)
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    stack = []
    visited = set()

    def dfs(curr_var: Variable) -> None:
        if curr_var.unique_id in visited or curr_var.is_constant():
            return
        for i in curr_var.parents:
            dfs(i)
        visited.add(curr_var.unique_id)
        stack.append(curr_var)

    dfs(variable)
    ordering = []
    while stack:
        ordering.append(stack.pop())

    return ordering
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sort = topological_sort(variable)
    unique_ids = [var.unique_id for var in sort]
    derivative = dict.fromkeys(unique_ids, 0)
    derivative[variable.unique_id] = deriv
    for i in sort:
        if i.is_leaf():
            i.accumulate_derivative(derivative[i.unique_id])
        else:
            for j, k in i.chain_rule(derivative.get(i.unique_id, 0)):
                derivative[j.unique_id] = derivative.get(j.unique_id, 0) + k
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
