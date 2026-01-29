import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union, Callable
from env.cmdp import CMDP
Point = np.ndarray  # type for state and action coordinates, np.array([int, int])

class Gridworld(CMDP):
    """
    Gridworld MDP.

    additional Attributes:
        actions: List of available actions as numpy arrays [a_right, a_up].
        grid_height: Integer for grid height.
        grid_width: Integer for grid width.
        noise: Chance of moving randomly.
    """

    def __init__(self, grid_width: int, grid_height: int, noise: float, gamma: float,
                 nu0: Optional[np.ndarray] = None, r: Optional[np.ndarray] = None,
                 constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:

        # gridworld specific attributes
        self.actions: List[Point] = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        self.grid_height: int = grid_height
        self.grid_width: int = grid_width
        self.noise = noise
        self.Psi, self.b = constraints if constraints is not None else (np.zeros((grid_width * grid_height, len(self.actions), 0)), np.array([]))
        self.r = r if r is not None else np.zeros((grid_width * grid_height, len(self.actions)))
        # general CMDP attributes
        self.n = grid_width * grid_height
        self.m = len(self.actions)
        self.k = self.Psi.shape[2]
        # print("construct transition matrix")
        self.P: np.ndarray = np.array(
            [[[self._transition_dynamics(s, a, s_next)
               for s_next in range(self.n)]
                for a in range(self.m)]
                 for s in range(self.n)])
        # print("transition matrix constructed")
        n = grid_height * grid_width
        self.gamma = gamma
        # Initial state distribution uniform over all non-constrained states
        self.nu0 = np.ones(n) / (n - len(np.nonzero(self.Psi[:,0,:])[0]))
        self.nu0[np.nonzero(self.Psi[:,0,:])[0]] = 0.0
        # Initial state distribution only at the bottom middle states
        # self.nu0 = np.zeros(n)
        # self.nu0[self.point2int((0, 4))] = 1.0
        # print("initial state distribution constructed")

    def _transition_dynamics(self, j: int, k: int, i: int) -> float:
        """
        Returns P(s_i | s_j, a_k) as a valid probability (sums to 1 across i for fixed j,k).
        j, k, i are integer indices (converted via int2point).
        """

        xi, yi = self.int2point(i)
        xj, yj = self.int2point(j)
        dx, dy = self.actions[k]

        # Intended position (may be off-grid)
        intended = (xj + dx, yj + dy)
        # If intended is off-grid, interpret "intended" as staying in place
        if not (0 <= intended[0] < self.grid_width and 0 <= intended[1] < self.grid_height):
            intended = (xj, yj)

        # Valid neighbor destinations caused by noise (adjacent cells that are inside grid)
        noise_targets = self._valid_adjacent(xj, yj)

        # Ensure intended is present in the domain we distribute probabilities over.
        # If intended is not one of the adjacent (rare if actions are weird), we include it.
        domain = list(noise_targets)  # copy
        if intended not in domain:
            domain.append(intended)

        # If domain is empty for some degenerate grid (1x1), fallback: only staying is possible
        if len(domain) == 0:
            domain = [(xj, yj)]

        # Assign probabilities
        probs: Dict[Point, float] = {p: 0.0 for p in domain}

        # Intended success mass
        probs[intended] += (1.0 - self.noise)

        # Distribute noise uniformly across the domain (including intended)
        # This yields total mass = (1-noise) + noise = 1
        per_target_noise = self.noise / len(domain)
        for p in domain:
            probs[p] += per_target_noise

        # Floating point guard: normalize so sum == 1 exactly (within machine epsilon)
        total = sum(probs.values())
        if total <= 0:
            # numeric safety: fall back to deterministic stay
            return 1.0 if (xi, yi) == (xj, yj) else 0.0

        # Normalize and return the requested entry
        # We keep high precision here; caller can round if needed.
        for p in probs:
            probs[p] = probs[p] / total

        requested = (xi, yi)
        return float(probs.get(requested, 0.0))

    # basic functionality
    def neighbouring(self, i: Point, k: Point) -> bool:
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        :param i: (x, y) int tuple.
        :param k: (x, y) int tuple.
        :return: Boolean.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def int2point(self, i: int) -> Point:
        """
        Convert a state int into the corresponding coordinate.

        :param i: State int.
        :return: (x, y) int tuple.
        """

        return np.array([i % self.grid_width, i // self.grid_width])
    
    def point2int(self, p: Point) -> int:
        """
        Convert a coordinate point (x, y) to the corresponding state integer.

        :param p: (x, y) sequence or numpy array
        :return: state int
        """
        arr = np.asarray(p)
        if arr.shape[0] != 2:
            raise ValueError("point must be length-2 (x, y)")
        x, y = int(arr[0]), int(arr[1])
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            raise ValueError(f"point {(x, y)} out of grid bounds")
        return y * self.grid_width + x

    def action2int(self, a: Point) -> int:
        """
        Convert an action such as [1,0] to an action integer.

        :param a: Action.
        :return: Corresponding integer.
        """

        return self.actions.index(a)
    
    def policy2stateocc(self, policy: np.ndarray, nu0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate state occupancy measure corresponding to policy.

        :param: Policy, shape (n,m).
        :param: Mu, initial state distribution, shape (n,).
        :return: State occupancy measure, shape (n,).
        """
        if nu0 is None:
            nu0 = self.nu0
        P_policy = np.einsum('s a t, s a -> s t', self.P, policy)
        occ = (1-self.gamma) * np.linalg.solve(np.eye(self.n) - self.gamma * P_policy.T, nu0)
        return occ / np.sum(occ)

    # basic MDP tools
    def policy2stateactionocc(self, policy: np.ndarray, nu0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate state-action occupancy measure corresponding to policy.

        :policy: Policy, shape (n,m).
        :return: State occupancy measure, shape (n,m).
        """
        if nu0 is None:
            nu0 = self.nu0
        occ = np.einsum('s, s a -> s a', self.policy2stateocc(policy, nu0), policy)
        return occ / np.sum(occ)

    def occ2policy(self, occ: np.ndarray) -> np.ndarray:

        z = np.where(np.sum(occ, axis=1, keepdims=True) > 1e-10, occ, 1.0)
        return z / np.sum(z, axis=1, keepdims=True)

    def _valid_adjacent(self, x: int, y: int):
        """Return list of valid adjacent coordinates inside grid."""
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for (nx, ny) in cand if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height]
