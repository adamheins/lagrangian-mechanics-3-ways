import numpy as np
import jax
import jax.numpy as jnp
import sympy as sym
import IPython


# link masses
M1 = 3
M2 = 2
M3 = 1

# link lengths
LX = 0
LY = 0
L2 = 1
L3 = 0.5

# moments of inertia (base link doesn't rotate so don't need it)
I2 = M2 * L2 ** 2 / 12
I3 = M3 * L3 ** 2 / 12

# inertia matrices
G1 = np.diag(np.array([M1, M1, 0]))
G2 = np.diag(np.array([M2, M2, I2]))
G3 = np.diag(np.array([M3, M3, I3]))

# gravity
G = 9.8


class SymbolicDynamics:
    # joint configuration and velocity are functions of time
    t = sym.symbols("t")
    q = sym.Array(
        [sym.Function("x1")(t), sym.Function("θ2")(t), sym.Function("θ3")(t)]
    )
    dq = q.diff(t)

    # link poses
    P1 = sym.Matrix([q[0], 0, 0])
    P2 = P1 + sym.Matrix([
        LX + 0.5 * L2 * sym.cos(q[1]),
        LY + 0.5 * L2 * sym.sin(q[1]),
        q[1],
    ])
    P3 = P2 + sym.Matrix([
        0.5 * L2 * sym.cos(q[1]) + 0.5 * L3 * sym.cos(q[1] + q[2]),
        0.5 * L2 * sym.sin(q[1]) + 0.5 * L3 * sym.sin(q[1] + q[2]),
        q[2],
    ])

    # link Jacobians
    J1 = P1.jacobian(q)
    J2 = P2.jacobian(q)
    J3 = P3.jacobian(q)

    # mass matrix
    M = J1.transpose() * G1 * J1 + J2.transpose() * G2 * J2 + J3.transpose() * G3 * J3

    # Christoffel symbols and Coriolis matrix
    dMdq = M.diff(q)
    Γ = sym.permutedims(dMdq, (2, 1, 0)) - 0.5 * dMdq
    C = sym.tensorcontraction(sym.tensorproduct(dq, Γ), (0, 2)).tomatrix()

    # gravity vector
    V = G * (M1 * P1[1] + M2 * P2[1] + M3 * P3[1])
    g = V.diff(q)

    # compile functions to numerical code
    mass_matrix = sym.lambdify([q], M)
    coriolis_matrix = sym.lambdify([q, dq], C)
    gravity_vector = sym.lambdify([q], g)

    @classmethod
    def tau(cls, q, dq, ddq):
        M = cls.mass_matrix(q)
        C = cls.coriolis_matrix(q, dq)
        g = cls.gravity_vector(q)

        return M @ ddq + C @ dq + g


class ManualDynamics:
    @classmethod
    def mass_matrix(cls, q):
        x1, θ2, θ3 = q
        θ23 = θ2 + θ3

        m11 = M1 + M2 + M3
        m12 = -(0.5 * M2 + M3) * L2 * np.sin(θ2) - 0.5 * M3 * L3 * np.sin(θ23)
        m13 = -0.5 * M3 * L3 * np.sin(θ23)

        m22 = (
            (0.25 * M2 + M3) * L2 ** 2
            + 0.25 * M3 * L3 ** 2
            + M3 * L2 * L3 * np.cos(θ3)
            + I2
            + I3
        )
        m23 = 0.5 * M3 * L3 * (0.5 * L3 + L2 * np.cos(θ3)) + I3

        m33 = 0.25 * M3 * L3 ** 2 + I3

        M = np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])
        return M

    @classmethod
    def christoffel_matrix(cls, q):
        x1, θ2, θ3 = q
        θ23 = θ2 + θ3

        # Partial derivatives of mass matrix
        dMdθ1 = np.zeros((3, 3))

        dMdθ2_12 = (
            -0.5 * M2 * L2 * np.cos(θ2) - M3 * L2 * np.cos(θ2) - 0.5 * M3 * L3 * np.cos(θ23)
        )
        dMdθ2_13 = -0.5 * M3 * L3 * np.cos(θ23)
        dMdθ2 = np.array([[0, dMdθ2_12, dMdθ2_13], [dMdθ2_12, 0, 0], [dMdθ2_13, 0, 0]])

        dMdθ3_12 = -0.5 * M3 * L3 * np.cos(θ23)
        dMdθ3_13 = -0.5 * M3 * L3 * np.cos(θ23)
        dMdθ3_22 = -M3 * L2 * L3 * np.sin(θ3)
        dMdθ3_23 = -0.5 * M3 * L2 * L3 * np.sin(θ3)
        dMdθ3 = np.array([
            [0, dMdθ3_12, dMdθ3_13],
            [dMdθ3_12, dMdθ3_22, dMdθ3_23],
            [dMdθ3_13, dMdθ3_23, 0],
        ])

        dMdq = np.zeros((3, 3, 3))
        dMdq[0, :, :] = dMdθ1
        dMdq[1, :, :] = dMdθ2
        dMdq[2, :, :] = dMdθ3

        Γ = dMdq.T - 0.5 * dMdq

        # The above is equivalent to but more efficient than this for-loop
        # construction:
        # Γ = np.zeros((3, 3, 3))
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             Γ[i, j, k] = 0.5*(dMdq[k, j, i] + dMdq[k, i, j] - dMdq[i, j, k])
        return Γ

    @classmethod
    def coriolis_matrix(cls, q, dq):
        Γ = cls.christoffel_matrix(q)
        return dq.T @ Γ

    @classmethod
    def gravity_vector(cls, q):
        x1, θ2, θ3 = q
        θ23 = θ2 + θ3
        return np.array([
            0,
            (0.5 * M2 + M3) * G * L2 * np.cos(θ2) + 0.5 * M3 * L3 * G * np.cos(θ23),
            0.5 * M3 * L3 * G * np.cos(θ23),
        ])

    @classmethod
    def tau(cls, q, dq, ddq):
        M = cls.mass_matrix(q)
        C = cls.coriolis_matrix(q, dq)
        g = cls.gravity_vector(q)

        return M @ ddq + C @ dq + g


class AutoDiffDynamics:
    @classmethod
    def link1_pose(cls, q):
        return jnp.array([q[0], 0, 0])

    @classmethod
    def link2_pose(cls, q):
        P1 = cls.link1_pose(q)
        return P1 + jnp.array([
            LX + 0.5 * L2 * jnp.cos(q[1]),
            LY + 0.5 * L2 * jnp.sin(q[1]), q[1]]
        )

    @classmethod
    def link3_pose(cls, q):
        P2 = cls.link2_pose(q)
        return P2 + jnp.array([
            0.5 * L2 * jnp.cos(q[1]) + 0.5 * L3 * jnp.cos(q[1] + q[2]),
            0.5 * L2 * jnp.sin(q[1]) + 0.5 * L3 * jnp.sin(q[1] + q[2]),
            q[2],
        ])

    @classmethod
    def mass_matrix(cls, q):
        # Jacobians
        J1 = jax.jacfwd(cls.link1_pose)(q)
        J2 = jax.jacfwd(cls.link2_pose)(q)
        J3 = jax.jacfwd(cls.link3_pose)(q)

        return J1.T @ G1 @ J1 + J2.T @ G2 @ J2 + J3.T @ G3 @ J3

    @classmethod
    def christoffel_matrix(cls, q):
        # Here dMdq is transposed with respect to the dMdq's in the manual and
        # symbolic implementations. Thus, the expression for Γ is also
        # transposed, giving the same end result.
        dMdq = jax.jacfwd(cls.mass_matrix)(q)
        Γ = dMdq - 0.5 * dMdq.T
        return Γ

    @classmethod
    def coriolis_matrix(cls, q, dq):
        Γ = cls.christoffel_matrix(q)
        return dq.T @ Γ

    @classmethod
    def potential_energy(cls, q):
        P1 = cls.link1_pose(q)
        P2 = cls.link2_pose(q)
        P3 = cls.link3_pose(q)
        return G * (M1 * P1[1] + M2 * P2[1] + M3 * P3[1])

    @classmethod
    def gravity_vector(cls, q):
        return jax.jacfwd(cls.potential_energy)(q)

    @classmethod
    def tau(cls, q, dq, ddq):
        M = cls.mass_matrix(q)
        C = cls.coriolis_matrix(q, dq)
        g = cls.gravity_vector(q)

        return M @ ddq + C @ dq + g


def main():
    q = np.array([0, 0.5, 0.5])
    dq = np.array([0.5, 1, 0])
    ddq = np.array([0.5, 1, 0])

    print(ManualDynamics.tau(q, dq, ddq))
    print(AutoDiffDynamics.tau(q, dq, ddq))
    print(SymbolicDynamics.tau(q, dq, ddq))


if __name__ == "__main__":
    main()
