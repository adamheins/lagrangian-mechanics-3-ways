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


def symbolic_dynamics(time):
    t = sym.symbols("t")
    q = sym.Matrix(
        [sym.Function("q1")(t), sym.Function("q2")(t), sym.Function("q3")(t)]
    )
    dq = q.diff(t)

    x1 = q[0] + LX + 0.5 * L1 * sym.cos(q[1])
    y1 = LY + 0.5 * L1 * sym.sin(q[1])
    x2 = q[0] + LX + L1 * sym.cos(q[1]) + 0.5 * L2 * sym.cos(q[1] + q[2])
    y2 = LY + L1 * sym.sin(q[1]) + 0.5 * L2 * sym.sin(q[1] + q[2])

    dx1 = x1.diff(t)
    dy1 = y1.diff(t)
    dx2 = x2.diff(t)
    dy2 = y2.diff(t)

    # Potential energy
    Pb = 0
    P1 = M1 * G * y1
    P2 = M2 * G * y2
    P = Pb + P1 + P2

    # Kinetic energy
    Kb = 0.5 * Mb * dq[0] ** 2
    K1 = 0.5 * M1 * (dx1 ** 2 + dy1 ** 2) + 0.5 * I1 * dq[1] ** 2
    K2 = 0.5 * M2 * (dx2 ** 2 + dy2 ** 2) + 0.5 * I2 * (dq[1] + dq[2]) ** 2
    K = Kb + K1 + K2

    # Lagrangian
    L = K - P

    # Generalized forces
    tau = L.diff(dq).diff(t) - L.diff(q)

    return tau.subs({q[0]: sym.sin(t), q[1]: t, q[2]: t * t, t: time}).doit()


def configuration(t, np=np):
    """ Define joint configuration as function of time. """
    q = np.array([np.sin(t), t, t * t])
    return q


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
        dMdq[:, :, 0] = dMdθ1
        dMdq[:, :, 1] = dMdθ2
        dMdq[:, :, 2] = dMdθ3

        # Γ = dMdq - 0.5 * dMdq.T

        # Construct matrix of Christoffel symbols
        # TODO: note transpose on dMdq: difference b/t math order and numpy
        # order
        Γ = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Γ[i, j, k] = 0.5*(dMdq[k, j, i] + dMdq[i, k, j] - dMdq[i, j, k])  # mine
                    # Γ[i, j, k] = dMdq[k, j, i] - 0.5*dMdq[i, j, k]  # mine
                    Γ[i, j, k] = 0.5*(dMdq.T[k, j, i] + dMdq.T[k, i, j] - dMdq.T[i, j, k])  # Spong
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
    # tau_func = auto_diff_dynamics()
    #
    # q_func = partial(configuration, np=jnp)
    # dq_func = jax.jit(jax.jacfwd(partial(configuration, np=jnp)))
    # ddq_func = jax.jit(jax.jacfwd(dq_func))
    #
    # t = 1.0
    # q = q_func(t)
    # dq = dq_func(t)
    # ddq = ddq_func(t)
    #
    # dMdq_func = jax.jacfwd(partial(calc_mass_matrix, np=jnp))
    # M = calc_mass_matrix(q)
    # g = calc_gravity_vector(q)
    # dMdq = dMdq_func(q)
    q = np.array([0, 0.5, 0.5])
    dq = np.array([0.5, 1, 0])
    ddq = np.array([0.5, 1, 0])

    print(ManualDynamics.tau(q, dq, ddq))
    print(AutoDiffDynamics.tau(q, dq, ddq))

    # print(np.array(symbolic_dynamics(t)).astype(np.float64).flatten())
    # print(manual_dynamics_mat(q, dq, ddq))

    # IPython.embed()


if __name__ == "__main__":
    main()
