import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax.scipy import linalg

Lx = 10.
Ly = 10.
theta = jnp.pi/3
jacobian = jnp.array([[Lx,jnp.cos(theta) * Ly],[0., jnp.sin(theta) * Ly]])  # (d,d) [A]

# inv_jacobian = jnp.array([[1/Lx, -jnp.cos(theta)/(jnp.sin(theta) * Lx)], [0., 1/(jnp.sin(theta) * Ly)]])  # (d,d) [A]
# print(linalg.inv(jacobian), inv_jacobian)


N = 10000
d = 2
key = jax.random.PRNGKey(0)  # You can use any PRNGKey
lower_bound = jnp.array([0.,]*d)
upper_bound = jnp.array([1.,]*d)

S = jax.random.uniform(key, (N,d), minval=lower_bound, maxval=upper_bound)
plt.scatter(*S.T)
plt.title('Scaled unit coordinates')
plt.show()

# R = jnp.array([(jacobian @ v).tolist() for v in S])
R = jnp.einsum('ij,...j->...i', jacobian, S)
plt.scatter(*R.T)
# plt.xlim(0, Lx)
# plt.ylim(0, Ly)
plt.title('Physical space')
plt.show()