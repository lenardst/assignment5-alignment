# Problem `baseline_calcs` — Variance of the policy gradient estimator

**Setup.** Policy π_θ over binary actions A ∈ {0,1} with
π_θ(A=1) = p = σ(θ), where σ(θ) = 1/(1+e^{-θ}).
Reward r(A) = 𝟏{A=1}.  Samples A_i ~iid π_θ.

---

## (a) Variance of the unadjusted estimator

The estimator is
$$\hat g = \frac{1}{n}\sum_{i=1}^n r(A_i)\,\nabla_\theta \log\pi_\theta(A_i).$$

**Step 1 — compute the gradient.**

$$\nabla_\theta \log\pi_\theta(A) = \begin{cases}
  \nabla_\theta \log\sigma(\theta) = 1-p & \text{if } A=1 \\
  \nabla_\theta \log(1-\sigma(\theta)) = -p & \text{if } A=0
\end{cases}$$

**Step 2 — write out Z_i = r(A_i)∇_θ log π_θ(A_i).**

$$Z_i = \begin{cases}
  (1)(1-p) = 1-p & \text{with prob } p \\
  (0)(-p) = 0 & \text{with prob } 1-p
\end{cases}$$

**Step 3 — compute moments.**

$$E[Z_i] = p(1-p),\qquad
E[Z_i^2] = p(1-p)^2.$$

**Step 4 — variance.**

$$\operatorname{Var}[Z_i]
= E[Z_i^2] - (E[Z_i])^2
= p(1-p)^2 - p^2(1-p)^2
= (1-p)^2\bigl[p - p^2\bigr]
= p(1-p)^3.$$

Since the A_i are i.i.d., the variance of the sample mean is:

$$\boxed{\operatorname{Var}[\hat g] = \frac{p(1-p)^3}{n}.}$$

---

## (b) Variance of the baseline-adjusted estimator

The estimator is
$$\hat g_b = \frac{1}{n}\sum_{i=1}^n (r(A_i)-b)\,\nabla_\theta\log\pi_\theta(A_i).$$

**Expectation is preserved.** Let W_i = (r(A_i) − b) ∇_θ log π_θ(A_i).

$$E[W_i] = E[r(A_i)\nabla_\theta\log\pi_\theta(A_i)]
         - b\underbrace{E[\nabla_\theta\log\pi_\theta(A_i)]}_{=\,0}
         = p(1-p).$$

**Per-sample values of W_i:**

$$W_i = \begin{cases}
  (1-b)(1-p) & \text{with prob }p \\
  (0-b)(-p) = bp & \text{with prob }1-p
\end{cases}$$

**Second moment:**

$$E[W_i^2] = p(1-b)^2(1-p)^2 + (1-p)b^2p^2
           = p(1-p)\bigl[(1-b)^2(1-p) + b^2 p\bigr].$$

**Variance:**

$$\operatorname{Var}[W_i]
= E[W_i^2] - (E[W_i])^2
= p(1-p)\bigl[(1-b)^2(1-p)+b^2p\bigr] - p^2(1-p)^2.$$

Factor out $p(1-p)$:

$$= p(1-p)\bigl[(1-b)^2(1-p)+b^2p - p(1-p)\bigr].$$

Expand the bracket:
$$
(1-b)^2(1-p)+b^2p - p(1-p)
= (1-2b+b^2)(1-p)+b^2p - p + p^2.
$$
$$
= 1-p - 2b(1-p) + b^2(1-p) + b^2p - p + p^2
= 1 - 2p + p^2 - 2b(1-p) + b^2
= (1-p)^2 - 2b(1-p) + b^2
= (1-p-b)^2.
$$

Therefore:

$$\boxed{\operatorname{Var}[\hat g_b] = \frac{p(1-p)(1-p-b)^2}{n}.}$$

**Discussion.** Setting $b=0$ recovers $p(1-p)^3/n$. Since $p(1-p)(1-p-b)^2 = p(1-p)(1-p-b)^2$, the baseline reduces variance iff $(1-p-b)^2 < (1-p)^2$, i.e. $|1-p-b| < |1-p| = 1-p$ (since $p\in(0,1)$), which holds iff $b\in(0, 2(1-p))$. Minimising over $b$ gives $b^* = 1-p$ with Var = 0 — the optimal baseline is the reward the model would receive if it were correct, discounted by $1-p$. In practice, using the mean reward $b=p$ is a common, easily computed choice.

---

## (c) Substituting the population-mean baseline b = p

$$\operatorname{Var}[\hat g_{b=p}]
= \frac{p(1-p)(1-p-p)^2}{n}
= \frac{p(1-p)(1-2p)^2}{n}.$$

**Comparison with the unadjusted estimator:**

$$\frac{\operatorname{Var}[\hat g_{b=p}]}{\operatorname{Var}[\hat g]}
= \frac{p(1-p)(1-2p)^2/n}{p(1-p)^3/n}
= \frac{(1-2p)^2}{(1-p)^2}
= \left(\frac{1-2p}{1-p}\right)^2.$$

The ratio equals 1 at $p = 2/3$ (i.e. $(1-2(2/3))/(1-2/3) = (-1/3)/(1/3) = -1$, so ratio $= 1$), is less than 1 for $p\in(0,2/3)$, and greater than 1 for $p\in(2/3,1)$.

**Conclusion: the population-mean baseline is NOT always lower variance.**

- For $p \in (0, 2/3)$: the baseline reduces variance (the action is hard or moderately likely, so centering the reward helps).
- For $p = 2/3$: the variance is identical.
- For $p \in (2/3, 1)$: **the baseline increases variance.** When the policy is very likely to take action 1, the rare $A=0$ samples receive a large negative adjustment ($-p \cdot (-p) = p^2$ in magnitude), inflating variance more than the adjustment saves.

Intuitively, the population-mean baseline $b = p$ is only a good choice when the policy is uncertain; when $p > 2/3$ the optimal baseline $b^* = 1-p < 1/3$ is much smaller than $p$.
