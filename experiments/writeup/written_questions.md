# Written questions — CS336 A5 Alignment

---

## `think_about_length_normalization` (Section 5.1, 1 point)

**Question:** What are the pros and cons of normalising each sequence by its own length vs. normalising all sequences by the same constant?

**Per-sequence normalisation** (standard GRPO, eq. 27):
- *Pro:* Every sequence contributes equally to the gradient regardless of how many tokens it generates. A long, rambling wrong answer does not dominate over a short, crisp correct one simply because it is longer.
- *Pro:* Gradient magnitude is stable across batches even when generation lengths vary widely.
- *Con:* This reweights the objective so that tokens in long sequences receive less gradient signal than tokens in short ones. It is no longer a correct estimate of the original policy gradient (∇_θ J_θ). Specifically, it biases the model toward producing shorter responses (all else equal), since short sequences have their tokens upweighted relative to long sequences.
- *Con:* If two responses are equally correct, the shorter one gets a larger per-token update, so the model is incentivised to produce short, terse answers even when more reasoning is needed.

**Constant normalisation** (Dr. GRPO / RFT, eq. 33):
- *Pro:* The gradient estimator remains a faithful (up to the group-mean baseline) estimate of ∇_θ J_θ. The expectation of the total token log-prob update is correct, not reweighted by length.
- *Pro:* Does not introduce a length bias; long-form correct reasoning is treated on equal footing with short correct answers.
- *Con:* Gradient magnitude depends heavily on how many tokens are in the batch. If generation length is variable (e.g. early in training when many responses hit the max), the effective learning rate varies.
- *Con:* Choosing the constant Z requires knowing the expected number of tokens per batch in advance; setting it wrong is equivalent to scaling the learning rate.

**When does one approach dominate?**

Per-sequence normalisation is safer when response lengths vary dramatically (e.g. some correct responses are 10 tokens and others are 200) because it prevents the long responses from dominating just by being long. Constant normalisation is preferable when the training objective is explicitly to estimate the expected reward accurately and when lengths are roughly constant. For GSM8K at 512-token max with an early-stopping `</answer>` tag, lengths vary substantially (10–400 tokens), so per-sequence normalisation is less biased in practice.

---

## `think_about_rft` (Section 5.2, 2 points)

**Question:** Compare the RFT objective and the on-policy Dr. GRPO objective for binary reward. Do they have the same expectation? Which has lower variance?

**RFT objective** (eq. 35):
$$J_\theta^{\text{RFT}} = \frac{1}{Z}\sum_x\sum_{j=1}^G \mathbf{1}\{r(y^{(j)}|x)=1\}\log\pi_\theta(y^{(j)}|x).$$

**Dr. GRPO objective** (eq. 36, constant normalisation, no std):
$$J_\theta^{\text{Dr.GRPO}} = \frac{1}{Z}\sum_x\sum_{j=1}^G (r(y^{(j)}|x)-\mu)\log\pi_\theta(y^{(j)}|x),$$
where $\mu = \frac{1}{G}\sum_j r(y^{(j)}|x)$ is the group mean.

**Same expectation?**

Taking expectations over the sampling distribution (group size G → ∞, so μ → η(x) = E_{π_θ}[r(y|x)]):

- Dr. GRPO gradient expectation (eq. 24): $\frac{G-1}{G}E_{y\sim\pi_\theta}[r(y|x)\nabla_\theta\log\pi_\theta(y|x)]$, which converges to the true gradient.
- RFT gradient expectation: $E_{y\sim\pi_\theta}[\mathbf{1}\{r=1\}\nabla_\theta\log\pi_\theta(y|x)] = E_{y\sim\pi_\theta}[r(y|x)\nabla_\theta\log\pi_\theta(y|x)]$ (same for binary reward).

So both have the **same expectation** (up to the G/(G-1) scaling in Dr. GRPO).

**Which has lower variance?**

RFT has weights $r(y)\in\{0,1\}$; Dr. GRPO has weights $r(y)-\mu\in\{-\mu, 1-\mu\}$.

From part (b) of `baseline_calcs`, subtracting a baseline equal to the group mean reduces variance when the group mean $\mu$ (equivalently $p$ in the scalar case) satisfies $p < 2/3$. For most GSM8K prompts early in training the model accuracy is low ($\mu \approx 0.01$–$0.2$), so $\mu < 2/3$ and **Dr. GRPO has lower variance than RFT** in these regimes.

However, RFT has a structural advantage: when the entire group is incorrect ($\mu=0$), Dr. GRPO still computes zero advantages and skips the update (same as RFT), but when the entire group is correct ($\mu=1$), RFT still upweights all responses (gradient signal present), while Dr. GRPO produces zero advantages (no update). In late training when the model is strong and most rollouts are correct, RFT continues to provide signal while Dr. GRPO goes silent.

**Preference:** RFT is simpler to implement correctly and is preferred when the model is strong (high $\mu$). Dr. GRPO is preferred early in training when rewards are sparse, because the baseline reduces the variance of the gradient estimate without wasting compute on zero-advantage batches.

---

## `derive_difficulty_reweightings` (Section 5.3, 6 points)

We want to find the prompt-reweighting function $w(x)$ such that the GRPO variant's gradient equals $\nabla_\theta J_{\theta,w} = \nabla_\theta E_{x\sim\rho}[w(x,\text{sg}(\pi_\theta)) E_{y\sim\pi_\theta}[r(y|x)]]$, in the large-G limit (group mean → true expected reward $\eta(x) = E_\pi[r(y|x)]$).

### (a) Dr. GRPO

The Dr. GRPO estimator (eq. 41) in the large-G limit becomes:
$$E_{x\sim\rho}\bigl[E_{y\sim\pi_\theta}[(r(y|x)-\eta(x))\nabla_\theta\log\pi_\theta(y|x)]\bigr].$$

Apply the policy gradient identity $E_y[f(y)\nabla_\theta\log\pi_\theta(y|x)] = \nabla_\theta E_y[f(y)]$ only when $f$ does not depend on $\theta$:

Using $\nabla_\theta E_{y\sim\pi_\theta}[r(y|x)] = E_{y\sim\pi_\theta}[r(y|x)\nabla_\theta\log\pi_\theta(y|x)]$ and the identity $E_{y\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(y|x)] = 0$:

$$E_{x\sim\rho}E_{y\sim\pi_\theta}[(r-\eta)\nabla_\theta\log\pi_\theta] = E_{x\sim\rho}\nabla_\theta E_{y\sim\pi_\theta}[r(y|x)] = \nabla_\theta E_{x\sim\rho}[\eta(x)].$$

This matches $\nabla_\theta J_{\theta,w}$ with $w(x) = 1$ (i.e. no reweighting). **Dr. GRPO recovers the standard policy gradient without any prompt reweighting.**

### (b) GRPO (std normalisation)

The GRPO estimator (eq. 42) in the large-G limit:
$$E_{x\sim\rho}\left[\frac{1}{\text{std}_x} E_{y\sim\pi_\theta}\left[(r(y|x)-\eta(x))\nabla_\theta\log\pi_\theta(y|x)\right]\right].$$

Since std$(x) = \sqrt{E_\pi[(r-\eta)^2]}$ and we treat it as a stop-gradient:

$$= E_{x\sim\rho}\left[\frac{1}{\text{std}(x)}\nabla_\theta \eta(x)\right] = \nabla_\theta E_{x\sim\rho}\left[\frac{1}{\text{std}(x)}\eta(x)\right].$$

This matches $J_{\theta,w}$ with
$$\boxed{w(x) = \frac{1}{\text{std}(x)} = \frac{1}{\sqrt{\eta(x)(1-\eta(x))}}}$$
(the last equality uses the fact that for binary rewards, $\text{std}^2 = \eta(1-\eta)$).

**Interpretation:** GRPO upweights prompts with low variance in their reward distribution — either very easy ($\eta \approx 1$) or very hard ($\eta \approx 0$) prompts get large weight; medium-difficulty ($\eta \approx 0.5$) prompts get the smallest weight. This is sometimes called "inverse-variance weighting."

### (c) MaxRL

The MaxRL estimator (eq. 43) in the large-G limit:
$$E_{x\sim\rho}\left[\frac{1}{\mu_x}E_{y\sim\pi_\theta}\left[(r(y|x)-\mu_x)\nabla_\theta\log\pi_\theta(y|x)\right]\right].$$

$$= E_{x\sim\rho}\left[\frac{1}{\eta(x)}\nabla_\theta\eta(x)\right] = \nabla_\theta E_{x\sim\rho}\left[\frac{\eta(x)}{\eta(x)}\right] ... $$

Hmm, let me be more careful. Treating $\eta(x)$ as a stop-gradient:

$$= E_{x\sim\rho}\left[\frac{1}{\eta(x)}\nabla_\theta\eta(x)\right] = E_{x\sim\rho}\left[\nabla_\theta\log\eta(x)\right] = \nabla_\theta E_{x\sim\rho}[\log\eta(x)].$$

So this matches $J_{\theta,w}$ with $w(x) = 1/\eta(x)$, corresponding to:

$$J_{\theta,w} = E_{x\sim\rho}\left[\frac{1}{\eta(x)}\cdot\eta(x)\right] = E_{x\sim\rho}[1] \text{ (constant)}.$$

Wait, that's not right — $\nabla_\theta[\log\eta(x)] = \nabla_\theta\eta(x)/\eta(x)$, so $\nabla_\theta E_x[\log\eta(x)] = E_x[\nabla_\theta\eta(x)/\eta(x)]$. Matching the form $\nabla_\theta J_{\theta,w} = \nabla_\theta E_x[w(x)\,\eta(x)]$ requires $w(x)\,\nabla_\theta\eta(x) = \nabla_\theta\eta(x)/\eta(x)$, giving:

$$\boxed{w(x) = \frac{1}{\eta(x)}.}$$

**Interpretation:** MaxRL upweights prompts that are currently *hard* for the model (low $\eta(x)$), giving them more influence in training. This is the opposite of what GRPO (std normalisation) does: GRPO focuses on easy and hard prompts equally (inverse std), while MaxRL concentrates almost entirely on prompts the model consistently fails. This can accelerate learning on hard examples but may be unstable when $\eta(x)\approx 0$.

---

## `think_about_advantage_normalization` (Section 5.3, 2 points)

**Question:** Compare normalising by group std, group mean, or no normalisation. Pros, cons, examples where one seems better.

**No normalisation (Dr. GRPO):**
- *Pro:* Unbiased estimator of ∇_θ J_θ; no distortion of the objective.
- *Con:* Gradient magnitude varies with the reward scale. If rewards are sparse (most batches have mean~0), gradients are small and learning is slow.
- *Best when:* Reward scale is stable and known; you want a theoretically correct gradient.

**Std normalisation (GRPO):**
- *Pro:* Gradient updates are approximately unit-norm across groups, stabilising training. Prevents large batches from dominating.
- *Con:* Dividing by the std distorts the gradient (no longer estimates ∇_θ J_θ). Reweights the effective prompt distribution, upweighting extreme-difficulty prompts.
- *Con:* Undefined when all rewards in a group are equal (std=0), requiring an epsilon stabiliser.
- *Best when:* Reward magnitudes are heterogeneous across prompts (e.g. rewards ranging from 0 to 10 across different problems); standardisation makes updates comparable.

**Mean normalisation (MaxRL):**
- *Pro:* Focuses gradient signal on hard prompts (low mean reward), which may be most informative.
- *Pro:* Naturally performs curriculum weighting without explicit curriculum design.
- *Con:* Divides by a very small number for hard prompts early in training, potentially causing numerical instability or very large updates.
- *Con:* Late in training when the model is accurate, mean→1 and the update → Dr. GRPO. But early on, for near-zero mean prompts, gradient magnitude explodes.
- *Best when:* Reward is binary and you want to emphasise still-failing prompts; requires careful epsilon clipping to avoid instability.

**Summary:** For stable early-stage training on a hard task like GSM8K, std normalisation is typically most robust. If the model already gets some prompts right (>30% accuracy), mean normalisation may accelerate convergence by focusing on the remaining hard prompts. No normalisation is preferable when you want the update to match the true policy gradient and reward variances are similar across prompts.
