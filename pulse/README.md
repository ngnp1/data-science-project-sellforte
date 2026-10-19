# Pulse detection

Spend-only detector for **channel pulses**. Entry point: `pulse.detect:detect` (layer C);
`detect_b` and `detect_a` use layers B and A
(`python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data`).

**Notation.** For one market × channel, $s_t$ is spend on day $t = 1..n$; $\pi_t \in \{0,1\}$
marks days with data. Usual level $\ell = \operatorname{median}\{s_t : s_t > 0\}$;
cutoff $\kappa = \max(10^{-6},\ \rho\,\ell)$ with $\rho = 0.15$.

## 1. On/off probability $\gamma_t = P(\text{off on day } t)$

**Layer A** (`onoff.p_off`), per day:

$$q_t = \sigma\!\left(\frac{\kappa - s_t}{w\,\ell}\right),\qquad \sigma(z) = \frac{1}{1+e^{-z}},\quad w = 0.03$$

so $q_t = \tfrac12 \iff s_t = \kappa$. Missing days: $q_t = 1$. A channel with no positive spend has $q \equiv 0$.

**Layer B** (`onoff.p_off_hmm`): two-state HMM, $z_t \in \{\text{on}, \text{off}\}$,
$P(z_t = z_{t-1}) = 0.95$, $P(z_1) = (\tfrac12, \tfrac12)$. Layer A is the observation, wrong with
probability $\eta = 0.05$:

$$e_t(\text{off}) = (1-\eta)\,q_t + \eta\,(1-q_t),\qquad e_t(\text{on}) = \eta\,q_t + (1-\eta)(1-q_t),$$

and $e_t \equiv 1$ on missing days. Scaled forward-backward:
$\alpha_t \propto (\alpha_{t-1}T)\odot e_t$, $\beta_t \propto T\,(e_{t+1}\odot\beta_{t+1})$,
$\gamma_t \propto \alpha_t \odot \beta_t$.

**Layer C** (`hsmm.p_off_hsmm`, default): explicit-duration (semi-Markov) version of B with the same
$e_t$. The series is a sequence of alternating spells $(a, b]$. Spell-length priors:

$$p_\text{on}(d) = \tfrac1n,\qquad p_\text{off}(d) \propto \frac1d \exp\!\left(-\frac12\left(\frac{\ln(d/14)}{0.4}\right)^{2}\right)$$

(lognormal, median 14 days). The first and last spells are cut by the data edges and use the survival
$S_k(d) = \sum_{j \ge d} p_k(j)$ instead of $p_k(d)$. Log weight of a spell of state $k$:

$$W(a,b,k) = \log\text{start}(a,k) + \log p_k(b-a) + \sum_{t=a+1}^{b} \log e_t(k),$$

with $\text{start}(0,k) = \tfrac12$ and $\text{start}(a,k) = F(a, \bar k)$ for the other state $\bar k$.
Forward $F(t,k) = \operatorname{LSE}_a W(a,t,k)$, backward
$B(t,k) = \operatorname{LSE}_{b>t}\big[\log p_{\bar k}(b-t) + \textstyle\sum_{u=t+1}^{b} \log e_u(\bar k) + B(b,\bar k)\big]$,
evidence $Z$ = LSE over spells ending at $n$. Day posterior:

$$\gamma_t = \sum_{(a,b]\ni t,\ k=\text{off}} \exp\big(W(a,b,k) + B(b,k) - Z\big).$$

**Off run.** Maximal run of $\gamma_t \ge \tfrac12$ (layers B/C: missing days trimmed from its ends), kept only if
$\operatorname{median}\{s_t : t \in \text{run},\ \pi_t = 1\} \le \kappa$. A run touching day 1 or day $n$ is
*censored*. $L$ = run length in days.

## 2. Pulse trains (`pulse.find_pulse_trains`)

Uncensored runs with $L \ge 7$, sorted by $L$, are cut into the longest stretches $G$ satisfying

$$\max_G L \le (1+\lambda)\operatorname{med}_G L \quad\text{and}\quad (1+\lambda)\min_G L \ge \operatorname{med}_G L,\qquad \lambda = 0.5.$$

Within $G$, in time order, a new train starts when the gap $a_{i+1} - b_i - 1 > 6\min(L_i, L_{i+1})$.
A train needs $\ge 2$ pauses. A censored run joins a train if $L \le (1+\lambda)\operatorname{med}L$ and the
gap rule holds. Confidence (`pulse.pulse_probability`):

$$P = \overline{\gamma}_{\text{pauses}} \cdot \Big(1 - \min\big(1, \tfrac{\operatorname{sd}(L)}{\operatorname{mean}(L)}\big)\Big) \cdot \min\Big(1, \frac{\sum L}{\sum_{\text{runs} \ge 7 \text{ in window}} |\text{run} \cap \text{window}|}\Big).$$

It is a heuristic score in $[0,1]$, not a calibrated probability.

## 3. Scores

Dev split (45 scenarios), channel pulse only; layers A, B and C score the same:

| metric | value |
|---|---|
| channel pulse P / R / F1 | 1.000 / 1.000 / 1.000 (10 found / 0 false / 0 missed) |
| individual pauses | 27 / 27 |
| boundary error (median / p90) | 0 / 0 days |
| false alarms on no-event scenarios | 0.000 per country-year |

The same pulse code on `pulse-extended` scores 1.000 (13 / 0 / 0) on the sealed test split.
