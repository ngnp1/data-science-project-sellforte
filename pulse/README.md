# Informative-period detection

Spend-only detector for the six period types. Entry point: `pulse.detect:detect`
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

**Layer C** (`hsmm.p_off_hsmm`, used by every detector): explicit-duration (semi-Markov) version of B
with the same $e_t$. The series is a sequence of alternating spells $(a, b]$. Spell-length priors:

$$p_\text{on}(d) = \tfrac1n,\qquad p_\text{off}(d) \propto \frac1d \exp\!\left(-\frac12\left(\frac{\ln(d/\mu)}{\varsigma}\right)^{2}\right)$$

(lognormal), with $(\mu, \varsigma) = (14, 0.4)$ for pulses and $(42, 0.8)$ for the other types.
The first and last spells are cut by the data edges and use the survival
$S_k(d) = \sum_{j \ge d} p_k(j)$ instead of $p_k(d)$. Log weight of a spell of state $k$:

$$W(a,b,k) = \log\text{start}(a,k) + \log p_k(b-a) + \sum_{t=a+1}^{b} \log e_t(k),$$

with $\text{start}(0,k) = \tfrac12$ and $\text{start}(a,k) = F(a, \bar k)$ for the other state $\bar k$.
Forward $F(t,k) = \operatorname{LSE}_a W(a,t,k)$, backward
$B(t,k) = \operatorname{LSE}_{b>t}\big[\log p_{\bar k}(b-t) + \textstyle\sum_{u=t+1}^{b} \log e_u(\bar k) + B(b,\bar k)\big]$,
evidence $Z$ = LSE over spells ending at $n$. Day posterior:

$$\gamma_t = \sum_{(a,b]\ni t,\ k=\text{off}} \exp\big(W(a,b,k) + B(b,k) - Z\big).$$

**Off spell.** Maximal run of $\gamma_t \ge \tfrac12$, missing days trimmed from its ends, kept only if
$\operatorname{median}\{s_t : t \in \text{run},\ \pi_t = 1\} \le \kappa$ (the 15 % definition). A run touching
day 1 or day $n$ is *censored*. $L$ = run length in days.

## 2. Period types

$M_{t,c} = 1$ if day $t$ is in an off spell of channel $c$; $N$ = active channels in the market.

**Channel pulse** (`pulse.find_pulse_trains`). Uncensored runs with $L \ge 7$, sorted by $L$, are cut into
the longest stretches $G$ satisfying

$$\max_G L \le (1+\lambda)\operatorname{med}_G L \quad\text{and}\quad (1+\lambda)\min_G L \ge \operatorname{med}_G L,\qquad \lambda = 0.5.$$

Within $G$, in time order, a new train starts when the gap $a_{i+1} - b_i - 1 > 6\min(L_i, L_{i+1})$.
A train needs $\ge 2$ pauses. A censored run joins a train if $L \le (1+\lambda)\operatorname{med}L$ and the gap rule
holds. Confidence:

$$P = \overline{\gamma}_{\text{pauses}} \cdot \Big(1 - \min\big(1, \tfrac{\operatorname{sd}(L)}{\operatorname{mean}(L)}\big)\Big) \cdot \min\Big(1, \frac{\sum L}{\sum_{\text{runs} \ge 7 \text{ in window}} |\text{run} \cap \text{window}|}\Big).$$

**Natural holdout** (`holdout.find_holdouts`). Off spell $[a,b]$ of channel $c$ with $L \ge 7$, $a > 1$, not
overlapping a pulse pause of $c$, and

$$\frac{1}{N-1}\sum_{o \ne c} \mathbf 1\Big[\overline{\gamma}_{o,[a,b]} < \tfrac12\Big] > \tfrac12,\qquad
\text{conf} = \overline{\gamma}_{c,[a,b]} \cdot \frac{1}{N-1}\sum_{o\ne c}\big(1 - \overline{\gamma}_{o,[a,b]}\big).$$

**Single channel** (`holdout.find_single_channel`). $N \ge 3$; maximal runs of $\ge 7$ days with
$\sum_c M_{t,c} = N-1$ and the same channel $c^\*$ on throughout;
$\text{conf} = \operatorname{mean}_{c \ne c^\*} \overline{\gamma}_c \cdot (1 - \overline{\gamma}_{c^\*})$.
With $N = 2$ the pattern equals a holdout and is reported as one.

**Dark period** (`holdout.find_dark_periods`). Market has $\ge 2$ channels; maximal runs of $\ge 7$ days with
$M_{t,c} = 1$ for every active $c$ (a never-spending channel is off); $\text{conf} = \operatorname{mean}_c \overline{\gamma}_c$.

**Staggered launch** (`holdout.find_staggered_launches`). Delay $D_{m,c}$ = length of the off spell of
channel $c$ in market $m$ starting on day 1 ($0$ if $< 7$). Report $(m, c)$ over that spell if
$\exists\, m' \ne m:\ D_{m',c} \le D_{m,c} - 7$; $\text{conf} = \overline{\gamma}$ over the spell.

**Step change** (`step.find_steps`). $x_t = \ln s_t$ on days with $\gamma_t < \tfrac12$, $\pi_t = 1$, $s_t > \kappa$
(re-indexed $1..n'$). Noise $\hat\sigma = \operatorname{MAD}(\Delta x) / (0.6745\sqrt2)$.
Bayesian online changepoint detection over run length $r_t$, hazard $H = 1/100$, segment mean prior
$\mathcal N(\mu_0, v_0)$ with $\mu_0 = \operatorname{median} x$, $v_0 = 4$:

$$\tau_r^2 = \Big(\tfrac1{v_0} + \tfrac{n_r}{\hat\sigma^2}\Big)^{-1},\quad m_r = \tau_r^2\Big(\tfrac{\mu_0}{v_0} + \tfrac{\sum_r x}{\hat\sigma^2}\Big),\quad
\phi_r = \mathcal N\big(x_t;\ m_r,\ \hat\sigma^2 + \tau_r^2\big),$$

$$P(r_t = r+1) \propto P(r_{t-1} = r)\,\phi_r\,(1-H),\qquad P(r_t = 0) \propto \textstyle\sum_r P(r_{t-1} = r)\,\phi_r\,H.$$

Change score $u_s = \sum_{k=11}^{17} P(r_{s+14} = k)$. Changepoints are local maxima of $u > \tfrac12$ (window ±7),
each moved to $\arg\max_j |\bar x_{[j-14,j)} - \bar x_{[j,j+14)}|$ within ±7. With segment means $\bar x_i$, boundary
$i$ starts a step if

- $|\bar x_i - \bar x_{i-1}| \ge \ln 1.5$ and $|\bar x_i - \bar x_{i-2}| \ge \ln 1.5$ (not a return),
- abrupt: $\arg\min_w \mathrm{SSE}_w \le 7$, where $\mathrm{SSE}_w$ is the least-squares residual of
  $x_t \approx \ell_0(1-f_t) + \ell_1 f_t$, $f_t = \operatorname{clip}\big((t - t_0)/w, 0, 1\big)$, minimised over $t_0$ with the
  transition containing the change, $w \in \{1, 7, 14, 28, 56\}$, over up to 56 days each side.

It extends over later segments $j$ while $|\bar x_j - \bar x_{i-1}| \ge \ln 1.5$. Ratio
$= e^{\,\bar x_\text{step} - \bar x_{i-1}}$; conf $= \max u$ within ±7 of the change.

Confidence values are heuristic scores in $[0,1]$, not calibrated probabilities.

## 3. Scores

Precision / recall / F1 (found / false / missed).

| | dev (45 scenarios) | test (55 scenarios) | teammate, test |
|---|---|---|---|
| **Overall** | **0.945 / 0.920 / 0.932** (69 / 4 / 6) | **0.989 / 0.939 / 0.963** (92 / 1 / 6) | 0.978 / 0.918 / 0.947 (90 / 2 / 8) |
| channel pulse | 1.000 (10 / 0 / 0) | 1.000 (13 / 0 / 0) | 1.000 |
| dark period | 1.000 (12 / 0 / 0) | 1.000 (15 / 0 / 0) | 1.000 |
| natural holdout | 0.850 (17 / 3 / 3) | 0.939 (23 / 0 / 3) | 0.898 |
| single channel | 0.824 (7 / 0 / 3) | 1.000 (12 / 0 / 0) | 1.000 |
| staggered launch | 0.952 (10 / 1 / 0) | 0.970 (16 / 1 / 0) | 0.970 |
| step change | 1.000 (13 / 0 / 0) | 0.897 (13 / 0 / 3) | 0.857 |
| false alarms on no-event scenarios | 0.000 | 0.000 | 0.000 |

Boundary error is 0 days (median and p90) on both splits. The test score is from `pulse-extended` @
0528a12, the project's **second** final run on the sealed test split (the first is 34cd15e on
`detection-library`).
