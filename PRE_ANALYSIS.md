# Pre-analysis statement

Written before any estimator was run against simulated output, and first
committed together with the code. Date: 2026-09-13. The body below is unedited.

We inject two parameters. The disposition strength `delta_i` enters as a multiplier on
a daily sell hazard: a position trading above its purchase price has its immediate
probability of sale scaled by `(1 + delta_i)`, one below its purchase price by
`(1 - delta_i)`. The overprecision parameter `kappa_i` enters the same hazard as a
level shift, `h_0 * (1 + 4 * kappa_i)`, so it governs how often an account acts at all
without touching which domain it closes. Neither PGR, PLR nor turnover is parametrised
anywhere in the code; all three are emergent. Per scenario, the tabulated `delta` and
`kappa` are the means of independent Beta draws across the population, degenerate at
zero where the scenario specifies zero.

We expect the following. In scenario 1 both estimators should be quiet: PGR - PLR
within two bootstrap standard errors of zero, and a turnover slope on gross returns
indistinguishable from zero. In scenarios 2 and 3 we expect PGR - PLR to be positive
and larger in 3 than in 2, monotone in the injected delta but not equal to it, since
the hazard multiplier and the realized ratio are different quantities. In scenarios 4
and 5 we expect the disposition estimator to stay quiet while the turnover slope on
net returns turns negative and more negative as kappa rises, with the gross slope
close to zero. Scenario 6 should show both, and we expect the measured disposition
ratio to be attenuated relative to scenario 3 because higher churn shortens holding
periods and moves positions closer to their purchase price. Scenarios 7 and 8 carry
delta = 0 and we nonetheless expect a positive, statistically significant
PGR - PLR in both: rebalancing trims the positions that have drifted up, and a belief
in reversal keys on trailing returns that correlate with the gain indicator. Those two
are the point of the exercise, not a failure of it.

On the overconfidence side we expect the Barber-Odean contrast only where kappa
carries the variation. Wherever the selling rule conditions on the purchase price
(scenarios 2, 3, 6) or trims winners (scenario 7), we expect reverse causality to push
the gross slope positive, because an account that happened to perform well holds more
winners, sells more of them, and books higher turnover. Cash drag is eliminated by
construction through same-day redeployment, and the bid-ask spread is excluded from
the gross measure by definition, so neither should appear unless we deliberately
switch them on.

Finally, we expect the realized sample correlation between `delta_i` and `kappa_i` to
be within Monte Carlo noise of zero by construction, and the correlation between
`delta_i` and realized turnover to be materially negative, because a strong
disposition agent holds losers indefinitely and thereby trades less.
