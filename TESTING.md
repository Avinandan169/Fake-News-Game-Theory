# Testing Guide

Run all automated checks with:

```bash
python main.py --test
```

The suite uses Python's built-in `unittest` module. A passing run ends with `OK`.

| Test case | What is tested | Expected behaviour |
|---|---|---|
| Payoff matrix structure | The matrix contains two creator strategies, two platform strategies, and two payoffs per cell. | Shape is `(2, 2, 2)`. |
| Nash equilibrium | The mixed-equilibrium formula for the chosen payoff matrix. | Creator spreads fake news with 12.5%; platform uses weak moderation with 66.7%. |
| Zero initial exposure | No user has encountered the item at the start. | Exposure stays 0 in every round. |
| Zero rounds | The simulation is not allowed to advance. | Final exposure equals the initial exposure. |
| Zero sharing probability | Exposed users never share. | Final exposure equals the initial exposure. |
| Population limit | Very high spread in a small population. | Exposure never becomes greater than the population. |
| Repeatability | The same inputs and random seed are used twice. | Both histories are exactly identical. |
| Invalid initial exposure | More initially exposed users than the population. | The program raises a validation error. |
| Invalid probability | A probability above 100%. | The program raises a validation error. |

## Manual terminal-input checks

Try these when demonstrating the program:

1. At the population prompt, enter `0` or `abc`. The program should explain the valid range and ask again.
2. At the initial-exposure prompt, enter a number greater than the population. The program should reject it.
3. At the sharing-probability prompt, enter `-1` or `101`. The program should reject it.
4. Run twice with the same seed. The results should match.
5. Change only the seed, then run again. The exact illustrative totals can change, but the three scenarios still use the same starting conditions within that run.

## What a successful comparison should show

For ordinary positive sharing values, moderate and strict moderation should reduce exposure relative to no moderation in this model. Exact totals can vary with inputs and seed. With zero sharing probability, all strategies correctly have the same final exposure because no spreading occurs.
