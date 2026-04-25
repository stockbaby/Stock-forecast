# Portfolio Module

This module converts model scores into a competition-ready `result.csv`.

Current baseline:

- select the top 5 stocks by score
- assign weights proportional to positive scores
- cap the total weight sum at 1.0

Later upgrades can add:

- equal-weight and softmax-weight comparisons
- sector diversification constraints
- uncertainty filtering
- cash retention logic
