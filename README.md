# Automatic Differentiation for Option Greeks

## Overview

This project implements automatic differentiation (autodiff) from scratch and applies it to Black–Scholes option pricing.

The main objective is to understand how derivatives can be calculated computationally using the chain rule, rather than relying on manually derived formulas or numerical approximations.

Two approaches to automatic differentiation are implemented:

- Forward-mode autodiff using dual numbers
- Reverse-mode autodiff using computational graphs

The project also demonstrates how the same framework can be extended to calculate higher-order derivatives.

---

## Key Concepts

### Forward-Mode Automatic Differentiation

Forward-mode autodiff propagates function values and derivatives from the inputs towards the output.

A custom `Dual` class was implemented in which each number stores:

- its numerical value
- its derivative

Arithmetic operations such as addition, multiplication, division and exponentiation were overloaded so that derivatives could be propagated automatically using the chain rule.

The implementation was extended to support:

- Addition
- Subtraction
- Multiplication
- Division
- Powers
- Exponentials
- Logarithms
- Square roots
- Normal CDF

### Reverse-Mode Automatic Differentiation

Reverse-mode autodiff propagates gradients backwards through a computational graph.

A custom `Node` class was implemented to store:

- numerical values
- gradients
- parent nodes
- local derivatives

A topological ordering is constructed before performing the backward pass, allowing gradients to be propagated correctly through shared computational nodes.

Reverse mode is particularly useful when a function has many inputs but a single output, and forms the basis of backpropagation in machine learning.

---

## Black–Scholes Model

The project applies the autodiff implementations to the Black–Scholes European call option:

\[
C = S\Phi(d_1) - Ke^{-rT}\Phi(d_2)
\]

where

\[
d_1 =
\frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}
{\sigma\sqrt{T}}
\]

and

\[
d_2=d_1-\sigma\sqrt{T}.
\]

The following sensitivities are calculated using automatic differentiation:

- Delta
- Gamma
- Vega
- Rho
- Theta
- Speed

---

## Higher-Order Derivatives

Nested dual numbers are used to calculate higher-order derivatives.

For example:

\[
\Gamma = \frac{\partial^2 C}{\partial S^2}
\]

and

\[
\text{Speed} =
\frac{\partial^3 C}{\partial S^3}.
\]

This demonstrates how the same automatic differentiation framework can be extended beyond first-order derivatives.

---

## Validation

The autodiff results are validated using two independent approaches.

### Finite Differences

Central finite differences are used to independently estimate Delta and Gamma.

### Analytical Black–Scholes Formulas

The automatically calculated Greeks are compared with the standard analytical Black–Scholes expressions.

The results agree to numerical precision, providing a check that both the forward- and reverse-mode implementations are working correctly.

---

## Visualisation

The project examines how Delta and Gamma vary with the underlying stock price.

### Delta

Delta increases as the stock price increases, approaching 0 for sufficiently low stock prices and 1 for sufficiently high stock prices.

### Gamma

Gamma measures the rate of change of Delta with respect to the stock price. For the parameters used in the project, Gamma reaches its maximum slightly below the strike price.

These plots provide an intuitive interpretation of the behaviour of the option Greeks.

---

## Forward Mode vs Reverse Mode

Both approaches implement the same underlying chain rule but propagate derivatives in different directions.

| Method | Direction | Typical advantage |
|---|---|---|
| Forward mode | Input → Output | Few inputs, many outputs |
| Reverse mode | Output → Input | Many inputs, one output |

For the Black–Scholes example, reverse mode can calculate sensitivities with respect to multiple model parameters from a single backward pass.

---

## Project Structure

```text
quant-project-3-autodiff-greeks/
├── README.md
├── autodiff_greeks.ipynb
├── src/
│   └── autodiff.py
├── requirements.txt
└── .gitignore
