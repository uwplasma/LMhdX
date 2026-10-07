# References

The versioned benchmark manifest is `benchmarks/provenance.json`. It records the
exact repository evidence, checksums for local literature artifacts, and the
tests and workflows that consume each source.

Core references used by the implemented models and validation contracts include:

- U. Müller and L. Bühler, *Magnetofluiddynamics in Channels and Containers*,
  Springer, 2001 — inductionless duct equations, wall effects, and canonical
  Hartmann/Shercliff/Hunt regimes.
- T. Q. Hua, J. S. Walker, B. F. Picologlou, and C. B. Reed, “Three-dimensional
  MHD flows in rectangular ducts of liquid-metal-cooled blankets,” Argonne
  National Laboratory report ANL/FPP/TM-228, 1988,
  [OSTI 6789158](https://www.osti.gov/biblio/6789158) — the inertialess
  core-flow equations, the side-layer closure and the ANL fringe reference of
  `lmhdx.coreflow`.
- J. S. Walker, “Magnetohydrodynamic flows in rectangular ducts with thin
  conducting walls. Part I,” *Journal de Mécanique* 20, 1981 — the fully
  developed thin-wall limits.
- C. Mistrangelo et al., *Fusion Engineering and Design* 173, 2021, 112795 —
  the `N^(-1/3)` inertial correction to inertialess pressure drops.
- S. Molokov and A. Politis, *Fusion Engineering and Design* 209, 2024, 114688
  — the core-flow model of a circular pipe.
- S. Smolentsev et al., “An approach to verification and validation of MHD
  codes for fusion applications,” *Fusion Engineering and Design*,
  [doi:10.1016/j.fusengdes.2014.04.049](https://doi.org/10.1016/j.fusengdes.2014.04.049).
- S. Smolentsev, R. Moreau, L. Bühler, and C. Mistrangelo, “MHD thermofluid
  issues of liquid-metal blankets,” *Fusion Engineering and Design*, 2010,
  [doi:10.1016/j.fusengdes.2010.02.038](https://doi.org/10.1016/j.fusengdes.2010.02.038).
- S. Smolentsev, “Physical background, computations and practical issues of
  the magnetohydrodynamic pressure drop in a fusion liquid metal blanket,”
  *Fluids* 6, 2021, 110,
  [doi:10.3390/fluids6030110](https://doi.org/10.3390/fluids6030110) — the
  thin-layer friction law of an insulated rectangular duct behind the duct
  design law.
- R. Nishio, M. Kondo, T. Tanaka, and N. Oono-Hori, “Experimental and analytical
  investigations to reduce MHD pressure drop for liquid LiPb fusion blanket
  systems,” *Nuclear Materials and Energy* 44, 2025, 101965,
  [doi:10.1016/j.nme.2025.101965](https://doi.org/10.1016/j.nme.2025.101965) —
  the fixed-area optimum aspect ratio of an insulated duct, before the duct
  design law stated it.
- I. Celik et al., “Procedure for estimation and reporting of uncertainty due
  to discretization in CFD applications,” *Journal of Fluids Engineering* 130,
  2008, [doi:10.1115/1.2960953](https://doi.org/10.1115/1.2960953) — the
  three-grid observed-order and Grid Convergence Index reporting convention.
- S. Patankar and D. Spalding, “A calculation procedure for heat, mass and
  momentum transfer in three-dimensional parabolic flows,” *International
  Journal of Heat and Mass Transfer* 15, 1972,
  [doi:10.1016/0017-9310(72)90054-3](https://doi.org/10.1016/0017-9310(72)90054-3)
  — the segregated pressure-linked momentum-correction foundation used by the
  B2 steady iteration.
- J. van Doormaal and G. Raithby, “Enhancements of the SIMPLE method for
  predicting incompressible fluid flows,” *Numerical Heat Transfer* 7, 1984,
  [doi:10.1080/01495728408961817](https://doi.org/10.1080/01495728408961817)
  — consistent momentum-diagonal pressure correction and convergence economy.
- J. Sommeria and R. Moreau, “Why, how, and when, MHD turbulence becomes
  two-dimensional,” *Journal of Fluid Mechanics* 118, 1982,
  [doi:10.1017/S0022112082001177](https://doi.org/10.1017/S0022112082001177) —
  basic Q2D Hartmann-friction closure.
- A. Pothérat, J. Sommeria, and R. Moreau, “An effective two-dimensional model
  for MHD flows with transverse magnetic field,” *Journal of Fluid Mechanics*
  424, 2000,
  [doi:10.1017/S0022112000001944](https://doi.org/10.1017/S0022112000001944) —
  higher-order Q2D corrections and model limits.
- ALEX results, “A comparison of measurements from a round and a rectangular
  duct with 3-D code predictions,” 1987 — Benchmark B1/B2 pressure and flow
  observables.
- B. Wynne et al., “FreeMHD: Validation and verification of the open-source,
  multi-domain, multi-phase solver for electrically conductive flows,”
  *Physics of Plasmas* 32, 2025,
  [doi:10.1063/5.0230242](https://doi.org/10.1063/5.0230242) — the
  [FreeMHD source](https://github.com/PlasmaControl/FreeMHD) supplies the
  independent finite-volume implementation used by the executable comparison.
- M. K. Jung et al., “Extension of a multi-region free-surface MHD solver
  beyond the inductionless approximation,” 2026,
  [arXiv:2606.18745](https://arxiv.org/abs/2606.18745) — FreeMHD2 resolves the
  induced field with a divergence-free vector-potential formulation and
  validates free-surface response against LMX-U. This establishes full
  induction and free surfaces as an external high-fidelity boundary rather
  than an LMhdX runtime target.
- T. Hua et al., “MHD capabilities in SAM and NekRS for fusion liquid metal
  blanket applications,” *Fusion Engineering and Design* 230, 2026,
  [doi:10.1016/j.fusengdes.2026.115888](https://doi.org/10.1016/j.fusengdes.2026.115888)
  — reduced system models, high-order GPU CFD, and ALEX square/round-duct
  validation define important independent comparison targets.
- C. Moreno, A. Bader, and P. Wilson, “ParaStell: parametric modeling and
  neutronics support for stellarator fusion power plants,” *Frontiers in
  Nuclear Engineering* 3, 2024,
  [doi:10.3389/fnuen.2024.1384788](https://doi.org/10.3389/fnuen.2024.1384788)
  — the external CAD/neutronics boundary for future stellarator blanket-design
  coupling.
- D. Panici et al., “The DESC stellarator code suite. Part 1. Quick and
  accurate equilibria computations,” *Journal of Plasma Physics* 89, 2023,
  [doi:10.1017/S0022377823000272](https://doi.org/10.1017/S0022377823000272)
  — differentiable, high-accuracy stellarator equilibrium and the natural
  upstream field/geometry boundary for liquid-metal design constraints.
- P. Fischer et al., “NekRS, a GPU-accelerated spectral element Navier--Stokes
  solver,” *Parallel Computing* 114, 2022,
  [doi:10.1016/j.parco.2022.102982](https://doi.org/10.1016/j.parco.2022.102982)
  — the reference for production GPU-resident high-order CFD and strong-scaling
  evidence.
- H. Walker and P. Ni, “Anderson acceleration for fixed-point iterations,”
  *SIAM Journal on Numerical Analysis* 49, 2011,
  [doi:10.1137/10078356X](https://doi.org/10.1137/10078356X) — residual-window
  acceleration, its linear-GMRES relationship, and the basis for treating
  history depth, scaling, regularization, and safeguarding as numerical
  algorithm choices rather than arbitrary tuning.
- A. Griewank and A. Walther, “Algorithm 799: Revolve: An implementation of
  checkpointing for the reverse or adjoint mode of computational
  differentiation,” *ACM Transactions on Mathematical Software* 26, 2000,
  [doi:10.1145/347837.347846](https://doi.org/10.1145/347837.347846) — the
  time/memory checkpointing principle used by bounded trajectory adjoints.
- H. Zhang and A. Constantinescu, “PETSc TSAdjoint: a discrete adjoint ODE
  solver for first-order and second-order sensitivity analysis,” 2019,
  [arXiv:1912.07696](https://arxiv.org/abs/1912.07696) — discrete transient
  adjoints and checkpoint scheduling for large time-dependent simulations.
- [Optimistix adjoint documentation](https://docs.kidger.site/optimistix/api/adjoints/)
  — the distinction between implicit derivatives of converged solves and
  checkpointed derivatives of finite solver iterations.
- D. Kochkov et al., “Machine learning–accelerated computational fluid
  dynamics,” *PNAS* 118, 2021,
  [doi:10.1073/pnas.2101784118](https://doi.org/10.1073/pnas.2101784118) —
  traceable finite-volume and pseudospectral CFD kernels on accelerators.
- D. Bezgin, A. Buhendwa, and N. Adams, “JAX-Fluids 2.0: Towards HPC for
  differentiable CFD of compressible two-phase flows,” *Computer Physics
  Communications* 308, 2025,
  [doi:10.1016/j.cpc.2024.109433](https://doi.org/10.1016/j.cpc.2024.109433)
  — multi-accelerator differentiation and scaling evidence for a broader CFD
  class.
- I. Yashchuk, “Bringing PDEs to JAX with forward and reverse modes automatic
  differentiation,” [arXiv:2309.07137](https://arxiv.org/abs/2309.07137) —
  implicit tangent and adjoint equations for composing converged PDE solves
  with JAX programs.
- The JAX documentation on
  [gradient checkpointing](https://docs.jax.dev/en/latest/301/remat.html),
  [distributed arrays](https://docs.jax.dev/en/latest/201/sharding.html), and
  [autodiff with sharding](https://docs.jax.dev/en/latest/301/sharding-ad.html)
  — current memory/parallel semantics used to design LMhdX acceptance tests.

Each numerical result should cite the LMhdX version/commit and the specific
benchmark source used for its physical claim.
