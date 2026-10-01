"""Machinery: *how* this project computes, as opposed to *what* the paper says.

Each block keeps the paper's own equations and one entry point. Everything else lives here,
so that reading a block shows you the phenomenon rather than the scaffolding around it.

Three of these are genuinely shared -- `tasks` (Slurm-array bookkeeping), `plotting` (the
paper's colour code and scale-bar style) and `checks` (qualitative trend tests). The rest are
machinery for one block: `psc`, `brian_batch` and `chunking` serve block 1, `glauber` and
`glauber_panels` serve block 2. They are here anyway, because being block-specific does not
make them science.

**The dependency rule is one-directional, but not the obvious one.** A lib module may import
a block's equations -- `glauber` builds its weights from block 2's S-Eq(1) connectivity
rather than restating it -- because the optimisation must compute the same thing the paper
states, and duplicating the statement is how the two drift apart. What never happens is a
block importing another block: the three are independent, and that is the property worth
protecting.
"""
