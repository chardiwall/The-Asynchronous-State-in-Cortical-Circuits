"""Machinery: *how* this project computes, as opposed to *what* the paper says.

Each block keeps the paper's own equations and one entry point. Everything else lives here,
so that reading a block shows you the phenomenon rather than the scaffolding around it.

Three of these are genuinely shared -- `tasks` (Slurm-array bookkeeping), `plotting` (the
paper's colour code and scale-bar style) and `checks` (qualitative trend tests). The rest are
machinery for one block: `psc`, `brian_batch` and `chunking` serve block 1, `glauber` and
`glauber_panels` serve block 2. They are here anyway, because being block-specific does not
make them science.

**Reuse over restatement.** A lib module may import a block's equations -- `glauber` builds
its weights from block 2's S-Eq(1) connectivity rather than restating it -- because the
optimisation must compute the same thing the paper states, and duplicating the statement is
how the two drift apart. The same reasoning applies between blocks: if two blocks genuinely
need the same thing, import it rather than copy it. Nothing forbids a block importing
another.

What that is not licence for: the blocks measure different models, so estimators that merely
*sound* alike are usually different on purpose. Block 2's `r` is a raw-sample correlation on
binary strings and block 3's is a windowed-rate spike-count correlation; collapsing them
would be a bug, not a deduplication. The part they really do share -- the rule for excluding
undefined pairs -- already lives in `analysis.mean_of_defined_pairs`. When a new shared piece
appears, put it in `analysis` if it is science and here if it is machinery.
"""
