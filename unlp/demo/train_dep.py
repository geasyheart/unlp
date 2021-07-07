# -*- coding: utf8 -*-

#

from unlp.components.parsers.biaffine.biaffine_dep import BiaffineDependency
from unlp.datasets.parsing.ctb8 import CTB8_SD330_DEV

dep = BiaffineDependency()
save_dir = "/tmp/dep/a"
dep.fit(
    # CTB8_SD330_TRAIN,
    CTB8_SD330_DEV,
    CTB8_SD330_DEV,
    save_dir=save_dir,
    transformer='ckiplab/albert-tiny-chinese',
    lr=5e-5,
    tree=True,
    proj=True,
    punct=True,
    max_sequence_length=128,
    n_mlp_arc=500,
    n_mlp_rel=100,
    mlp_dropout=0.2,
    epochs=10000,
    word_dropout=0.2
)

dep.predict(
    save_dir=save_dir,
    data=CTB8_SD330_DEV,
    transformer='ckiplab/albert-tiny-chinese',
    max_sequence_length=128,
    n_mlp_arc=500,
    n_mlp_rel=100,
    mlp_dropout=0.2,

)
