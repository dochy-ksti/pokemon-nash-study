# poke-ai3-python

Python bindings and scripts for `poke-ai3`.

This crate keeps PyO3 and maturin out of the core `poke-ai3` build. The Python
module name remains `poke_ai3` so existing Python scripts can continue to import
the native executor wrapper.
