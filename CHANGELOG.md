# CHANGELOG

## UNRELEASED (0.0.1)

### ⚠ BREAKING CHANGES

* handlers now use `process` method rather than `__call__`.
* update message handling and validation protocols; improve single-responsibility

### Features

* demo notebooks
* openai compatibility layer
* refactor Caller, add tool invocation
* simplify templates; add tests for JinjaMessageTemplate initialization and rendering with complex templates

### Bug Fixes

* configure .typos.toml to ignore output cell tracebacks in .ipynb files
* enhance protocol type definitions and improve CallableWithSignature documentation
* simplify protocol naming and unify CallerWithSignature

### Code Refactoring

* handlers now use `process` method rather than `__call__`.
* update message handling and validation protocols; improve single-responsibility
