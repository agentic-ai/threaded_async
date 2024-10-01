#!/bin/bash
(cd src && mypy threaded_async --check-untyped-defs)
