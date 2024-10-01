#!/bin/bash
(cd src && pylint --rcfile=../.vscode/pylintrc threaded_async)
(cd tests && pylint --rcfile=../.vscode/pylintrc tests)
