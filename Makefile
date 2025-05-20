#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of plugin_libSGM
#
#     https://github.com/CNES/Pandora_plugin_libSGM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Autodocumented Makefile for python and C++ dev, see two sections.
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Dependencies : python3 venv g++ gcc ar
# Some Makefile global variables can be set in make command line: VENV, PYTHON, ...
# Recall: .PHONY  defines special targets not associated with files

############### GLOBAL VARIABLES ######################

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: PLUGIN_LIBSGM_VENV="other-venv/" make install
ifndef PLUGIN_LIBSGM_VENV
	PLUGIN_LIBSGM_VENV = "venv"
endif

# Check python3 globally
PYTHON=$(shell command -v python3)
ifeq (, $(PYTHON))
    $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

# Check Python version supported globally
PYTHON_VERSION_MIN = 3.8
PYTHON_VERSION_CUR=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')
ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif

################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      plugin_libSGM MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort


## Install section

.PHONY: venv
venv: ## create virtualenv in PLUGIN_LIBSGM_VENV directory if not exists
	@test -d ${PLUGIN_LIBSGM_VENV} || python3 -m venv ${PLUGIN_LIBSGM_VENV}
	@${PLUGIN_LIBSGM_VENV}/bin/python -m pip install --upgrade pip setuptools wheel # no check to upgrade each time

.PHONY: install
install: venv ## install plugin_libSGM
	@if ! ${PLUGIN_LIBSGM_VENV}/bin/pip list | grep -q pandora_plugin_libsgm; then \
		echo "Install plugin_libSGM from local directory"; \
		${PLUGIN_LIBSGM_VENV}/bin/pip install -e .[dev,docs]; \
	fi 2>/dev/null

.PHONY: check-library
check-library: ## check if the plugin is already installed in the PLUGIN_LIBSGM_VENV environment
	@if ${PLUGIN_LIBSGM_VENV}/bin/pip list | grep -q pandora_plugin_libsgm; then \
		echo "pandora_plugin_libsgm already installed"; \
	else \
		echo "pandora_plugin_libsgm not already installed"; \
		echo "the command to install it is : make install"; \
	fi 2>/dev/null


## Test section

.PHONY: test
test: install ## run all tests (
	@${PLUGIN_LIBSGM_VENV}/bin/pytest -m "not functional_tests" --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-functional
test-functional: install ## run functional tests only
	@echo "Run functional tests"
	@${PLUGIN_LIBSGM_VENV}/bin/pytest -m "functional_tests"


## Documentation section

.PHONY: docs
docs: install ## build sphinx documentation (source venv before)
	@${PLUGIN_LIBSGM_VENV}/bin/sphinx-build -M clean doc/source/ docs/build
	@${PLUGIN_LIBSGM_VENV}/bin/sphinx-build -M html doc/source/ doc/build -W --keep-going


## Clean section
.PHONY: clean ## remove all build, test, coverage and Python artifacts
clean: clean-venv clean-build clean-test clean-doc ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${PLUGIN_LIBSGM_VENV}

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr build/

.PHONY: clean-doc
clean-doc:
	@echo "+ $@"
	@rm -rf doc/build/
	@rm -rf doc/source/api_reference/

.PHONY: clean-test
clean-test:
	@echo "+ $@"
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml



