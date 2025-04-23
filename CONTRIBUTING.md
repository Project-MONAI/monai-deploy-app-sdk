- [Introduction](#introduction)
  - [Communicate with us](#communicate-with-us)
- [The contribution process](#the-contribution-process)
  - [Preparing pull requests](#preparing-pull-requests)
    - [Checking the coding style](#checking-the-coding-style)
      - [Exporting modules](#exporting-modules)
    - [Unit testing](#unit-testing)
    - [Building the documentation](#building-the-documentation)
    - [Automatic code formatting](#automatic-code-formatting)
    - [Signing your work](#signing-your-work)
    - [Utility functions](#utility-functions)
  - [Submitting pull requests](#submitting-pull-requests)
- [The code reviewing process](#the-code-reviewing-process)
  - [Reviewing pull requests](#reviewing-pull-requests)
- [Admin tasks](#admin-tasks)
  - [Release a new version](#release-a-new-version)

## Introduction

Welcome to Project MONAI Deploy App SDK! We're excited you're here and want to contribute. This documentation is intended for individuals and institutions interested in contributing to MONAI Deploy App SDK. MONAI Deploy App SDK is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

Please also refer to [MONAI Contributing Guide](https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md) for general information as well as MONAI Core specifics.

### Communicate with us

We are happy to talk with you about your needs for MONAI Deploy App SDK and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point. If you are looking for an issue to resolve that will help Project MONAI Deploy App SDK, see the [*good first issue*](https://github.com/Project-MONAI/monai-deploy-app-sdk/labels/good%20first%20issue) and [*Contribution wanted*](https://github.com/Project-MONAI/monai-deploy-app-sdk/labels/Contribution%20wanted) labels.

## The contribution process

_Pull request early_

We encourage you to create pull requests early. It helps us track the contributions under development, whether they are ready to be merged or not. Change your pull request's title, to begin with `[WIP]` and/or [create a draft pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests) until it is ready for formal review.

Please note that, as per PyTorch, MONAI Deploy App SDK uses American English spelling. This means classes and variables should be: normali**z**e, visuali**z**e, colo~~u~~r, etc.

For local development, please execute the following command:

```bash
./run setup  # '-h' for help
```

This will set up the development environment, installing necessary packages.

### Preparing pull requests

To ensure the code quality, MONAI Deploy App SDK relies on several linting tools ([flake8 and its plugins](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), [isort](https://github.com/timothycrosley/isort)),
static type analysis tools ([mypy](https://github.com/python/mypy), [pytype](https://github.com/google/pytype)), as well as a set of unit/integration/system tests.

This section highlights all the necessary preparation steps required before sending a pull request.
To collaborate efficiently, please read through this section and follow them.

- [Checking the coding style](#checking-the-coding-style)
- [Unit testing](#unit-testing)
- [Building documentation](#building-the-documentation)
- [Signing your work](#signing-your-work)

#### Checking the coding style

Coding style is checked and enforced by flake8, black, and isort, using [a flake8 configuration](https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/setup.cfg) similar to [PyTorch's](https://github.com/pytorch/pytorch/blob/master/.flake8).
Before submitting a pull request, we recommend that all linting should pass, by running the following command locally:

```bash
# Setup development environment
./run setup

# Run the linting and type checking tools
./run check --codeformat  # or -f

# Try to fix the coding style errors automatically
./run check --autofix

# Show help
./run check --help
```

License information: all source code files should start with this paragraph:

```python
# Copyright 2021-2024 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

##### Exporting modules

If you intend for any variables/functions/classes to be available outside of the file with the edited functionality, then:

- Create or append to the `__all__` variable (in the file in which functionality has been added), and
- Add to the `__init__.py` file.

#### Unit testing

MONAI Deploy App SDK tests are located under `tests/` (it has `unit`/`integration`/`performance`/`system` subfolders.)

- The unit test's file name currently follows `test_[module_name].py` or `test_[module_name]_dist.py`.
- The `test_[module_name]_dist.py` subset of unit tests requires a distributed environment to verify the module with distributed GPU-based computation.
- The integration test's file name follows `test_[workflow_name].py`.

A bash script (`run`) is provided to run all tests locally.
Please run `./run test -h` to see all options.

To run a particular test, for example, `tests/unit/test_sizeutil.py`:

```bash
./run pytest tests/unit/test_sizeutil.py
```

Before submitting a pull request, we recommend that all linting and tests
should pass, by running the following command locally:

```bash
./run check -f
./run test
```

or (for new features that would not break existing functionality):

```bash
./run check -f
./run test all unit  # execute unit tests only
```

_If it's not tested, it's broken_

All new functionality should be accompanied by an appropriate set of tests.
MONAI Deploy App SDK functionality has plenty of unit tests from which you can draw inspiration,
and you can reach out to us if you are unsure of how to proceed with testing.

MONAI Deploy App SDK's code coverage report is available at [CodeCov](https://codecov.io/gh/Project-MONAI/monai-deploy-app-sdk).

#### Building the documentation

:::{note}
Please note that the documentation builds successfully in Python 3.9 environment, but fails with Python 3.10.
:::

MONAI's documentation is located at `docs/`.

```bash
./run gen_docs  # '-h' for help
```

The above command builds HTML documents in the `dist/docs` folder, they are used to automatically generate documents in [https://docs.monai.io](https://docs.monai.io).

To test HTML docs locally with development server, execute below command:

```bash
./run gen_docs_dev  # '-h' for help
```

The above command launches a live server to build docs in runtime.

Before submitting a pull request, it is recommended to:

- edit the relevant `.md`/`.ipynb` files in [`docs/source`](https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/docs/source) and [`notebooks/tutorials`](https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/notebooks/tutorials) accordingly.
  - [The MyST Syntax Guide](https://myst-parser.readthedocs.io/en/v0.15.2_a/syntax/syntax.html)
  - [Optional MyST Syntaxes](https://myst-parser.readthedocs.io/en/v0.15.2_a/syntax/optional.html)
  - [MyST Syntax Reference](https://myst-parser.readthedocs.io/en/v0.15.2_a/syntax/reference.html)
  - [MyST with Sphinx](https://myst-parser.readthedocs.io/en/v0.15.2_a/sphinx/index.html)
  - [MyST-NB User Guide](https://myst-nb.readthedocs.io/en/latest/)
- For Diagram, we use [sphinxcontrib-mermaid](https://sphinxcontrib-mermaid-demo.readthedocs.io/en/latest/).
  - [Mermaid syntax](https://mermaid-js.github.io/mermaid/#/)
  - We use [`classDiagram`](https://mermaid-js.github.io/mermaid/#/classDiagram) to visualize Operator (as a class) with multiple input/output labels (input attributes as fields, output attributes as methods) and connections between operators (as an edge label).
- Please check [#124](https://github.com/Project-MONAI/monai-deploy-app-sdk/pull/124) for the detailed instruction.

- check the auto-generated documentation (by browsing the generated documents after executing `./run gen_docs_dev`)
- build HTML documentation locally (`./run gen_docs`)
- execute `./run clean_docs` to clean temporary and final documents.

#### Automatic code formatting

#### Signing your work

MONAI enforces the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) on all pull requests.
All commit messages should contain the `Signed-off-by` line with an email address. The [GitHub DCO app](https://github.com/apps/dco) is deployed on MONAI. The pull request's status will be `failed` if commits do not contain a valid `Signed-off-by` line.

Git has a `-s` (or `--signoff`) command-line option to append this automatically to your commit message:

```bash
git commit -s -m 'a new commit'
```

The commit message will be:

```text
    a new commit

    Signed-off-by: Your Name <yourname@example.org>
```

Full text of the DCO:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

#### Utility functions

MONAI provides a set of generic utility functions and frequently used routines.
These are located in [`monai/deploy/utils`](https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/monai/deploy/utils).
Users are encouraged to use these common routines to improve code readability and reduce code maintenance burdens.

For string definition, [f-string](https://www.python.org/dev/peps/pep-0498/) is recommended to use over `%-print` and `format-print` from python 3.6. So please try to use `f-string` if you need to define any string object.

### Submitting pull requests

Please see this [general guidance](https://github.com/gabrieldemarmiesse/getting_started_open_source)

## The code reviewing process

Please see this [general guidance](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)

### Reviewing pull requests

At least one contributor of the project needs to approve a pull request.

## Admin tasks

The contributors with Admin role in the project handle admin tasks.

### Release a new version

[github ci](https://github.com/Project-MONAI/monai-deploy-app-sdk/actions)

[monai-deploy-app-sdk issue list](https://github.com/Project-MONAI/monai-deploy-app-sdk/issues)
