# Publish on PyPI

1. The code to publish is located in the `src/nn_with_backprop_pk` folder.

2. The folders `nn_with_backprop_pk`, `core`, and `utils` must each contain an `__init__.py` file.

3. This section in the `pyproject.toml` file defines the package metadata. The `name`, `version`, and `authors` fields are required:

    ```toml
    [tool.poetry]
    name = "nn-with-backprop-pk"
    version = "0.1.0"
    description = "Poetry file for the project NN with backprop"
    authors = ["ppkhairn"]
    readme = "README.md"
    packages = [
        { include = "nn_with_backprop_pk", from = "src" }
    ]
    ```

4. After making changes, run:

    ```bash
    poetry lock
    poetry install
    ```

5. To build the package, run:

    ```bash
    poetry build
    ```

    This will create a `dist/` folder containing `.tar.gz` and `.whl` files.

6. To test the package locally, run:

    ```bash
    pip install dist/<your-wheel-file>.whl
    ```

7. To configure Poetry with your PyPI token, run:

    ```bash
    poetry config pypi-token.pypi <your-api-token>
    ```

    You can get the token from PyPI under `Account Settings → Add API Token`.

8. Finally, to publish the package:

    ```bash
    poetry publish
    ```
