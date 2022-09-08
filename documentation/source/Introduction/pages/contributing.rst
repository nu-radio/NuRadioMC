Contributing to NuRadioMC
=========================
Thank you for considering to contribute to NuRadioMC.
NuRadioMC is intended as community based simulation and reconstruction software for
radio (neutrino) detectors. Everyone is invited to contribute and use the
software for their experiment.

If you intend to contribute significantly to NuRadioMC, please contact
`@anelles <https://github.com/anelles>`_ and `@cg-laser <https://github.com/cg-laser>`_ on GitHub so that we are informed about ongoing
activities. Both will also be able to provide commit access to the repository.

Workflow
--------------
If you find an issue or bug in NuRadioMC, please `create an issue on GitHub <https://github.com/nu-radio/NuRadioMC/issues>`_.
If you want to contribute to NuRadioMC, please provide your code addition in a new branch and `make a pull request <https://github.com/nu-radio/NuRadioMC/pulls>`_.

We loosely follow the git flow model. A detailed tutorial is given  `here <https://jeffkreeftmeijer.com/git-flow/>`_.
A short summary is provided below.

* The ``master`` branch is reserved for stable releases. A user can always check out ``master`` to get the latest stable version of NuRadioMC.
* All development happens on the ``develop`` branch. All feature branches will be merged (after review) into ``develop``.
  Make sure to specify that you want to merge into ``develop`` when creating a new pull request on github.
* We allow for hotfixes. These branches will be merged both into ``develop`` as well as into ``master`` where also a new tag and release is made.

  .. Important::

    When merging hotfixes into both develop and master, make sure the changelog & version number are correct for both!

To start developing a new feature or hotfix, first create a new branch:

.. code-block:: Python

  git checkout develop
  git pull
  git checkout -b feature/my_new_feature # creates a new branch

.. Note::

  If you are writing a **hotfix**, which should also be merged into ``master``,
  replace the first line by ``git checkout master``

Now code can be written, fixed, committed and pushed to git as normally (**exception**: for your first
push to the git repository, you need to include ``--set-upstream``, as the branch initially only exists
on your local machine).

.. code-block:: Python

  git push --set-upstream origin feature/my_new_feature


Once you are ready for your code to be merged into ``develop`` (for features and hotfixes) and/or
``master`` (for hotfixes only), you should `create a pull request <https://github.com/nu-radio/NuRadioMC/pulls>`_.

Before you make a pull request, make that your code:

* is correct - it should fix bugs, not introduce more of them!
* is clearly documented - functions should have
  :ref:`correctly written docstrings <Introduction/pages/contributing:Writing docstrings>`
  , and comments where appropriate.
* is reflected both in the **changelog** and by an appropriate update of the
  :ref:`version number <Introduction/pages/contributing:Update the version number / dependencies>`.

You will only be able to merge your pull request once:

* It succesfully completes the `tests <https://github.com/nu-radio/NuRadioMC/actions/workflows/run_tests.yaml>`_.
  These will run automatically each time you push to the repository, and are implemented to check that your code
  does not break anything, the new code is correctly documented, and NuRadioMC can still be built.
* One of the core developers has approved your pull request. **Please wait at least 24 hours to merge your pull request,
  even if it has been approved, so that other developers may also have a look - they might find something the first reviewer
  missed!**

Coding conventions
------------------
In general we try to follow 'industry' coding workflow conventions. So, if
something is not explicitly mentioned here, please resort to 'best practices'.
NuRadioMC tries to follow `PEP-8 coding conventions <https://www.python.org/dev/peps/pep-0008/>`_.

Please document your code extensively, especially the physics behind it.
Each function should come with a docstring where all variables are defined.
All variables representing something with a unit must make use of the
NuRadioMC/utilities/units convention. Please avoid adding additional
dependencies without clearing this with the core developers.

How to
------

Writing docstrings
__________________
All parts of the software are documented in the source code using python
docstrings. Human-Readable documentation can then be generated using Sphinx.
We use the `numpy docstring syntax <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
Briefly, this means a docstring should look like this:

.. code-block:: Python

  def example_function(x, y, mode="quickly")
  """
  Short, one-line description of function or method

  Here there is space for an optional longer, more detailed description.
  While this is not currently enforced, please **always** include at least
  the one-line docstring!

  The longer docstring may contain multiple paragraphs. Paragraphs are separated
  by newlines.

  Parameters
  ----------
  x : float
    This is a description of x (e.g., distance in metres)
  y : int
    This is a description of y
  mode : str, default "quickly"
    Some parameters may have a limited list of options
    Lists need to be separated from the rest of the docstring
    by newlines, like so:

    * "quickly" - do it quickly (default)
    * "slowly" - do it slowly
    * "multiline" - for a list entry over multiple lines,
      don't forget to indent!

  Returns
  -------
  res : float
    This is a description of the function result


  Examples
  --------

  .. code-block::

    x = 1.3
    y = 4
    result = example_function(x, y)
  """

Please only use docstrings sections allowed by `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The most useful ones are
the short + (optional) extended summary, ``Parameters``, ``Returns``, ``Yields``,  ``See Also``, ``Notes``, ``Examples``.
Section titles should always be underlined with (at least) the same number of hyphens ``-``
as the length of the section title, as in the above example.

Docstrings, as well as the rest of the documentation, are written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`_.
Please consult this link for correct syntax. Some of the basics are also summarized in the
:ref:`Writing additional documentation <Introduction/pages/contributing:Writing additional documentation>` section below.

Update the version number / dependencies
________________________________________
``NuRadioMC`` is built and published using `poetry <https://python-poetry.org/docs/pyproject/>`_. To update the current version number,
open the ``pyproject.toml`` file in the NuRadioMC root directory, and update ``version`` under ``[tool.poetry]``:

.. code-block::

  [tool.poetry]
  name = "NuRadioMC"
  version = "2.1.0"

We use `semantic versioning <https://semver.org/>`_, i.e. MAJOR.MINOR.PATCH.
Dependencies are also maintained in ``pyproject.toml``. To update the dependencies:

* If you are adding a **core** dependency, first ensure that the core developers agree!
  Then add your dependency (e.g. ``numpy``)

  .. code-block::

    [tool.poetry.dependencies]
    numpy = "1.21.1"

  under ``[tool.poetry.dependencies]``. Acceptable version specifications are ``"4.1.1"`` (4.1.1 only),
  ``">=4.1.1"`` (4.1.1 or greater), or ``"*"`` (any version). Please do not use poetry-specific version
  specifiers like ``^`` or ``~``.
* If you are adding an **optional** dependency, add your dependency under ``[tool.poetry.dev-dependencies]``.
  Additionally, please name the feature that requires this dependency, and add it under ``[tool.poetry.extras]``.
  E.g. in order to generate the documentation, we require ``Sphinx``, ``sphinx-rtd-theme`` and ``numpydoc`` to be installed.
  This is specified in ``pyproject.toml`` as follows:

  .. code-block::

    [tool.poetry.dev-dependencies]
    Sphinx = "*"
    sphinx-rtd-theme = "*"
    numpydoc = "*"

    [tool.poetry.extras]
    documentation = ["Sphinx", "sphinx-rtd-theme", "numpydoc"]

Writing additional documentation
________________________________
Code documentation is generated automatically using `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_
and `sphinx.ext.autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc>`_.
Any new modules with correctly written docstrings will therefore be added to the :doc:`code documentation </NuRadioMC/pages/code_documentation>`
without additional input. However, in many cases it is extremely helpful if additional documentation is available.
This may take different forms:

* Clear, well-annotated examples scripts that users can run and modify to get to grips with the new features.
  Please place these examples in the ``NuRadioMC/examples`` or ``NuRadioReco/examples`` folders - having scripts
  inside the module folders confuses ``sphinx``.
* Additional :doc:`manuals </NuRadioMC/pages/manuals>` or tutorials, to be published as part of the online documentation.

The documentation is written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`_ and built using
`sphinx <https://www.sphinx-doc.org/en/master/index.html>`_. Please consult these websites for a more extensive overview
of the correct syntax. You can use any existing page of the documentation (which can be found in the ``documentation/source`` directory)
as a template for how to write more code. However, below is a summary of the basics:

Compiling the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to compile the documentation locally, make sure the required
:ref:`dependencies <Introduction/pages/installation:Optional Dependencies>`
are installed (this can be done by running the :ref:`installer <Introduction/pages/installation:Development version>`).
Compiling the documentation is then done by running

.. code-block::

  python documentation/make_docs.py

This will build the documentation at `documentation/build/html`
(open `main.html` to view it in your browser).

Headings and text
^^^^^^^^^^^^^^^^^
.. code-block:: reStructuredText

  Document Title
  ==============
  Headings should always be underlined by one of the following symbols:
  "= - _ ^ + ~ # < >".
  The underline must be at least as long as the title text.
  Nesting determines the level of heading.

  Subheading
  ----------
  This is a subsection

  Long text may be split over multiple lines; a new line / new paragraph is only
  started if it is separated by a newline

Some commonly used text formatting:

* ``*italicized*`` results in *italicized* text;
* ``**bold**`` results in **bold** text;
* ````single-spaced```` results in ``single-spaced`` text.

Lists
^^^^^
Lists can be included using "-", "*" or "+" (for bullet points), or
"1.", "2.", ... (enumerated) / "#." (automatically enumerated). Lists should always
be separated by newlines above and below from other text:

.. code-block:: reStructuredText

  Lists
  -----
  This is some text

  * This is the first bullet point
  * This is the second bullet point. Longer
    text may be split over multiple lines by
    indenting by 2 spaces
  * This is another bullet point

    #. This is an enumerated sub-list.
       Notice that it has been separated from its
       parent bullet point by a newlines
    #. Similarly, there will be another newline
       before continuing the bullet list

  * This is the last bullet point

Links and cross-references
^^^^^^^^^^^^^^^^^^^^^^^^^^
Links look like this: ```link text <https://link-url.com>`_``. Note the trailing underscore!
For internal links (e.g. to other parts of the documentation), we prefer
`cross-references <https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html>`_
instead. These depend on what is being linked to:

* For another page in the documentation, use ``:doc:``. E.g. ``:doc:`introduction </Introduction/pages/introduction>``` renders as
  :doc:`introduction </Introduction/pages/introduction>`. Use a leading ``/`` to use paths starting from
  the root ``documentation/source`` directory.
* One can reference a specific subsection instead by using ``:ref:`` and appending ``:Section title``. E.g.
  ``:ref:`this paragraph <Introduction/pages/contributing:Links and cross-references>``` links to
  :ref:`this paragraph <Introduction/pages/contributing:Links and cross-references>`. Note that there is **no**
  leading ``/`` in this case!
* Finally, one can refer to python modules, classes, functions etc. by using ``:mod:``, ``:class:``, ``:func:``
  respectively. The name of the function follows the same logic as in Python, e.g.
  ``:class:`base trace class <NuRadioReco.framework.base_trace>``` refers to the NuRadioReco
  :class:`base trace class <NuRadioReco.framework.base_trace>`

Showing code
^^^^^^^^^^^^
To render code, use the ``.. code-block::`` directive, optionally followed by the code language that is used
(e.g. ``Python``). As with lists, the code block needs to be separated from the rest of the text using
newlines. E.g. the following:

.. code-block:: reStructuredText

  Some text

  .. code-block:: Python

    def example_function(r):
      return r**2 + 5

  Some more text

renders as:

Some text

.. code-block:: Python

  def example_function(r):
    return r**2 + 5

Some more text
