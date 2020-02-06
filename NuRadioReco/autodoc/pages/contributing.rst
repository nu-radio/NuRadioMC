Contributing to NuRadioReco
=============
  Thank you for considering to contribute to NuRadioReco.
  NuRadioReco is intended as community based reconstruction software for
  radio (neutrino) detectors. Everyone is invited to contribute and use the
  software for their experiment.

  If you intend to contribute significantly to NuRadioReco, please contact
  @anelles and @cg-laser on GitHub so that we are informed about ongoing
  activities. Both will also be able to provide commit access to the repository.

Workflow
--------------
Filing Tickets
____________
  If you work with NuRadioReco and you encounter a problem that you CANNOT solve,
  please file an issue. Please provide as much information as you can. Best, if
  you provide us with a minimal working example that reproduces the problem or
  refer to specific lines of code.

Submitting Pull Requests
____________
  If you work with NuRadioReco and you encounter a problem that you CAN solve,
  please provide your fix in a new branch and make a pull request. We work with
  continous integration, so you will immediately see, whether your code causes
  siginifcant problems or changes existing results. This is of course no
  guarantee that every piece of code is correct.

  The core team of developers will review your pull request as soon as possible
  and provide feedback. Once approved, you can merge your code into the master
  and delete the branch. Allow for at least 24h review time between the last
  change (commit) and the merge even if the pull request was approved quickly
  to allow for additional comments or concerns. Before merging the pull request,
  document the relevant changes in "changelog.txt". We use this information for
  our releases.

  .. Important::

    It is not permitted (or possible) to push changes directly to the master branch. Please always use pull requests!

Coding Conventions
____________
  In general we try to follow 'industry' coding workflow conventions. So, if
  something is not explicitly mentioned here, please resort to 'best practices'.
  NuRadioMC tries to follow `PEP-8 coding conventions <https://www.python.org/dev/peps/pep-0008/>`_

  Please document your code extensively, especially the physics behind it.
  Each function should come with its doc string where all variables are defined.
  All variables representing something with a unit must make use of the
  NuRadioMC/utilities/units convention. Please avoid adding additional
  dependencies without clearing this with the core developers.

Versioning
____________
  NuRadioReco differentiates between versions of the software and the file format.
  The software version is specified in the *__init__.py* file at the top level
  of the repository. The current version is 1.0.1. It is upped in increments of
  0.0.1 for large changes that change physics behavior and a tag is added on GitHub
  for every increment.

  File versions consist of a major and a minor file version, which are specified
  in the ``NuRadioRecoio`` module. Major versions are incremented for changes
  that break backward compatibility. Minor versions are incremented if the file
  format changes but old files remain readable, e.g. if a parameter is added
  or removed.

  By default, ``NuRadioRecoio`` will throw an error if the major file version is
  different and show a warning if the minor version is different. This can be
  overridden, but one should of course be careful when doing so.

Unit Tests
____________
  After every commit `Travis CI <https://travis-ci.com/>`_ will run a test on
  the repository that executes a number of test scripts, checks for errors
  and compares the results to a reference. For safety, the tests implemented in
  NuRadioMC are run as well, since it uses some elements of NuRadioReco.
  Whether the test passes or fails
  is shown as a green tick or red cross next to the commit on the GitHub page.
  Pull requests can not be merged until all tests pass.

What to do if Unit Tests Fail
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  If a unit test fails, it usually means that your changes have some unintended
  side effects that cause the event simulation or reconstruction to lead to
  different results. In that case, please find out which of your changes caused
  this and fix it.

  In rare cases, the tests can fail for other reasons:

    - Travis fails to download some necessary package. In that case, restarting the
      test usually fixes the issue.
    - Changes in NuRadioMC cause the tests to fail. A good sign for this is that
      the tests fail on the master branch as well. In this case, the issue with
      NuRadioMC needs to be fixed instead
    - Some elements of NuRadioMC use random numbers, which can cause random
      fluctuations to lead to failing tests. Especially the V_eff test in
      NuRadioMC is susceptible to this. Usually re-running the test solves this.
    - The C++ raytracer in NuRadioMC is numerically unstable. The effect is small,
      but results can vary slightly between different systems. Therefore, some
      margin of error is given when comparing tests that involve raytracing.
    - The changes in the reconstruction may be intended, i.e. because a
      reconstruction method was improved or a bug was found. In this case, the
      references have to be updated. All test scripts can be run with the option
      ``--create_reference``, which will make them produce a new reference file.
      Just create a new reference and commit it to the repository.

  .. Important::
    Only change the references if you are absolutely sure that all changes are
    intentional!


Documentation
-------------

Writing Docstrings
____________
  All parts of the software are documented in the source code using python
  docstrings. Human-Readable documentation can then be generated using Sphinx.
  We use the `numpy docstring syntax <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

  If you added, changed or removed a function in an existing module, there is
  nothing else you need to do (just make sure all your functions have docstrings).
  The same is true for changing or adding docstrings, your changes will
  automatically be included the next time the documentation is generated.

  If you added a new module, you need to make sure sphinx knows about it. To do
  so, go into the autodoc folder and open the .rst file corresponding to the
  folder that your module is in. The file contains a list of modules inside
  the folder that should be included in the documentation, so just add your
  module to the list and the next time the documentation is generated it will
  be included.

Deploying the Documentation
____________
  We update the documentation regularly (about weekly), but if you don't want
  to wait that long, here is how to update it yourself:
  First you need to set up a repository for the gh-pages branch. Just follow
  the steps in the section Setting up cloned repos on another machine from
  `this tutorial <https://daler.github.io/sphinxdoc-test/includeme.html>`_.
  The directory in which to set up the gh-pages repo
  (called *sphinxdoc-test-docs* in the tutorial) is specified under BUILDDIR in
  NuRadioReco/autodoc/Makefile . In our case, it is a folder called
  docs_NuRadioReco next to your NuRadioReco repo.
  Once you set everything up, go into the NuRadioReco/autodoc directory and
  execute the command ``make html``. This generates the documentation into the
  docs_NuRadioReco directory. Go into that directory and commit all changes to
  the html files. Push them to GitHub and the documentation is updated.

Technical Implementation
____________
  To generate the documentation from scratch,
  `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_
  is used.
  The documentation uses the sphinx rtd theme, which has to be installed before
  generating the documentation html files (``pip install sphinx-rtd-theme``). Some
  custom css code is stored under *custom_scripts/styling.css*.
  The generated html documentation has to be pushed onto the *gh-pages* branch in
  order to be hosted on github. To set this up, follow
  `this tutorial <https://daler.github.io/sphinxdoc-test/includeme.html>`_.

Usage convention
-------------

  Please cite C. Glaser, A. Nelles, I. Plaisier, C. Welling et al., "NuRadioReco: A reconstruction framework for radio neutrino detectors", Eur. Phys. J. C (2019) 79: 464, doi:10.1140/epjc/s10052-019-6971-5, arXiv:1903.07023 when using NuRadioReco.

  Should the code improve/change significantly, we will consider writing an
  updated publication. All people actively contributing to the main part of the
  code will be included in such a publication.

  Thank you for reading and for considering to contribute.
