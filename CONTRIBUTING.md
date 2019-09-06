# Contributing guidelines for NuRadioReco

Thank you for considering to contribute to NuRadioReco.

NuRadioReco is intended as community based reconstruction software for radio (neutrino) detectors.
Everyone is invited to contribute and use the software for their experiment.

If you intend to contribute significantly to NuRadioReco, please contact @anelles and @cg-laser so that we
are informed about ongoing activities. Both will also be able to provide commit access to the repository.

* Workflow

If you work with NuRadioReco and you encounter a problem that you CANNOT solve, please file an issue.
Please provide as much information as you can.
Best, if you provide us with a minimal working example that reproduces the problem or refer to specific lines of code.

If you work with NuRadioReco and you encounter a problem that you CAN solve,
please provide your fix in a new branch and make a pull request.
We work with continous integration, so you will immediately see, whether your code causes siginifcant problems or changes existing results. This is of course no guarantee that every piece of code is correct.

The core team of developers will review your pull request as soon as possible and provide feedback.
Once approved, you can merge your code into the master and delete the branch. Allow for at least 24h review time between the last change (commit) and the merge even if the pull request was approved quickly to allow for additional comments or concerns.
Before merging the pull request, document the relevant changes in "changelog.txt". We use this information for our releases.

In general we try to follow 'industry' coding workflow conventions. So, if something is not explicitly mentioned here, please resort to 'best practices'.

* Coding conventions

NuRadioMC tries to follow PEP-8 coding conventions, https://www.python.org/dev/peps/pep-0008/

Please document your code extensively, especially the physics behind it. Each function should come with its doc string where all variables are defined.
All variables representing something with a unit, must make use of the NuRadioMC/utilities/units convention.
Please avoid adding additional dependencies without clearing this with the core developers.

* Usage convention

Please cite
C. Glaser, A. Nelles, I. Plaisier, C. Welling et al., "NuRadioReco: A reconstruction framework for radio neutrino detectors", Eur. Phys. J. C (2019) 79: 464, doi:10.1140/epjc/s10052-019-6971-5, arXiv:1903.07023
when using NuRadioReco.

Should the code improve/change significantly, we will consider writing an updated publication. All people actively contributing to the main part of the code will be included in such a publication.

Thank you for reading and for considering to contribute.


