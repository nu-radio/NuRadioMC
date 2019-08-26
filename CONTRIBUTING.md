Great that you want to contribute to NuRadioMC.

Please contact @cg-laser or @anelles to obtain permissions to contribute. 
You can of course also use the github tools to do so. 

* Workflow

If you work with NuRadioMC and you encounter a problem that you CANNOT solve, please file an issue. 
Please provide as much information as you can. 
Best, if you provide us with a minimal working example that reproduces the problem or refer to specific lines of code. 

If you work with NuRadioMC and you encounter a problem that you CAN solve, 
please provide your fix in a branch and make a pull request.
We works with continous integration, so you will immediately see, whether your code causes problems. 
The core team of developers will review your pull request as soon as possible and provide feedback. 
Once approved, you can merge your code into the master and delete the branch. Allow for at least 24h review time between the last change (commit) and the merge even if the pull request was approved quickly. 

This is following general coding workflow conventions. 

* Coding conventions

NuRadioMC tries to follow PEP-8 coding conventions, https://www.python.org/dev/peps/pep-0008/

Please document your code extensively, especially the physics behind it. Each function should come with its doc string where all variables are defined. 
All variables representing something with a unit, must make use of the NuRadioMC/utilities/units convention.
Please avoid adding additional dependencies without clearing this with the core developers. 

Thank you for reading and for considering to contribute. 

