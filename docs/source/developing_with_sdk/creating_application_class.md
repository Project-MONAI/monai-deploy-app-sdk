# Creating Application class

The Application class is perhaps the most important class that MONAI Deploy App developers will interact with.  A developer will inherit a new Application from the monai.core.Application base class. The base application class provides support for chaining up operators, as well as a mechanism to execute the application. The compose method of this class needs to be implemented in the inherited class to instantiate Operators and connect them together to form a Directed Acyclic Graph.
