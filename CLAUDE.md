# Agentic AI prompt file

Always read and obey instructions in this file.

## General instructions
* Do not make assumptions. Stop and ask for clarification when required information cannot be discovered from local files, the SDK container or the paired DevKit. 
* Do not install any modules, libraries, packages or GStreamer plugins on the devkit except those specified in devkit.sh. Stage them so that they can be used for application build and execution. List all staged modules, libraries, packages or GStreamer plugins in a markdown file named staged_devkit.md. Overwrite any existing version of staged_devkit.md.
* Generate the minimum code that satisfies the task.
    * No features beyond what was asked.
    * No abstractions for single-use code.
    * No "flexibility" or "configurability" that wasn't requested.
    * No error handling for impossible scenarios.
* Do not modify any other scripts without asking for permission first.


