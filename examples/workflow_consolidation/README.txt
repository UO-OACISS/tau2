This example shows how to consolidate profiles from multiple TAU measurements
into one JSON file containing a workflow.  The required input file for this
functionality is a workflow.json file.  That file will some day be automatically
collected, but for now it is manually generated.  It contains 
 - Workflow Instance : a description of the workflow 
 - Workflow Component : one or more components in the workflow
 - Workflow Component-input : one or more inputs to the workflow components
 - Workflow Component-output : one or more outputs from the workflow components
 - Metric : the unit(s) of measurement for each profile
 - Location : where the workflow was executed
 - Dependency Graph : This is the workflow dependency graph. Each workflow 
      component has a list of workflow components that it depends on. In 
      this case, xmain_0 has no dependencies, and reader2_0 has a 
      dependency on xmain_0.

In this example, xmain was called with 8 processes, and reader2 was called
with 4 processes.  The profiles from each application are in numbered 
profile directories.

To run the example, call it with the following arguments:

./tau_prof2json.py -w workflow.json xmain_0 reader2_0 -o foo.json
