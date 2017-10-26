This example shows how to consolidate profiles from multiple TAU measurements
into one JSON file containing a workflow.  The required input file for this
functionality is a workflow_in.json file.  That file will some day be automatically
collected, but for now it is manually generated.  It contains:

 - Workflow Instance : a description of the workflow. Top level object.
 - Workflow Component : one or more components in the workflow
   - the name for each component should match the name of a directory of profiles,
     so that tau_prof2json.py can match up performance data with workflow components,
     and automatically generate the application and application-instance objects
     in the output JSON.
 - Workflow Component-input : one or more inputs to the workflow components.
   - each input for each component should be listed here.
 - Workflow Component-output : one or more outputs from the workflow components
   - each output for each component should be listed here.
 - Metric : the unit(s) of measurement for each profile
   - This can be extracted from the profiles, and should be.  It will be in the
     future.
 - Location : where the workflow was executed
   - The HPC system name.
 - Dependency Graph : This is the workflow dependency graph. Each workflow 
      component has a list of workflow components that it depends on. In 
      this case, xmain_0 has no dependencies, and reader2_0 has a 
      dependency on xmain_0.

In this example, xmain was called with 8 processes, and reader2 was called
with 4 processes.  The profiles from each application are in numbered 
profile directories.

To run the example, call it with the following arguments:

../../tools/src/tau_prof2json.py -w workflow_in.json xmain_0 reader2_0 -o merged_workflow.json

...or call it from your TAU bin installation directory.
