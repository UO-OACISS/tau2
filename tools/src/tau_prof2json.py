#!/usr/bin/env python

"""A python script to convert TAU profiles to JSON
"""

from __future__ import print_function
import os
import sys
import argparse
import json
from collections import OrderedDict

workflow_metadata_str = "Workflow metadata"
metadata_str = "metadata"
global_data = None
have_workflow_file = False
group_totals = None
group_counts = None
workflow_start = 0
workflow_end = 0

"""
    Parse the "invalid" TAU XML
"""
def parse_tau_xml(instring):
    mydict = OrderedDict()
    tmp = instring
    metadata_tokens = tmp.split("<metadata>",1)
    metadata_tokens = metadata_tokens[1].split("</metadata>",1)
    tmp = metadata_tokens[0]
    while len(tmp) > 0:
        attribute_tokens = tmp.split("<attribute>",1)
        attribute_tokens = attribute_tokens[1].split("</attribute>",1)
        attribute = attribute_tokens[0]
        tmp = attribute_tokens[1]
        name_tokens = attribute.split("<name>",1)
        name_tokens = name_tokens[1].split("</name>",1)
        name = name_tokens[0]
        value_tokens = name_tokens[1].split("<value>",1)
        value_tokens = value_tokens[1].split("</value>",1)
        value = value_tokens[0]
        # print(name,value)
        if name == "Metric Name":
            continue
        mydict[name] = value
        if tmp == "</metadata>":
            break
    return mydict       

"""
This method will parse the arguments.
"""
def parse_args(arguments):
    global workflow_metadata_str
    global metadata_str
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('indir', help="One or more input directories (can be raw TAU profiles or generated from tau_coalesce)", nargs='+')
    parser.add_argument('-w', '--workflow', help="Workflow Metadata file (JSON)",
                        default=None)
    parser.add_argument('-o', '--outfile', help="Output file instead of print to stdout (optional)",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)
    # print(args)
    return args

def extract_workflow_metadata(component_name, thread_name, max_inclusive):
    global global_data
    global have_workflow_file
    global workflow_metadata_str
    global workflow_start
    global workflow_end
    # This is the root rank/thread for an application, so extract some key info
    workflow_dict = global_data[workflow_metadata_str]
    app_dict = global_data[component_name][metadata_str][thread_name]
    start_time_stamp = app_dict["Starting Timestamp"]
    end_time_stamp = None
    if "Ending Timestamp" not in app_dict:
        end_time_stamp = str(long(start_time_stamp) + max_inclusive)
    else:
        end_time_stamp = app_dict["Ending Timestamp"]
    if workflow_start == 0 or workflow_start > long(start_time_stamp):
        workflow_start = long(start_time_stamp)
    if workflow_end == 0 or workflow_end < long(end_time_stamp):
        workflow_end = long(end_time_stamp)
    local_time = app_dict["Local Time"]
    if start_time_stamp != None and have_workflow_file:
        for wc in workflow_dict["Workflow Component"]:
            if wc["name"] == component_name:
                wc["start-timestamp"] = start_time_stamp
                wc["end-timestamp"] = end_time_stamp
                wc["Local-Time"] = local_time
        for wc in workflow_dict["Workflow Instance"]:
            if "timestamp" not in wc or wc["start-timestamp"] > start_time_stamp:
                wc["start-timestamp"] = start_time_stamp
                wc["end-timestamp"] = end_time_stamp
                wc["Local-Time"] = local_time
    app_name = app_dict["Executable"]
    found = False
    app_id = 0
    for app in workflow_dict["Application"]:
        if app["name"] in component_name or app["name"] == app_name:
            Found = True
            app_id = app["id"]
            break
    if not found:
        app = OrderedDict()
        app_id = len(workflow_dict["Application"]) + 1
        app["id"] = app_id
        app["location-id"] = 1
        app["name"] = app_name
        app["version"] = "0.1"
        app["uri"] = ""
    	workflow_dict["Application"].append(app)
    app_instance = OrderedDict()
    app_instance["id"] = len(workflow_dict["Application-instance"]) + 1
    app_instance["process-id"] = app_dict["pid"]
    app_instance["application-id"] = app_id
    app_instance["event-type"] = ""
    app_instance["start-timestamp"] = start_time_stamp
    app_instance["end-timestamp"] = end_time_stamp
    app_instance["Local-Time"] = local_time
    workflow_dict["Application-instance"].append(app_instance)
    index = 0
    while True:
        name = "posix open[" + str(index) + "]"
        if name in app_dict:
            json_str = app_dict[name]
        else:
            break
        index = index + 1
    # add workflow instance data to the global metadata section
    if not have_workflow_file:
        workflow_component = OrderedDict()
        workflow_component["id"] = app_id
        workflow_component["name"] = app_name
        workflow_component["location-id"] = 1
        workflow_component["application-id"] = app_id
        workflow_component["start-timestamp"] = start_time_stamp
        workflow_component["end-timestamp"] = end_time_stamp
        workflow_component["Local-Time"] = local_time
        workflow_dict["Workflow Component"].append(workflow_component)
    return workflow_component

def parse_functions(node, context, thread, infile, data, num_functions, function_map):
    global group_totals
    global group_counts
    max_inclusive = 0
    # Function looks like:
    # "".TAU application" 1 0 8626018 8626018 0 GROUP="TAU_USER""
    if num_functions > 0:
        """
        if "Timers" not in data:
            data["Timers"] = OrderedDict()
        if "Timer Measurement Columns" not in data:
            data["Timer Measurement Columns"] = list()
            data["Timer Measurement Columns"].append("Process Index")
            #data["Timer Measurement Columns"].append("Context Index")
            data["Timer Measurement Columns"].append("Thread Index")
            data["Timer Measurement Columns"].append("Function Index")
            data["Timer Measurement Columns"].append("Calls")
            data["Timer Measurement Columns"].append("Subroutines")
            data["Timer Measurement Columns"].append("Exclusive TIME")
            data["Timer Measurement Columns"].append("Inclusive TIME")
        """
        if "Timers" not in data:
            data["Timers"] = list()
        for i in range(0,num_functions):
            line = infile.readline()
            tokens = line.split("\"",2)
            function_name = tokens[1].strip()
            stats = tokens[2]
            """
            if function_name not in function_map:
                #data["Timers"][function_name] = len(data["Timers"])
                function_map[function_name] = len(data["Timers"])
                data["Timers"][len(data["Timers"])] = function_name
            """
            # split the stats
            tokens = stats.strip().split(" ", 6)
            """
            timer = list()
            timer.append(int(node))
            #timer.append(int(context))
            timer.append(int(thread))
            # function index
            timer.append(function_map[function_name])
            # calls
            timer.append(long(tokens[0]))
            # Subroutines
            timer.append(long(tokens[1]))
            # Exclusive
            timer.append(long(tokens[2]))
            # Inclusive
            timer.append(long(tokens[3]))
            # Profile Calls
            #timer.append(long(tokens[4]))
            data["Timer Measurements"].append(timer)
            """
            timer = OrderedDict()
            timer["process index"] = int(node)
            timer["thread index"] = int(thread)
            timer["Function"] = function_name
            timer["Calls"] = long(tokens[0])
            timer["Subroutines"] = long(tokens[1])
            timer["Exclusive Time"] = long(tokens[2])
            timer["Inclusive Time"] = long(tokens[3])
            group = tokens[5]
            # handle the ADIOS special case
            if "ADIOS" in function_name:
                if "ADIOS" not in group_totals:
                    group_totals["ADIOS"] = long(tokens[2])
                    group_counts["ADIOS"] = long(tokens[0])
                else:
                    group_totals["ADIOS"] = group_totals["ADIOS"] + long(tokens[2])
                    group_counts["ADIOS"] = group_counts["ADIOS"] + long(tokens[0])
            else:
                if group not in group_totals:
                    group_totals[group] = long(tokens[2])
                    group_counts[group] = long(tokens[0])
                else:
                    group_totals[group] = group_totals[group] + long(tokens[2])
                    group_counts[group] = group_counts[group] + long(tokens[0])
            data["Timers"].append(timer)
            if max_inclusive < long(tokens[3]):
                max_inclusive = long(tokens[3])
    return max_inclusive

def extract_group_totals():
    global group_totals
    global group_counts
    global global_data
    tmp = global_data["Workflow metadata"]["Workflow Component"]
    application_metadata = tmp[len(tmp) - 1]
    threads = group_totals["threads"]
    application_metadata["Processes"] = threads
    comm_calls = 0
    comm_time = 0
    io_time = 0
    adios_time = 0
    user_time = 0
    collective_bytes = 0
    send_bytes = 0
    recv_bytes = 0
    read_bytes = 0
    write_bytes = 0
    adios_bytes = 0
    for key in group_counts:
        if key.find("MPI") != -1:
            comm_calls = comm_calls + group_counts[key]
    for key in group_totals:
        if key.find("MPI") != -1:
            comm_time = comm_time + group_totals[key]
        if key.find("TAU_IO") != -1:
            io_time = io_time + group_totals[key]
        # ADIOS is a special case, because it doesn't have a group - yet
        # It is also in the TAU_DEFAULT and/or TAU_USER group.
        if key.find("ADIOS") != -1:
            adios_time = adios_time + group_totals[key]
        elif key.find("TAU_USER") != -1:
            user_time = user_time + group_totals[key]
        elif key.find("TAU_DEFAULT") != -1:
            user_time = user_time + group_totals[key]
        if key.find("Collective_Bytes") != -1:
            collective_bytes = collective_bytes + group_totals[key]
        if key.find("Send_Bytes") != -1:
            send_bytes = send_bytes + group_totals[key]
        if key.find("Recv_Bytes") != -1:
            recv_bytes = recv_bytes + group_totals[key]
        if key.find("Read_Bytes") != -1:
            read_bytes = read_bytes + group_totals[key]
        if key.find("Write_Bytes") != -1:
            write_bytes = write_bytes + group_totals[key]
        if key.find("ADIOS_data_size") != -1:
            adios_bytes = adios_bytes + group_totals[key]
    if comm_calls > 0:
        application_metadata["aggr_communication_calls"] = comm_calls
    if comm_time > 0:
        application_metadata["aggr_communication_time"] = comm_time/threads
    if adios_time > 0:
        application_metadata["aggr_adios_time"] = adios_time/threads
    if io_time > 0:
        application_metadata["aggr_io_time"] = io_time/threads
    if user_time > 0:
        application_metadata["aggr_execution_time"] = user_time/threads
    if collective_bytes > 0:
        application_metadata["aggr_communication_collective_bytes"] = collective_bytes
    if send_bytes > 0:
        application_metadata["aggr_communication_sent_bytes"] = send_bytes
    if recv_bytes > 0:
        application_metadata["aggr_communication_recv_bytes"] = send_bytes
    if adios_bytes > 0:
        application_metadata["aggr_adios_bytes"] = adios_bytes
    if read_bytes > 0:
        application_metadata["aggr_io_read_bytes"] = read_bytes
    if write_bytes > 0:
        application_metadata["aggr_io_write_bytes"] = write_bytes
    application_metadata["total_time"] = long(application_metadata["end-timestamp"]) - long(application_metadata["start-timestamp"])

def parse_aggregates(node, context, thread, infile, data):
    aggregates = infile.readline()
    tokens = aggregates.split()
    num_aggregates = int(tokens[0])
    # data["Num Aggregates"] = tokens[0]
    for i in range(1,num_aggregates):
        # do nothing
        line = infile.readline()

def parse_counters(node, context, thread, infile, data, counter_map):
    global group_totals
    userevents = infile.readline()
    if (userevents == None or userevents == ""):
        return
    tokens = userevents.split()
    num_userevents = int(tokens[0])
    # data["Num Counters"] = tokens[0]
    # ignore the header line
    line = infile.readline()
    for i in range(1,num_userevents):
        """
        if "Counters" not in data:
            data["Counters"] = OrderedDict()
        if "Counter Measurement Columns" not in data:
            data["Counter Measurement Columns"] = list()
            data["Counter Measurement Columns"].append("Process Index")
            #data["Counter Measurement Columns"].append("Context Index")
            data["Counter Measurement Columns"].append("Thread Index")
            data["Counter Measurement Columns"].append("Function Index")
            data["Counter Measurement Columns"].append("Sample Count")
            data["Counter Measurement Columns"].append("Maximum")
            data["Counter Measurement Columns"].append("Minimum")
            data["Counter Measurement Columns"].append("Mean")
            data["Counter Measurement Columns"].append("SumSqr")
        """
        if "Counters" not in data:
            data["Counters"] = list()
        for i in range(0,num_userevents):
            line = infile.readline()
            line = line.strip()
            if len(line) == 0:
                break
            tokens = line.split("\"",2)
            counter_name = tokens[1].strip()
            stats = tokens[2]
            if counter_name == "(null)":
                continue
            """
            if counter_name not in counter_map:
                #data["Counters"][counter_name] = len(data["Counters"])
                counter_map[counter_name] = len(data["Counters"])
                data["Counters"][len(data["Counters"])] = counter_name
            """
            # split the stats
            tokens = stats.strip().split(" ", 5)
            """
            counter = list()
            counter.append(int(node))
            counter.append(int(context))
            counter.append(int(thread))
            # function index
            counter.append(counter_map[counter_name])
            # numevents
            counter.append(float(tokens[0]))
            # max
            counter.append(float(tokens[1]))
            # min
            counter.append(float(tokens[2]))
            # mean
            counter.append(float(tokens[3]))
            # sumsqr
            counter.append(float(tokens[4]))
            data["Counter Measurements"].append(counter)
            """
            counter = OrderedDict()
            counter["process index"] = int(node)
            counter["thread index"] = int(thread)
            counter["Counter"] = counter_name
            counter["Num Events"] = long(tokens[0])
            counter["Max Value"] = float(tokens[1])
            counter["Min Value"] = float(tokens[2])
            counter["Mean Value"] = float(tokens[3])
            counter["SumSqr Value"] = float(tokens[4])
            if "Message size for " in counter_name:
                value = long(tokens[0]) * float(tokens[3])
                c_name = "Collective_Bytes"
                if c_name not in group_totals:
                    group_totals[c_name] = value
                else:
                    group_totals[c_name] = group_totals[c_name] + value
            if counter_name == "Message size sent to all nodes":
                value = long(tokens[0]) * float(tokens[3])
                c_name = "Send_Bytes"
                if c_name not in group_totals:
                    group_totals[c_name] = value
                else:
                    group_totals[c_name] = group_totals[c_name] + value
            if counter_name == "Message size received from all nodes":
                value = long(tokens[0]) * float(tokens[3])
                c_name = "Recv_Bytes"
                if c_name not in group_totals:
                    group_totals[c_name] = value
                else:
                    group_totals[c_name] = group_totals[c_name] + value
            if counter_name == "Bytes Read":
                value = long(tokens[0]) * float(tokens[3])
                c_name = "Read_Bytes"
                if c_name not in group_totals:
                    group_totals[c_name] = value
                else:
                    group_totals[c_name] = group_totals[c_name] + value
            if counter_name == "Bytes Written":
                value = long(tokens[0]) * float(tokens[3])
                c_name = "Write_Bytes"
                if c_name not in group_totals:
                    group_totals[c_name] = value
                else:
                    group_totals[c_name] = group_totals[c_name] + value
            if counter_name == "ADIOS data size":
                value = long(tokens[0]) * float(tokens[3])
                c_name = "ADIOS_data_size"
                if c_name not in group_totals:
                    group_totals[c_name] = value
                else:
                    group_totals[c_name] = group_totals[c_name] + value
            data["Counters"].append(counter)

def parse_profile(indir, profile, application_metadata, function_map, counter_map):
    global metadata_str
    global global_data
    group_totals["threads"] = group_totals["threads"] + 1
    # FIrst, split the name of the profile to get the node, context, thread
    tokens = profile.split(".")
    node = tokens[1]
    context = tokens[2]
    thread = tokens[3]
    # Open the file, get the first line
    infile = open(os.path.join(indir, profile), "r")
    functions = infile.readline()
    # First line looks like:
    # "16 templated_functions_MULTI_TIME"
    tokens = functions.split()
    num_functions = int(tokens[0])
    # application_metadata["Metric"] = tokens[1]
    # The header for the functions looks like this:
    # "# Name Calls Subrs Excl Incl ProfileCalls # <metadata>...</metadata>"
    header = infile.readline()
    tokens = header.split("#",2)
    # Parse the metadata
    thread_name = "Rank " + str(node) + ", Thread " + str(thread)
    application_metadata[metadata_str][thread_name] = parse_tau_xml(str.strip(tokens[2]))
    # Parse the functions
    max_inclusive = parse_functions(node, context, thread, infile, application_metadata, num_functions, function_map)
    if node == "0" and thread == "0":
        extract_workflow_metadata(indir, thread_name, max_inclusive)
    # Parse the aggregates
    parse_aggregates(node, context, thread, infile, application_metadata)
    # Parse the counters
    parse_counters(node, context, thread, infile, application_metadata, counter_map)

"""
This method will parse a directory of TAU profiles
"""
def parse_directory(indir, index):
    global metadata_str
    global global_data
    global group_totals
    global group_counts
    # assume just 1 metric for now...

    # create a dictionary for this application
    application_metadata = OrderedDict()
    #application["source directory"] = indir

    # add the application to the master dictionary
    # tmp = "Application " + str(index)
    #data[tmp] = application
    global_data[indir] = application_metadata
    
    # get the list of profile files
    profiles = [f for f in os.listdir(indir) if (os.path.isfile(os.path.join (indir, f)) and f.startswith("profile."))]
    #application["num profiles"] = len(profiles)

    # sort the profiles alphanumerically
    profiles.sort()

    application_metadata[metadata_str] = OrderedDict()
    function_map = {}
    counter_map = {}
    group_totals = OrderedDict()
    group_counts = OrderedDict()
    group_totals["threads"] = 0
    for p in profiles:
        parse_profile(indir, p, application_metadata, function_map, counter_map)
    extract_group_totals()

def make_workflow_instance():
    instance = OrderedDict()
    instance["id"] = 1
    instance["name"] = "Workflow Instance"
    instance["location-id"] = 1
    instance["version"] = ""
    return instance

def write_metrics():
    global global_data
    # assuming 1 metric for now
    metrics = []
    metric = OrderedDict()
    metric["id"] = 1
    metric["location-id"] = 1
    metric["measurement"] = "Time"
    metric["units"] = "microseconds"
    metric["description"] = "This is a metric used to measure the applications."
    metrics.append(metric)
    global_data["Workflow metadata"]["Metrics"] = metrics

"""
Main method
"""
def main(arguments):
    global workflow_metadata_str
    global global_data
    global have_workflow_file
    global workflow_start
    global workflow_end
    # parse the arguments
    args = parse_args(arguments)
    global_data = OrderedDict()
    if args.workflow != None:
        global_data[workflow_metadata_str] = json.load(open(args.workflow), object_pairs_hook=OrderedDict)
        have_workflow_file = True
    else:
        global_data[workflow_metadata_str] = OrderedDict()
        global_data[workflow_metadata_str]["Workflow Instance"] = make_workflow_instance()
        global_data[workflow_metadata_str]["Workflow Component"] = []
    global_data[workflow_metadata_str]["Application"] = []
    global_data[workflow_metadata_str]["Application-instance"] = []

    index = 1
    for indir in args.indir:
        print ("Processing:", indir)
        parse_directory(indir, index)
        index = index + 1

    # write the main workflow instance metadata
    global_data[workflow_metadata_str]["Workflow Instance"]["execution_time"] = workflow_end - workflow_start
    write_metrics()

    # write the JSON output
    if args.outfile == None:
        # json.dump(global_data, args.outfile)
        json.dumps(global_data, indent=2)
    else:
        # json.dump(global_data, args.outfile)
        args.outfile.write(json.dumps(global_data, indent=2))

"""
Call the main function if not called from another python file
"""
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

