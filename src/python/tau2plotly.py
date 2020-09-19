#!/usr/bin/env python3
import plotly.express as px
#import plotly.graph_obs as go
import pandas as pd
from mpi4py import MPI
import numpy as np
import adios2
import time
import argparse
from operator import add
import json
import io
from pathlib import Path
import os
import sys
if sys.version_info[0] < 3 or sys.version_info[1] < 3:
    raise Exception("Must be using Python 3.3 or newer.")

host_bbox = 'tight'
rank_bbox = 'tight'
top_x_bbox = 'tight'

def SetupArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instream", "-i", help="Name of the input stream", required=True)
    parser.add_argument("--config", "-c", help="Name of the config JSON file", default="charts.json")
    parser.add_argument("--nompi", "-nompi", help="ADIOS was installed without MPI", action="store_true")
    args = parser.parse_args()

    return args

def get_num_hosts(attr_info):
    names = {}
    # Iterate over the metadata, and get our hostnames.
    # Build a dictionary of unique values, if the value is
    # already there overwrite it.
    for key in attr_info:
        if "Hostname" in key:
            names[(attr_info[key]['Value'])] = 1
    return len(names)

def get_valid_ranks(attr_info):
    ranks_per_host = {}
    for key in attr_info:
        if "Hostname" in key:
            rank_id = int(key.split(':')[1])
            host_name = attr_info[key]['Value']
            if host_name in ranks_per_host:
                ranks_per_host[host_name].append(rank_id)
            else:
                ranks_per_host[host_name] = [rank_id,]
    valid_ranks = [min(ranks_per_host[host]) for host in ranks_per_host]
    return valid_ranks

# Get the tight bbox once per figure because it is slow
'''
def get_renderer_bbox(ax):
    fig = ax.get_figure()
    fig.canvas.print_svg(io.BytesIO())
    bbox = fig.get_tightbbox(fig._cachedRenderer).padded(0.35)
    return bbox
'''
# Format the output image file name
def get_image_filename(config):
    imgfile = ""
    if "filename" not in config:
        config["filename"] = config["name"].replace(" ", "-")
    imgfile = config["HTML output directory"]+"/"+config["filename"]+".html"

    return imgfile


# Build a dataframe that has per-node data for this timestep of the output data

def build_per_host_dataframe(fr_step, step, num_hosts, valid_ranks, config, charts):
    df = charts[config["name"]][1]
    # Read the number of ranks - check for the new method first
    num_ranks = 1
    if len(fr_step.read('num_ranks')) == 0:
        num_ranks = len(fr_step.read('num_threads'))
    else:
        num_ranks = fr_step.read('num_ranks')[0]

    rows = []
    # For each variable, get each MPI rank's data, some will be bogus (they didn't write it)
    for name in config["variables"]:
        rows.append(fr_step.read(name))
    if len(rows[0]) == 0:
        print("No rows!  Is TAU configured correctly?")
        return
    print("Processing dataframe...")
    # Now, transpose the matrix so that the rows are each rank, and the variables are columns
    df2 = pd.DataFrame(rows).transpose()
    # Add a name for each column
    if "labels" not in config:
        config["labels"] = config["components"]
    df2.columns = config["labels"]
    # Add the MPI rank column (each row is unique)
    df2['Node'] = range(0, len(df2))
    # Add the step column, all with the same value
    step_num = fr_step.read(config["Timestep for filename"])
    if step_num.size == 0:
        df2['step'] = 0
    else:
        df2['step']= int(step_num[0])
    # Filter out the rows that don't have valid data (keep only the lowest rank on each host)
    # This will filter out the bogus data
    df_trimmed = df2[df2['Node'].isin(valid_ranks)]
    df = df.append(df_trimmed)
    charts[config["name"]][1] = df
    return charts


# Build a dataframe that has per-rank data for this timestep of the output data

def build_per_rank_dataframe(fr_step, step, config, charts):
    df = charts[config["name"]][1]
    rows = []
    # For each variable, get each MPI rank's data
    for name in config["variables"]:
        rows.append(fr_step.read(name))
    if len(rows[0]) == 0:
        print("No rows!  Is TAU configured correctly?")
        return
    if config["error bar"]:
        std_rows = []
        for i in range(len(config["variables"])):
            name = config["variables"][i]
            events_name = name.replace(" / Mean", " / Num Events")
            num_events = fr_step.read(events_name)
            squares_name = name.replace(" / Mean", " / Sum Squares")
            sum_squares = fr_step.read(squares_name)

            std_rows.append((sum_squares/num_events)-(np.square(rows[i])))
        std_df = pd.DataFrame(std_rows).transpose()
    print("Processing dataframe...")
    # Now, transpose the matrix so that the rows are each rank, and the variables are columns
    df2 = pd.DataFrame(rows).transpose()
    print(df2)
    print(std_df)
    # Add a name for each column
    if "labels" not in config:
        config["labels"] = config["components"]
    df2.columns = config["labels"]
    # Add the MPI rank column (each row is unique)
    df2['Rank'] = range(0, len(df2))
    # Add the step column, all with the same value
    step_num = fr_step.read(config["Timestep for filename"])
    if step_num.size == 0:
        df2['step'] = 0
    else:
        df2['step']= int(step_num[0])
    df = df.append(df2)
    charts[config["name"]][1] = df

    return charts

# Build a dataframe for the top X timers

def build_topX_timers_dataframe(fr_step, step, config, charts):
    df = charts[config["name"]][1]
    #variables = fr_step.get_variable_names()
    totalTime = fr_step.read('.TAU application / Inclusive TIME')[0]
    variables = fr_step.available_variables()
    num_threads = fr_step.read('num_threads')[0]
    timer_data = {}
    # Get all timers
    #for name, _ in variables:
    for name in variables:
        if ".TAU application" in name:
            continue
        if "addr=" in name:
            continue
        if "Exclusive TIME" in name:
            shortname = name.replace(" / Exclusive TIME", "")
            timer_data[shortname] = []
            temp_vals = fr_step.read(name)
            for i in temp_vals:
                if i > totalTime:
                    timer_data[shortname].append(0)
                else:
                    timer_data[shortname].append(i)
    print("Processing dataframe...")
    df2 = pd.DataFrame(timer_data)
    # Get the mean of each column
    mean_series = df2.mean()
    # Get top X timers
    sorted_series = mean_series.sort_values(ascending=False)
    if "number of timers" not in config:
        config["number of timers"] = 5
    topX = config["number of timers"]
    topX_cols = sorted_series[:topX].axes[0].tolist()
    # Add all other timers together
    other_series = sorted_series[topX+1:].axes[0].tolist()
    df2["other"] = 0
    for other_col in other_series:
        df2[other_col].clip(lower=0)
        df2["other"] += df2[other_col]
    topX_cols.insert(0,"other")
    topX_cols.append("step")
    # Add the step column , all values the same
    step_num = fr_step.read(config["Timestep for filename"])
    if step_num.size == 0:
        df2['step'] = 0
    else:
        df2['step']= int(step_num[0])
    df = df.append(df2[topX_cols])
    # Add dataframe back to charts dictionary
    charts[config["name"]][1] = df
    return charts



# Generate the correct chart based on type
def generate_html(charts):
    for chart_name, data in charts.items():
        if "Timer" in chart_name:
            print("PLOTTING", chart_name)
            generate_timer_plot(data[1], data[2])
        elif data[0] == "bar":
            print("PLOTTING", chart_name)
            # Pass in the df and the config
            generate_bar_plot(data[1], data[2])
        else:
            print("PLOTTING", chart_name)
            generate_scatter_plot(data[1], data[2])


# Plot the dataframe once complete
def generate_bar_plot(df, config):
    print("Plotting...")
    #print(config)
    #print(df)
    fig = px.bar(df, x='Node',y=config['labels'],animation_frame="step")
    imgfile = get_image_filename(config)
    print("Writing...")
    fig.write_html(imgfile)
    print("done.")

def generate_scatter_plot(df, config):
    print("Plotting...")
    fig = px.scatter(df, x='Rank', y=config['labels'], animation_frame='step', log_y=True)
    imgfile = get_image_filename(config)
    print("Writing...")
    fig.write_html(imgfile)
    print("done.")

def generate_timer_plot(df, config):
    # Create default axes labels if they're not configured
    '''if "x axis" not in config:
        config["x axis"] = "Rank"
    if "y axis" not in config:
        config["y axis"] = "Time"'''
    print(df.columns)
    # trim any column names to correct length
    if "max label length" not in config:
        config["max label length"] = 30 # default value if not set
    '''labels={}
    for col in df.columns:
        if len(col) > config["max label length"]:
            label = col[:(config["max label length"]-3)]+'...'
        else:
            label = col
        labels[col] = label
    print(labels)'''
    #df.rename(str[0:config["max label length"]], axis='columns')
    short_names = [col[0:(config["max label length"]-3)]+'...' if len(col)>config["max label length"] else col for col in df.columns]
    name_set = set()
    counter = 0
    for i in range(len(short_names)):
        if short_names[i] in name_set:
            new_name = (short_names[i])[:(config["max label length"]-1)]+str(counter)
            short_names[i] = new_name
            counter+=1
        else:
            name_set.add(short_names[i])
    df.columns = short_names
    print(df.columns)
    print("Plotting...")
    #print(df)
    fig = px.bar(df, animation_frame="step")
    #fig.update_layout(legend=dict(yanchor="top", y=-0.75 , xanchor="left", x=0.01))
    imgfile = get_image_filename(config)
    print("Writing...")
    fig.write_html(imgfile)
    print("done.")


# Process the ADIOS2 file

def process_file(args):
    with open(args.config) as config_data:
        config = json.load(config_data)

    # make the output directory
    if "HTML output directory" not in config or config["HTML output directory"] == ".":
        config["HTML output directory"] = os.getcwd()
    else:
        Path(config["HTML output directory"]).mkdir(parents=True, exist_ok=True)

    if "Timestep for filename" not in config:
        config["Timestep for filename"] = "default"

    for f in config["figures"]:
        if "HTML output directory" not in f or f["HTML output directory"] == ".":
            f["HTML output directory"] = config["HTML output directory"]
        else:
            Path(config["HTML output directory"]).mkdir(parents=True, exist_ok=True)
        if "Timestep for filename" not in f:
             f["Timestep for filename"] = config["Timestep for filename"]

    filename = args.instream
    print ("Opening:", filename)
    if not args.nompi:
        fr = adios2.open(filename, "r", MPI.COMM_SELF, "adios2.xml", "TAUProfileOutput")
    else:
        fr = adios2.open(filename, "r", config["ADIOS2 config file"], "TAUProfileOutput")
    # Get the attributes (simple name/value pairs)
    attr_info = fr.available_attributes()
    # Get the unique host names from the attributes
    num_hosts = get_num_hosts(attr_info)
    cur_step = 0
    # Make the dictionary we can add all the charts we want
    charts = {}
    for fr_step in fr:
        cur_step = fr_step.current_step()
        print(filename, "Step = ", cur_step)
        for f in config["figures"]:
            print(f["name"])
            if "Timer" in f["name"]:
                if f["name"] in charts:
                    charts = build_topX_timers_dataframe(fr_step, cur_step, f, charts)
                else:
                    timer_df = pd.DataFrame()
                    charts[f["name"]] = ["bar", timer_df, f]
                    charts = build_topX_timers_dataframe(fr_step, cur_step, f, charts)
            elif f["granularity"] == "node":
                valid_ranks = get_valid_ranks(attr_info)
                if f["name"] in charts:
                    charts = build_per_host_dataframe(fr_step, cur_step, num_hosts, valid_ranks, f, charts)
                else:
                    host_df = pd.DataFrame()
                    charts[f["name"]] = ["bar", host_df, f]
                    charts = build_per_host_dataframe(fr_step, cur_step, num_hosts, valid_ranks, f, charts)
            else:
                if f["name"] in charts:
                    charts = build_per_rank_dataframe(fr_step, cur_step, f, charts)
                else:
                    rank_df = pd.DataFrame()
                    charts[f["name"]] = ["scatter", rank_df, f]
                    charts = build_per_rank_dataframe(fr_step, cur_step, f, charts)

        fr.end_step()
    # Now that all timesteps have been processed we can generate the charts
    generate_html(charts)



if __name__ == '__main__':
    args = SetupArgs()
    begin_time = time.time()
    process_file(args)
    total_time = time.time() - begin_time
    print(f"Processed file in {total_time} seconds", flush=True)
