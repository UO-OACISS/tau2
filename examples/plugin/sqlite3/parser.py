#!/usr/bin/env python3

import sqlite3
import itertools

have_graphviz = False
try:
    from graphviz import Digraph
    have_graphviz = True
except ImportError as e:
    pass  # module doesn't exist, deal with it.

def open_database():
    conn = sqlite3.connect('tauprofile.db')
    return conn

def get_trials(conn):
    c = conn.cursor()
    rows = c.execute('SELECT id, name, created FROM trial')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_metadata(conn):
    c = conn.cursor()
    rows = c.execute('SELECT trial, name, value FROM metadata')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_threads(conn):
    c = conn.cursor()
    rows = c.execute('SELECT id, node_rank, thread_rank FROM thread')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_metrics(conn):
    c = conn.cursor()
    rows = c.execute('SELECT trial, name FROM metric')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_timers(conn):
    c = conn.cursor()
    rows = c.execute('SELECT id, trial, parent, short_name FROM timer')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_timer_values(conn):
    c = conn.cursor()
    rows = c.execute('SELECT timer, metric, thread, value FROM timer_value')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_counters(conn):
    c = conn.cursor()
    rows = c.execute('SELECT id, trial, name FROM counter')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_counter_values(conn):
    c = conn.cursor()
    rows = c.execute('SELECT counter, timer, thread, sample_count, maximum_value, minimum_value, mean_value, sum_of_squares FROM counter_value')
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def write_callgraph(timers):
    dot = Digraph(comment='TAU Callgraph')
    dot.edges = []
    for timer in timers:
        if timer['trial'] == 1:
            if timer['id'] == 1 or timer['parent'] != None:
                dot.node(str(timer['id']), timer['short_name'])
            if timer['parent'] != None:
                dot.edge(str(timer['id']), str(timer['parent']))
    print(dot.source)
    #dot.render('callgraph.gv', view=True)

if __name__ == '__main__':
    conn = open_database()
    trials = get_trials(conn)
    #print(trials)
    metadata = get_metadata(conn)
    #print(metadata)
    threads = get_threads(conn)
    #print(threads)
    metrics = get_metrics(conn)
    #print(metrics)
    timers = get_timers(conn)
    for timer in timers:
        print(timer)
    timer_values = get_timer_values(conn)
    #print(timer_values)
    counters = get_counters(conn)
    #print(counters)
    counter_values = get_counter_values(conn)
    #print(counter_values)
    if have_graphviz:
        write_callgraph(timers)
