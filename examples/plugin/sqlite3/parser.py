#!/usr/bin/env python3

import sqlite3
import itertools

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
    rows = c.execute('SELECT id, trial, parent, short_name, name FROM timer')
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
    #print(timers)
    timer_values = get_timer_values(conn)
    #print(timer_values)
    counters = get_counters(conn)
    #print(counters)
    counter_values = get_counter_values(conn)
    #print(counter_values)

