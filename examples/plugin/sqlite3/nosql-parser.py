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

"""
    NB: if you have pandas, you can load this whole view like this:
import pandas as pd
    df = pd.read_sql_query(sql, connex)
    df.head()
"""

def get_timers(conn):
    c = conn.cursor()
    sql = """select
        trial.id as trial_id,
        trial.name as trial_name,
        trial.created as trial_created,
        timer.id as timer_id,
        timer.parent as timer_parent,
        timer.name as timer_name,
        timer.short_name as timer_short_name,
        timer.timergroup as timer_timergroup,
        metric.name as metric_name,
        thread.node_rank as thread_node,
        thread.thread_rank as thread_thread,
        timer_value.value as timer_value_value
        from trial
        left outer join timer on timer.trial = trial_id
        left outer join metric on metric.trial = trial_id
        left outer join thread on thread.trial = trial_id
        left outer join timer_value on timer_value.timer = timer.id
            and timer_value.metric = metric.id
            and timer_value.thread = thread.id
        where timer_value_value is not null;"""
    rows = c.execute(sql)
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

def get_counters(conn):
    c = conn.cursor()
    sql = """select
        trial.id as trial_id,
        trial.name as trial_name,
        trial.created as trial_created,
        counter.id as counter_id,
        thread.node_rank as thread_node,
        thread.thread_rank as thread_thread,
        counter_value.timer as counter_value_context,
        counter_value.sample_count,
        counter_value.maximum_value,
        counter_value.minimum_value,
        counter_value.mean_value,
        counter_value.sum_of_squares
        from trial
        left outer join counter on counter.trial = trial_id
        left outer join thread on thread.trial = trial_id
        left outer join counter_value on counter_value.counter = counter_id
            and counter_value.thread = thread.id
        where counter_value_sample_count is not null
        and counter_value_sample_count > 0;"""
    rows = c.execute(sql)
    desc = c.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in c.fetchall()]
    return data

if __name__ == '__main__':
    conn = open_database()
    timers = get_timers(conn)
    for timer in timers:
        print(timer)
    counters = get_counters(conn)
    for counter in counters:
        print(counter)

