#!/usr/bin/env python3
"""TAU trial data for TAU Profile.x.y.z format profiles

Parses a set of TAU profile files and yields multi-indexed Pandas dataframes for the
interval and atomic events.
"""
from __future__ import print_function
import csv
import glob
import mmap
import os
import re
import xml.etree.ElementTree as ElementTree
from sys import stderr

import pandas
import sys


class TauProfileParser(object):
    """Parser for TAU's profile.* format."""

    _interval_header_re = re.compile(b'(\\d+) templated_functions_MULTI_(.+)')

    _atomic_header_re = re.compile(b'(\\d+) userevents')

    def __init__(self, trial, metric, metadata, indices, interval_data, atomic_events):
        self.trial = trial
        self.metric = metric
        self.metadata = metadata
        self.indices = indices
        self._interval_data = interval_data
        self._atomic_data = atomic_events

    def interval_data(self):
        return self._interval_data

    def atomic_data(self):
        return self._atomic_data

    def get_value_types(self):
        return [key for key in dict(self._interval_data.dtypes)
                if dict(self._interval_data.dtypes)[key] in ['float64', 'int64']]

    def summarize_samples(self, across_threads=False, callpaths=True):
        groups = 'Timer Name' if across_threads else ['Node', 'Context', 'Thread', 'Timer Name']
        if callpaths:
            base_data = self._interval_data.loc[self._interval_data['Group'].str.contains("TAU_SAMPLE")]
        else:
            base_data = self._interval_data.loc[self._interval_data['Timer Type'] == 'SAMPLE']
        summary = base_data.groupby(groups).sum()
        summary.index = summary.index.map(
            lambda x: '[SUMMARY] ' + x if across_threads else (x[0], x[1], x[2], '[SUMMARY] ' + x[3]))
        return summary

    def summarize_allocations(self):
        sums = self.atomic_data().groupby('Timer').agg({'Count': 'sum', 'Mean': 'mean'})
        allocs = sums[sums.index.to_series().str.contains('alloc')][['Count', 'Mean']]
        allocs['Total'] = allocs['Count'] * allocs['Mean']
        return allocs


    @classmethod
    def _parse_header(cls, fin):
        match = cls._interval_header_re.match(fin.readline())
        interval_count, metric = match.groups()
        return int(interval_count), metric

    @classmethod
    def _parse_metadata(cls, fin):
        fields, xml_wanabe = fin.readline().split(b'<metadata>')
        xml_wanabe = b'<metadata>' + xml_wanabe
        if (fields != b"# Name Calls Subrs Excl Incl ProfileCalls" and
                    fields != b'# Name Calls Subrs Excl Incl ProfileCalls # '):
            raise RuntimeError('Invalid profile file: %s' % fin.name)
        try:
            metadata_tree = ElementTree.fromstring(xml_wanabe)
        except ElementTree.ParseError as err:
            raise RuntimeError('Invalid profile file: %s' % err)
        metadata = {}
        for attribute in metadata_tree.iter('attribute'):
            name = attribute.find('name').text
            value = attribute.find('value').text
            metadata[name] = value
        return metadata

    @classmethod
    def _parse_interval_data(cls, fin, count):
        pass

    @classmethod
    def _parse_atomic_header(cls, fin):
        aggregates = fin.readline().split(b' aggregates')[0]
        if aggregates != b'0':
            print("aggregates != 0, is '%s'" % aggregates, file=stderr)
        match = cls._atomic_header_re.match(fin.readline())
        try:
            count = int(match.group(1))
            if fin.readline() != b"# eventname numevents max min mean sumsqr\n":
                raise RuntimeError('Invalid profile file: %s' % fin.name)
        except AttributeError:
            count = 0
        return count

    @staticmethod
    def extract_from_timer_name(name):
        import re
        tag_search = re.search('^\[(\w+)\]\s+(.*)', name)
        timer_type, rest = tag_search.groups() if tag_search else (None, name)
        name_search = re.search('(.+)\[({.*)\]', rest)
        func_name, location = name_search.groups() if name_search else (rest, None)
        return func_name, location, timer_type

    @classmethod
    def parse(cls, dir_path, filenames=None, trial=None):
        if not os.path.isdir(dir_path):
            print("Error: %s is not a directory." % dir_path, file=stderr)
            sys.exit(1)
        intervals = []
        atomics = []
        indices = []
        trial_data_metric = None
        trial_data_metadata = None
        if filenames is None:
            filenames = [os.path.basename(x) for x in glob.glob(os.path.join(dir_path, 'profile.*'))]
        if not filenames:
            print("Error: No profile files found.")
            sys.exit(1)
        for filename in sorted(filenames,
                               key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]):
            location = os.path.basename(filename).replace('profile.', '')
            node, context, thread = (int(x) for x in location.split('.'))
            file_path = os.path.join(dir_path, filename)
            with open(file_path) as fin:
                mm = mmap.mmap(fin.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
                interval_count, metric = cls._parse_header(mm)
                if not trial_data_metric:
                    trial_data_metric = metric
                metadata = cls._parse_metadata(mm)
                if not trial_data_metadata:
                    trial_data_metadata = metadata
                interval = pandas.read_csv(mm, nrows=interval_count, delim_whitespace=True,
                                             names=['Calls', 'Subcalls', 'Exclusive',
                                                    'Inclusive', 'ProfileCalls', 'Group'],
                                             engine='c')
                split_index = interval.reset_index()['index'].apply(cls.extract_from_timer_name)
                for n, col in enumerate(['Timer Name', 'Timer Location', 'Timer Type']):
                    interval[col] = split_index.apply(lambda l: l[n]).values
                mm.seek(0)
                for i in range(0, interval_count + 2):
                    mm.readline()
                cls._parse_atomic_header(mm)
                atomic = pandas.read_csv(mm, names=['Count', 'Maximum', 'Minimum', 'Mean', 'SumSq'],
                                           delim_whitespace=True, engine='c')
                mm.close()
                intervals.append(interval)
                atomics.append(atomic)
                indices.append((node, context, thread))
        interval_df = pandas.concat(intervals, keys=indices)
        interval_df.index.rename(['Node', 'Context', 'Thread', 'Timer'], inplace=True)
        atomic_df = pandas.concat(atomics, keys=indices)
        atomic_df.index.rename(['Node', 'Context', 'Thread', 'Timer'], inplace=True)
        return cls(trial, trial_data_metric, trial_data_metadata, indices, interval_df, atomic_df)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = '.'
    elif len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("Usage: %s [path]" % sys.argv[0])
        sys.exit(1)

    data = TauProfileParser.parse(path)
    print(data._interval_data.to_csv(quoting=csv.QUOTE_NONNUMERIC))
    print(data._atomic_data.to_csv(quoting=csv.QUOTE_NONNUMERIC))
