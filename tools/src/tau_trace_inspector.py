#!/usr/bin/env python3

import os
import sys
import re
from enum import Enum
import json
import struct
import argparse


def ptime(time):
    if not isinstance(time,float):
        time = float(time)

    # Convert to seconds
    time /= 1e6

    ret = ""

    if 3600 < time:
        h = int(time / 3600)
        time = time - h * 3600
        ret = ret + f"{h}h"

    if 60 < time:
        m = int(time / 60)
        time = time - m * 60
        ret = ret + f"{m}m"

    if time:
        ret = ret + f"{time}s"

    return ret

def psize(size):
    if not isinstance(size,float):
        size = float(size)

    if (1024*1024*1024) < size:
        gb = (size / (1024*1024*1024))
        return f"{gb} GB"

    if (1024*1024) < size:
        mb = (size / (1024*1024))
        return f"{mb} MB"

    if 1024 < size:
        kb = (size / (1024))
        return f"{kb} KB"

    return f"{size} B"



class TauTraceFormat(Enum):
    TAU_FORMAT_UNDEFINED = 0
    """
    /* event record description */
    typedef struct {
    x_int32  ev;    /* event id                    */
    x_uint16 nid;   /* node id                     */
    x_uint16 tid;   /* thread id                   */
    x_int64  par;   /* event parameter             */
    x_uint64 ti;    /* timestamp (in microseconds) */
    } TAU_EV;
    """
    TAU_FORMAT_NATIVE= 1
    """
    /* for 32 bit platforms */
    typedef struct {
    x_int32            ev;    /* -- event id        -- */
    x_uint16           nid;   /* -- node id         -- */
    x_uint16           tid;   /* -- thread id       -- */
    x_int64            par;   /* -- event parameter -- */
    x_uint64           ti;    /* -- time [us]?      -- */
    } TAU_EV32;
    """
    TAU_FORMAT_32 = 2
    # Same with opposite endianness
    TAU_FORMAT_32_SWAP = 3
    """
    /* for 64 bit platforms */
    typedef struct {
    x_int64            ev;    /* -- event id        -- */
    x_uint16           nid;   /* -- node id         -- */
    x_uint16           tid;   /* -- thread id       -- */
    x_uint32           padding; /*  space wasted for 8-byte aligning the next item */ 
    x_int64            par;   /* -- event parameter -- */
    x_uint64           ti;    /* -- time [us]?      -- */
    } TAU_EV64;
    """
    TAU_FORMAT_64 = 4
    # Same with opposite endianness
    TAU_FORMAT_64_SWAP = 5

class TauRawTraceEvent():
    def __init__(self, ev, nid, tid, par, ti, padding=0):
        self.ev = ev
        self.nid = nid
        self.tid = tid
        self.par = par
        self.ti = ti 
        self.padding = padding

    def ctx(self):
        return "{} @ {}.{}".format(self.ti/1e6, self.nid, self.tid)

    def attrs(self):
        return {"nid": self.nid, "tid" : self.tid, "ti" : self.ti, "value" : self.par}




class TauEventDecoder():
    
    def __init__(self, trace_file):
        self.trace_file = trace_file

        bo = sys.byteorder

        my_modifier = ">" if bo == "big" else "<"
        sw_modifier = "<" if bo == "big" else  ">"

        ev32 = "iHHqQ"
        my_32 = my_modifier + ev32
        sw_32 = sw_modifier + ev32

        ev64 = "qHHIqQ"
        my_64 = my_modifier + ev64
        sw_64 = sw_modifier + ev64

        native = "@iHHqQ"

        self._packs = { 
            TauTraceFormat.TAU_FORMAT_32 : my_32,
            TauTraceFormat.TAU_FORMAT_32_SWAP : sw_32,
            TauTraceFormat.TAU_FORMAT_64 : my_64,
            TauTraceFormat.TAU_FORMAT_64_SWAP : sw_64,
            TauTraceFormat.TAU_FORMAT_NATIVE : native
        }

        self._sizes = {
            TauTraceFormat.TAU_FORMAT_32 : struct.calcsize(my_32),
            TauTraceFormat.TAU_FORMAT_32_SWAP : struct.calcsize(sw_32),
            TauTraceFormat.TAU_FORMAT_64 : struct.calcsize(my_64),
            TauTraceFormat.TAU_FORMAT_64_SWAP : struct.calcsize(sw_64),
            TauTraceFormat.TAU_FORMAT_NATIVE : struct.calcsize(native)
        }

        self._first_ts = 0

        self.fmt = self._get_format(trace_file)

    def get_ev_size(self, tformat):
        return self._sizes.get(tformat)

    def get_ev_packing(self, tformat):
        return self._packs.get(tformat)

    def _par_offset(self, tformat):
        off = { 
            TauTraceFormat.TAU_FORMAT_32 : 3,
            TauTraceFormat.TAU_FORMAT_32_SWAP : 3,
            TauTraceFormat.TAU_FORMAT_64 : 4,
            TauTraceFormat.TAU_FORMAT_64_SWAP : 4,
            TauTraceFormat.TAU_FORMAT_NATIVE : 3
        }
        return off.get(tformat)

    def _get_format(self, trace_file):
        # Read enough to cover the largest type
        with open(trace_file, "rb") as f:
            trc_first_32_bytes = f.read(max([x for x in self._sizes.values()]))

        candidates = [
            TauTraceFormat.TAU_FORMAT_NATIVE,
            TauTraceFormat.TAU_FORMAT_64,
            TauTraceFormat.TAU_FORMAT_32,
            TauTraceFormat.TAU_FORMAT_32_SWAP,
            TauTraceFormat.TAU_FORMAT_64_SWAP
        ]

        for f in candidates:
            #print("IN L {} EXP {} OFF {}".format(len(trc_first_32_bytes),
            #                                     self.get_ev_size(f),
            #                                     self._par_offset(f)))
            data = struct.unpack(self.get_ev_packing(f),
                                 trc_first_32_bytes[:self.get_ev_size(f)])

            #print(data)
            if not data:
                continue

            par_off = self._par_offset(f)

            if data[par_off] == 3:
                self._first_ts = data[par_off + 1]
                return f

        raise Exception("Could not infer trace format")

    def read_trace(self, callback, evid=None, nid=None, tid=None):
        ev_size = self.get_ev_size(self.fmt)
        par_off = self._par_offset(self.fmt)

        if evid:
            # Make sure events IDs are all ints
            evid = [ int(x) for x in evid ]

        with open(self.trace_file, "rb") as f:
            # Skip first event used for detection
            f.seek(ev_size)

            while True:
                buff = f.read(10000 * ev_size)

                if not buff:
                    break

                for e in struct.iter_unpack(self.get_ev_packing(self.fmt), buff):
                    skip = 0
                    if evid is not None:
                        if e[0] not in evid:
                            continue

                    if nid is not None:
                        if nid != e[1]:
                            continue

                    if tid is not None:
                        if tid != e[2]:
                            continue

                    rev = TauRawTraceEvent(e[0], e[1], e[2], e[par_off], e[par_off + 1] - self._first_ts)
                    callback(rev)


class TauEventType(Enum):
    UNDEFINED = 0
    TRIGGERVALUE = 1
    ENTRYEXIT = 2

    @classmethod
    def parse(cls, str):
        if str == "TriggerValue" or str == "par":
            return cls.TRIGGERVALUE
        elif str == "EntryExit":
            return cls.ENTRYEXIT
        else:
            return cls.UNDEFINED

LOCUS_RE = re.compile(r"(.*) \[\{(.*)\} \{(.*)\}\]")
STACK_RE = re.compile(r"(.*) : (.*)")
EXTENDED_ATTR_RE = re.compile(r"(.*) \| (\{.*\})")
VALUE_ATTR_RE = re.compile(r"(.*) \| (.*)")
BRACKEDTED_PARAMS = re.compile(r"([^\s]+)\s+\[ <(.*)> = <(.*)> \]")
FILE_WITH_PATH = re.compile(r"(.*) <file=(.*)>")

class TauEvent():
    def _load_attrs(self, name):
        m = EXTENDED_ATTR_RE.match(name)

        # It is then a value (not JSON after |)
        if m is None:
            mm = VALUE_ATTR_RE.match(name)

            if mm:
                self.name = mm[1]
                self.attrs["value"] = mm[2]
            else:
                self.name = name
            return

        # Try to get JSON data
        try:
            self.name = m[1]
            self.attrs = json.loads(m[2])
        except Exception as e:
            # Failed
            print("Bad JSON in event : {}".format(name))
            print(e)
            self.name = name

    def _parse_name(self, name):

        locus = LOCUS_RE.match(name)

        if locus:
            self.name = locus[1]
            self.attrs["src_file"] = locus[2]
            self.attrs["line"] = locus[3]
            return True

        # Case of either extended ATTR or values
        if " | " in name:
            # Extended attr case
            self._load_attrs(name)
            return True

        # Extract stack info if present
        stack = STACK_RE.match(name)

        if stack:
            self.name = stack[1]
            self.attrs["stack"] = stack[2]
            name = stack[1]

        # Extract path info if present
        fwp = FILE_WITH_PATH.match(name)

        if fwp:
            self.name = fwp[1]
            self.attrs["file"] = fwp[2]
            return True
        elif stack:
            # Only parsed a stack
            return True

        params = BRACKEDTED_PARAMS.match(name)

        if params:
            self.name = params[1]
            self.attrs[params[2].replace(" ", "_")] = params[3]
            return True

        # Failed keep name as is
        return False

    def __init__(self, iid, group, tag, name, type, attrs=None):
        self.id = iid
        self.group = group
        self.tag = tag

        if attrs:
            self.attrs = attrs
        else:
            self.attrs = {}

        # Try to extract extended attrs
        if not self._parse_name(name):
            # No attribute case keep name as is
            self.name = name

        self.type = type

    def __str__(self, light=False, hide=None, convert=True):
        if hide:
            attrs = {}
            for k,v in self.attrs.items():
                if k not in hide:
                    attrs[k] = v
        else:
            attrs = self.attrs

        cpy = {}

        for k,v in attrs.items():
            if k == "ti":
                cpy[k] = ptime(attrs[k])
            else:
                cpy[k] = attrs[k]

        attr_desc = "\n\t".join([ "- {} : {}".format(k,v) for k,v in cpy.items()])
        common = "{}\t{}\t{}".format(self.id, self.type.name, self.name)
        if attr_desc and not light:
            return common + "\n\t" + attr_desc
        else:
            return common

    @property
    def nid(self):
        if "nid" in self.attrs:
            return self.attrs["nid"]
        else:
            return None

    @property
    def tid(self):
        if "tid" in self.attrs:
            return self.attrs["tid"]
        else:
            return None

    @property
    def ti(self):
        if "ti" in self.attrs:
            return self.attrs["ti"]
        else:
            return 0

    @property
    def value(self):
        if "value" in self.attrs:
            return self.attrs["value"]
        else:
            return 0

    def decode_msg_event(self):
        """   
            This is how parameter is generated for MSG events
            parameter = (xlength >> 16 << 54 >> 22) |
            ((xtype >> 8 & 0xFF) << 48) |
            ((xother >> 8 & 0xFF) << 56) |
            (xlength & 0xFFFF) |
            ((xtype & 0xFF)  << 16) |
            ((xother & 0xFF) << 24) |
            (xcomm << 58 >> 16);
        """
        if "value" not in self.attrs:
            print("Could not get attribute for TAU_MESSAGE event")
            return


        par = self.attrs["value"]

        mlength = (par & 0xFFFF) | (par & 0x3FF00000000) << 16
        mtype = (par >> 16 & 0xFF) | (par >> 48 & 0xFF << 8)
        mother = (par >> 24 & 0xFF) | (par >> 56 & 0xFF<< 8)
        mcomm = (par >> 42 & 0x3F)

        del self.attrs["value"]
        self.attrs.update({ "msg_length" : mlength,
                            "msg_tag": mtype,
                            "msg_remote": mother,
                            "msg_comm" : mcomm})

    def map(self, rawev):
        ret =  TauEvent(self.id, self.group, self.tag, self.name, self.type, self.attrs)

        ret.attrs.update(rawev.attrs())

        if ret.group == "TAU_MESSAGE":
            ret.decode_msg_event()

        return ret

    def serialize(self, verbose=False):
        ret  = {}
        ret["attrs"] = self.attrs
        ret["id"] = self.id
        if verbose:
            ret["name"] = self.name
            ret["group"] = self.group
            ret["tag"] = self.tag
        return ret


class TauTriggerValue(TauEvent):

    def __init__(self, iid, group, tag, name):
        super(TauTriggerValue, self).__init__(iid, group, tag, name, TauEventType.TRIGGERVALUE)


class TauEntryExit(TauEvent):

    def __init__(self, iid, group, tag, name):
        super(TauEntryExit, self).__init__(iid, group, tag, name, TauEventType.ENTRYEXIT)


class TauTrace():

    def _check_trace_path(self):
        if self.path_to_edf is None:
            raise Exception("Null trace file")
        if not os.path.isfile(self.path_to_edf):
            raise Exception("{} could not be found".format(self.path_to_edf))
        if self.path_to_trc is None:
            if not self.path_to_edf.endswith(".edf"):
                raise Exception("Please specify trace file manually")
            self.path_to_trc = self.path_to_edf.replace(".edf", ".trc")
        if not os.path.isfile(self.path_to_trc):
            raise Exception("{} could not be found".format(self.path_to_trc))
        #print("Processing {} over {}".format(self.path_to_edf, self.path_to_trc))

    def _predefined_events(self):
        # Tracer events
        self._idmap[60000] = TauEvent(60000, "TRACER", 0, "EV_INIT", TauEventType.TRIGGERVALUE)
        self._idmap[60001] = TauEvent(60001, "TRACER", 0, "TAU_EV_FLUSH", TauEventType.TRIGGERVALUE)
        # 60002 commented out
        self._idmap[60003] = TauEvent(60003, "TRACER", 0, "FLUSH_CLOSE", TauEventType.TRIGGERVALUE)
        self._idmap[60004] = TauEvent(60004, "TRACER", 0, "TAU_EV_INITM", TauEventType.TRIGGERVALUE)
        self._idmap[60005] = TauEvent(60005, "TRACER", 0, "WALL_CLOCK", TauEventType.TRIGGERVALUE)
        self._idmap[60006] = TauEvent(60006, "TRACER", 0, "TAU_EV_CONT_EVENT", TauEventType.TRIGGERVALUE)
        #self._idmap[60007] = TauEvent(60007, "TAU_MESSAGE", -7, "MESSAGE_SEND", TauEventType.TRIGGERVALUE)
        #self._idmap[60008] = TauEvent(60008, "TAU_MESSAGE", -8, "MESSAGE_RECV", TauEventType.TRIGGERVALUE)
        self._idmap[60009] = TauEvent(60009, "TRACER", 0, "TAU_MESSAGE_UNKNOWN", TauEventType.TRIGGERVALUE)

        self._group["TRACER"] = set()
        self._group["TRACER"].update([60000, 60001, 60003, 60004, 60005, 60006, 60009])

        self._group["TAU_MESSAGE"] = set()
        #self._group["TAU_MESSAGE"].update([60007, 60008])

        # GPUS
        self._idmap[70000] = TauEvent(70000, "TAUEVENT", 0, "ONESIDED_MESSAGE_SEND", TauEventType.TRIGGERVALUE)
        self._idmap[70001] = TauEvent(70001, "TAUEVENT", 0, "TAU_ONESIDED_MESSAGE_RECV", TauEventType.TRIGGERVALUE)
        self._idmap[70002] = TauEvent(70002, "TAUEVENT", 0, "TAU_ONESIDED_MESSAGE_ID_1", TauEventType.TRIGGERVALUE)
        self._idmap[70003] = TauEvent(70003, "TAUEVENT", 0, "TAU_ONESIDED_MESSAGE_ID_2", TauEventType.TRIGGERVALUE)
        self._idmap[70004] = TauEvent(70004, "TAUEVENT", 0, "TAU_ONESIDED_MESSAGE_UNKNOWN", TauEventType.TRIGGERVALUE)
        self._idmap[70004] = TauEvent(70005, "TAUEVENT", 0, "TAU_ONESIDED_MESSAGE_RECIPROCAL_SEND", TauEventType.TRIGGERVALUE)
        self._idmap[70004] = TauEvent(70006, "TAUEVENT", 0, "TAU_ONESIDED_MESSAGE_RECIPROCAL_RECV", TauEventType.TRIGGERVALUE)

        self._group["TAUEVENT"] = set()
        self._group["TAUEVENT"].update(range(70000, 70006))


    def _parse_edf_entry(self, entry):
        m = self.edf_re.match(entry)
        if m is None:
            return
        try:
            # Get DATA
            function_id = m[1]
            group = m[2]
            tag = m[3]
            name_type = m[4]
            parameters = m[5]

            # Instanciate Object
            type = TauEventType.parse(parameters)
            int_id = int(function_id)

            new_event = None

            if type == TauEventType.TRIGGERVALUE:
                new_event = TauTriggerValue(int_id, group, tag, name_type)

                if new_event.id in self._triggerValues:
                    raise Exception("{} is already registered for TauTriggerValue {}".format(int_id, name_type))

                self._triggerValues[int_id] = new_event

            elif type == TauEventType.ENTRYEXIT:
                new_event = TauEntryExit(int_id, group, tag, name_type)

                if new_event.id in self._entryExit:
                    raise Exception("{} is already registered for TauEntryExit {}".format(int_id, name_type))

                self._entryExit[int_id] = new_event
            else:
                new_event = TauEvent(int_id, group, tag, name_type, type)

                if new_event.id in self._others:
                    raise Exception("{} is already registered for UNDEFINED Kind {}".format(int_id, name_type))

                self._others[int_id] = new_event

            # Insert in the global ID table
            if int_id not in self._idmap:
                self._idmap[int_id] = new_event

            # Now reg the Group
            if group not in self._group:
                self._group[group] = set()

            self._group[group].add(int_id)

        except Exception as e:
            print(e)
            print("Failed to parse {}".format(entry))
            return

    def _read_edf_def(self):
        self._predefined_events()
        self.edf_re = re.compile(r"([0-9]+) ([A-Z_]+) ([-0-9]+) \"(.*)\" ([a-zA-Z_]+)")
        with open(self.path_to_edf, "r", errors='replace') as f:
            lines = f.readlines()
            for l in lines:
                self._parse_edf_entry(l)

    def get_by_id(self, id):
        if isinstance(id, str):
            id = int(id)
        return self._idmap.get(id)

    def __init__(self, path_to_edf, path_to_trc=None):
        self.path_to_edf = path_to_edf
        self.path_to_trc = path_to_trc
        self._check_trace_path()

        # All triggers
        self._triggerValues = {}
        # All entry & exit
        self._entryExit= {}
        # All other kind of events
        self._others = {}
        # All events in the same dict
        self._idmap = {}
        # A dict of sets listing
        # individual events IDs per group
        self._group = {}

        # Dict of nodes and then threads in these nodes
        self._topo = None

        self._read_edf_def()

        self._ev_decoder = TauEventDecoder(self.path_to_trc)


    def _topo_scan(self, rawev):
        if rawev.nid not in self._topo:
            self._topo[rawev.nid] = set()

        self._topo[rawev.nid].add(rawev.tid)



    def topology(self):
        if self._topo:
            return self._topo

        print("Scanning trace to list nodes and threads ... ", end="")
        sys.stdout.flush()
        self._topo = {}
        self._ev_decoder.read_trace(self._topo_scan)
        print("DONE")

        return self._topo



    def read(self, evid=None, nid=None, tid=None, callback=None, header=True):
        if not callback:
            def pcb(ev):
                print(ev)
            callback = pcb

        def local_cb(rawev):
            ev = self.get_by_id(rawev.ev)
            if ev:
                callback(ev.map(rawev))
            else:
                print("!!! NO DEC {} {} {}".format(rawev.ctx(), rawev.ev, rawev.par))

        if header:
            print("Processing trace... ", end="")
            sys.stdout.flush()
            
        self._ev_decoder.read_trace(local_cb, evid=evid, nid=nid, tid=tid)
        
        if header:
            print("DONE")

    @property
    def all_events(self):
        return self._idmap

    def _list_hiding(self, list, hide):
        for k,v in list.items():
            print("")
            print("="* (len(k) + 2))
            print("{} :".format(k))
            print("="* (len(k) + 2))
            print("")
            for evid in v:
                ev = self.get_by_id(evid)
                print(ev.__str__(hide=hide))

    def _list_events_for(self, keys):
        files = {}
        for v in self.all_events.values():
            for key in keys:
                if key in v.attrs:
                    f = v.attrs[key]

                    if f not in files:
                        files[f] = []

                    files[f].append(v.id)

        self._list_hiding(files, keys)

    @property
    def groups(self):
        return self._group.keys()

    def group_events(self, group):
        return self._group.get(group)

    def list_events_for_files(self):
        self._list_events_for(["file", "filename"])

    def list_events_for_stacks(self):
        self._list_events_for(["stack"])

    def params(self):
        return [ x for x in self.all_events.values() if "value" in x.attrs]

    def list_params(self):
        for v in self.params():
            print("{} = {}".format(v.name, v.attrs["value"]))

    def list_all_events(self):
        for v, e in self._group.items():
            print("==={}===".format(v))
            for v in e:
                evt = self.get_by_id(v)
                print(evt)


    def profile(self, evid=None, nid=None, tid=None):
        if evid:
            for e in evid:
                ev = self.get_by_id(e)
                if not ev:
                    raise Exception(f"No such evid: {e}")
                if ev.type != TauEventType.ENTRYEXIT:
                    raise Exception("Only ENTRYEXIT events are relevant for profiles")

        prof_data = {}

        def prof_cb(event):
            if event.type != TauEventType.ENTRYEXIT:
                return
            
            # Get slot for this call
            node = dict.setdefault(prof_data, event.nid, {})
            thread = dict.setdefault(node, event.tid, {})
            func = dict.setdefault(thread, event.name, {"event" : event.id, 
                                                        "stack" : 0,
                                                        "hits": 0,
                                                        "time": 0,
                                                        "entry_ts" : 0})

            # Update entry/exit counts
            func["stack"] = func["stack"] + event.value

            # Just make sure you are coherent
            if func["stack"] < 0:
                #print(f"Event Underflow on {event.name}")
                func["stack"] = 0

            # Count entries
            if(event.value == 1):
                func["hits"] = func["hits"] + 1

            # First entry for the thread
            if func["stack"] == 1:
                func["entry_ts"] = event.ti

            # Last exit for the thread
            if func["stack"] == 0:
                func["time"] = func["time"] + (event.ti - func["entry_ts"])



        self.read(evid=evid,
                  nid=nid,
                  tid=tid,
                  callback=prof_cb)

        for n in prof_data:
            ndata = prof_data[n]
            print(f"PROCESS {n}")
            for t in ndata:
                print(f"\tTHREAD {t}")
                tdata = ndata[t]
                print("\t\t#FUNC [event_id]\tHITS\tTIME")
                for f in tdata:
                    func = tdata[f]
                    print(f"\t\t{f}[{func['event']}]\t{func['hits']}\t{ptime(func['time'])}")


    def file_stats(self, nid=None, tid=None):
        # First generate a list of all events IDs attached to files
        file_events = set()
        
        for v in self.all_events.values():
            if ("file" in v.attrs) or ("filename" in v.attrs):
                file_events.add(v.id)


        evlist=[ x for x in file_events]

        files = {}

        def file_cb(event):
            file = ""
            if "file" in event.attrs:
                file = event.attrs["file"]
            elif "filename" in event.attrs:
                file = event.attrs["filename"]
            else:
                return

            fdict = None

            # Map per file
            if file not in files:
                files[file] = {}
            
            fdict = files[file]


            # Accumulate events per thread
            thread = f"n={event.nid} t={event.tid}"
            fdict = dict.setdefault(fdict, thread, {"stacks" : {}})

            # Map per stack if relevant
            for k in ["name", "stack"]:
                if k in event.attrs:
                    if k not in fdict["stacks"]:
                        fdict["stacks"][event.attrs[k]] = {"events" : set(), "event" : event.id}
                    fdict["stacks"][event.attrs[k]]["events"].add(event.name)

            value = event.attrs["value"]

            if event.name == "Bytes Read":
                rd = dict.setdefault(fdict, "read", {"total": 0, "max_bw" : 0, "event" : event.id})
                if "event" not in rd:
                    rd["event"] = event.id
                rd["total"] = rd["total"] + value
            elif event.name == "Read Bandwidth (MB/s)":
                rd = dict.setdefault(fdict, "read", {"total": 0, "max_bw" : 0})
                if (rd["max_bw"] == 0) or (rd["max_bw"] < value):
                    rd["max_bw"] = value
            elif event.name == "Bytes Written":
                rd = dict.setdefault(fdict, "write", {"total": 0, "max_bw" : 0, "event" : event.id})
                if "event" not in rd:
                    rd["event"] = event.id
                rd["total"] = rd["total"] + value
            elif event.name == "Write Bandwidth (MB/s)":
                rd = dict.setdefault(fdict, "write", {"total": 0, "max_bw" : 0})
                if (rd["max_bw"] == 0) or (rd["max_bw"] < value):
                    rd["max_bw"] = value


        self.read(evid=evlist, nid=nid, tid=tid, callback=file_cb)

        for f in files:
            print(f"- {f}")
            tdat = files[f]
            for t in tdat:
                print(f"\t* {t}")
                dat = tdat[t]
                if ("read" in dat) or ("write" in dat):
                    print("\t\t#op[event_id]\ttotal\tmax_bw")
                if "read" in dat:
                    bw = float(dat['read']['max_bw']) * (1024*1024)
                    eid = -1
                    if "event" in dat['read']:
                        eid = dat['read']['event']
                    print(f"\t\tREAD[{eid}]\t{psize(dat['read']['total'])}\t{psize(bw)}/sec")
                if "write" in dat:
                    eid = -1
                    if "event" in dat['write']:
                        eid = dat['write']['event']
                    bw = float(dat['write']['max_bw']) * (1024*1024)
                    print(f"\t\tWRITE[{eid}]\t{psize(dat['write']['total'])}\t{psize(bw)}/sec")

                if "stacks" in dat:
                    print("\t\tcallsites[event_id]:")
                    for s in dat["stacks"]:
                        print(f"\t\t\t- {s}[{dat['stacks'][s]['event']}] : {' '.join(dat['stacks'][s]['events'])}")



    def chrome(self, evid=None, nid=None, tid=None, path="app.json"):
        trace = {"displayTimeUnit": "us",
                 "otherData": {},
                 "traceEvents" : []
        }

        if not path:
            raise Exception("A path should be provided to write chrome trace")

        params = [x.name for x in self.params()]

        # Get all parameters events data and put in otherData
        for v in self.params():
            trace["otherData"][v.name] = v.attrs["value"]

        def chrome_event_handler(event):
            if event.type == TauEventType.ENTRYEXIT:
                if "value" not in event.attrs:
                    return
                value = event.attrs["value"]
                if int(value) == -1:
                    # Exit
                    etype = "E"
                elif int(value) == 1:
                    # Enter
                    etype = "B"
                else:
                    # What ?
                    return
                
                evt = {"name": event.name,
                       "cat": event.group,
                       "ph": etype,
                       "pid": event.attrs["nid"],
                       "tid": event.attrs["tid"],
                       "args" : event.attrs,
                       "ts": int(event.attrs["ti"])}

                trace["traceEvents"].append(evt)

            elif event.type == TauEventType.TRIGGERVALUE:
                if event.group == "TAU_MESSAGE":
                    # Handle Message Arrows
                    if event.name == "MESSAGE_SEND":
                        etype = ""
                    elif event.name == "MESSAGE_RECV":
                        etype = ""
                    else:
                        return
                else:
                    # Do not show params as counters
                    if event.name in params:
                        return

                    # Counter
                    etype = "C"

                    # Show Opens as Instant Events
                    if "posix open" in event.name:
                        etype = "i"

                    evt = {"name": event.name,
                       "cat": event.group,
                       "ph": etype,
                       "pid": event.attrs["nid"],
                       "tid": event.attrs["tid"],
                       "args" : {"value" :event.attrs["value"]},
                       "ts": event.attrs["ti"]}
                    trace["traceEvents"].append(evt)

            

        self.read(evid=evid,
                  nid=nid,
                  tid=tid,
                  callback=chrome_event_handler,
                  header=False)

        with open(path, "w") as f:
            f.write(json.dumps(trace))


""" 
Argument Parsing
"""


parser = argparse.ArgumentParser(description='TAU trace analyzer.')
# Trace file
parser.add_argument('-i', '--trace', type=str, default="./tau.edf", help='Provide custom path to .edf file')


# List events / counters
parser.add_argument('-l', '--list', choices=['all', 'params', 'topo'], default=None, help='List trace attributes')
# List by context
parser.add_argument('-ls', '--list-stacks', help="List events attached to a stackframe", action='store_true')
parser.add_argument('-lf', '--list-files', help="List events attached to a file", action='store_true')

# Print
parser.add_argument('-p', '--print', help="Print trace events",  action='store_true')
parser.add_argument('-j', '--json', help="Print trace events using JSON format",  action='store_true')
parser.add_argument('-jv', '--json-verbose', help="Print trace events using verbose JSON format",  action='store_true')
parser.add_argument('-c', '--chrome', type=str, default=None, help='Generate a trace in Chrome trace format')


# Profile
parser.add_argument('-cs', '--call-stats', help="Print function calls statistics for trace",  action='store_true')
parser.add_argument('-fs', '--file-stats', help="Print per file statistics for trace",  action='store_true')


# Filtering
parser.add_argument('-f', '--filter', type=str, help="Filter a comma separated list of events ids", default=None)
parser.add_argument('-n', '--process', type=int, help="Filter events from a given process", default=None)
parser.add_argument('-t', '--thread', type=int, help="Filter events from a given thread", default=None)


args = parser.parse_args()

trace = TauTrace(args.trace)

# Handle Listing

if args.list:
    if args.list == "all":
        trace.list_all_events()
    elif args.list == "params":
        trace.list_params()
    elif args.list == "topo":
        print(trace.topology())

    exit(0)

# Handle Grouped Listing

if args.list_stacks:
    trace.list_events_for_stacks()
    exit(0)

if args.list_files:
    trace.list_events_for_files()
    exit(0)


if args.filter:
    filter = args.filter.split(",")
else:
    filter = None

def use_json():
    return (args.json or args.json_verbose)

if not args.print:
    if use_json():
        raise Exception("JSON format modifier is only meaningful with the -p option")
else:
    pcallback = None
    ctx = {"is_first" : 1}

    data = []
    def js_callback(evt):
        if not ctx["is_first"] :
            print(",")
        ctx["is_first"] = 0
        edata = evt.serialize(verbose=args.json_verbose)
        print(json.dumps(edata), end="")

    if use_json():
        print("[")
        pcallback = js_callback


    trace.read(evid=filter,
            nid=args.process,
            tid=args.thread,
            callback=pcallback,
            header=False)

    if use_json():
        print("]")

    exit(0)

if args.call_stats:
    trace.profile(evid=filter, nid=args.process, tid=args.thread)
    exit(0)

if args.file_stats:
    if args.filter:
        raise Exception("Filter (-f) argument is not relevant for file statistics")
    trace.file_stats(nid=args.process, tid=args.thread)
    exit(0)

if args.chrome:
    trace.chrome(evid=filter, nid=args.process, tid=args.thread, path=args.chrome)
    exit(0)
