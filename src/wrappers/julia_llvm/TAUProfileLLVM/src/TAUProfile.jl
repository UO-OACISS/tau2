#=
/****************************************************************************
**			TAU Portable Profiling Package                                 **
**			http://www.cs.uoregon.edu/research/tau                         **
*****************************************************************************
**    Copyright 2009-2026                    						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TAUProfile.jl 		          	                   **
**	Description 	: TAU Tracing for native OTF2 generation			   **
**  Author      : Nicholas Chaimov                                         **
**	Contact		: tau-bugs@cs.uoregon.edu               	               **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau           **
**                                                                         **
**      Description     :  LLVM Plugin-based Julia rewriter                **
**                                                                         **
****************************************************************************/
#
# This file contains code derived from GPUCompiler.jl, which is licensed under the terms of the 
# MIT "Expat" License:
#
# Copyright (c) 2019-present: Julia Computing and other contributors
#
#Copyright (c) 2014-2018: Tim Besard
#
#All Rights Reserved.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

#=
TAUProfile.jl — Function entry/exit tracing via LLVM pass plugin
=#

module TAUProfile

export tau_rewrite_and_call, trace_code, @tau_rewrite,
       tau_rewrite_exclude_function, tau_rewrite_exclude_module,
       tau_rewrite_exclude_prefix, tau_rewrite_reset_exclusions,
       tau_rewrite_include_module_only,
       tau_rewrite_set_recursion_limit, tau_rewrite_include_types,
       tau_rewrite_force_noinline, tau_rewrite_time_phase1,
       tau_rewrite_time_phase2,
       enable_tracing!, disable_tracing!, tracing_enabled,
       tau_start, tau_stop, @tau, @tau_func, tau_rewrite,
       tau_rewrite_deferred_contexts, tau_rewrite_set_min_complexity,
       TauTimer, tau_get_calls, tau_get_child_calls,
       tau_get_inclusive, tau_get_exclusive,
       tau_counter_names, tau_function_names,
       tau_set_name, tau_set_type, tau_set_group,
       TauEvent, tau_event, tau_context_event,
       tau_metadata, tau_context_metadata,
       tau_dynamic_start, tau_dynamic_stop,
       tau_enable_instrumentation, tau_disable_instrumentation,
       tau_enable_group, tau_disable_group,
       tau_enable_all_groups, tau_disable_all_groups,
       tau_dump, tau_snapshot, tau_exit

using LLVM
using GPUCompiler
import Core.Compiler as CC

include("runtime.jl")
include("tau_api.jl")
include("config.jl")
include("rewrite_timers.jl")
include("metadata.jl")
include("emit.jl")
include("llvm_pass.jl")
include("phase1.jl")
include("phase2.jl")
include("api.jl")

function __init__()
    _init_runtime!()
    _init_phase2!()
end

end # module TAUProfile
