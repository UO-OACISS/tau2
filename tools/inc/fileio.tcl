############################################################################
# File IO operations:
#
#   FileIO_INIT <tauroot>
#
#   FileIO_ls <host> [<dir>]
#   FileIO_pwd <host>
#   FileIO_file_readable <host> <file>
#   FileIO_file_writable <host> <file>
#   FileIO_file_exists <host> <file>               - returns false for dir
#   FileIO_dir_exists <host> <dir>                 - returns false for file
#   FileIO_file_open <host> <file> [<access>]
#   FileIO_file_close <host> <file_id>
#   FileIO_exec <host> <script>
#
#   Remote_ls <host> [<dir>]
#   Remote_ls-F <host> [<dir>]
#   Remote_pwd <host>
#   Remote_file_readable <host> <file>
#   Remote_file_writable <host> <file>
#   Remote_file_exists <host> <file>               - returns false for dir
#   Remote_dir_exists <host> <dir>                 - returns false for file
#   Remote_file_open <host> <file> [<access>]
#   Remote_file_close <file_id>
#   Remote_exec <host> <script>
#
###########################################################################


###########################################################################
#
# FileIO_INIT - must be called prior to calls to any other FileIO or 
#               RemoteIO functions.
#   Usage:  FileIO_INIT <tau root directory>
#

set Local_Hostname   ""
set FileIO_Error     ""
set RemoteIO_Error   ""
set FileIO_Root      ""

proc FileIO_INIT {rootdir} {
    global \
	    Local_Hostname \
	    FileIO_Error \
	    RemoteIO_Error \
	    FileIO_Root

    set FileIO_Error     ""
    set RemoteIO_Error   ""

    # Set the local hostname
    if {$Local_Hostname == ""} {
	if {![catch "exec hostname" h__n]} {
	    set Local_Hostname $h__n
	    unset h__n
	} else {
	    set Local_Hostname "localhost"
	}
    }

    set FileIO_Root $rootdir
}



############################################################################
#
# High-level I/O Operations that work for local or remote hosts
#
# These all return FILEIO_ERROR when an error occurs and side effect the
# global variable FileIO_Error with the error message returned.
#

proc FileIO_ls {host {dir ""}} {
    global Local_Hostname FileIO_Error Remote_Error

    if {($host == "localhost") || ($host == $Local_Hostname)} {
	if {[catch "exec ls $dir" result] == 1} {
	    set FileIO_Error "Couldn't open directory"
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    } else {
	set result [Remote_ls $host $dir]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_pwd {host} {
    global Local_Hostname FileIO_Error Remote_Error

    if {($host == "localhost") || ($host == $Local_Hostname)} {
	if {[catch "pwd" result] == 1} {
	    set FileIO_Error $result
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    } else {
	set result [Remote_pwd $host]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_file_readable {host filen} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	set FileIO_Error ""
	return [file readable $filen]
    } else {
	set result [Remote_file_readable $host $filen]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_file_writable {host filen} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	set FileIO_Error ""
	return [file writable $filen]
    } else {
	set result [Remote_file_writable $host $filen]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_file_exists {host filen} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	set FileIO_Error ""
	return [expr [file exists $filen] && ![file isdirectory $filen]]
    } else {
	set result [Remote_file_exists $host $filen]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_dir_exists {host dirn} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	set FileIO_Error ""
	return [expr [file exists $dirn] && [file isdirectory $dirn]]
    } else {
	set result [Remote_dir_exists $host $dirn]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_file_open {host filen {access "r"}} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	if {[catch "open $filen $access" result] == 1} {
	    set FileIO_Error $result
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    } else {
	set result [Remote_file_open $host $filen $access]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}


proc FileIO_file_close {host file_desc} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	if {[catch "close $file_desc" result] == 1} {
	    set FileIO_Error $result
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return FILEIO_OKAY
	}
    } else {
	set result [Remote_file_close $file_desc]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return FILEIO_OKAY
	}
    }
}


proc FileIO_exec {host script} {
    global Local_Hostname FileIO_Error Remote_Error
    
    if {($host == "localhost") || ($host == $Local_Hostname)} {
	if {[catch "exec $script" result] == 1} {
	    set FileIO_Error $result
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    } else {
	set result [Remote_exec $host $script]
	if {$result == "REMOTE_ERROR"} {
	    set FileIO_Error $Remote_Error
	    return FILEIO_ERROR
	} else {
	    set FileIO_Error ""
	    return $result
	}
    }
}



###########################################################################
#
# Low-level remote host operations
#
# These all return REMOTE_ERROR when an error occurs and side effect the
# global variable Remote_Error with the error message returned.

proc Remote_ls {host {dir ""}} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -listdir $dir" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_ls-F {host {dir ""}} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -listdirF $dir" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_pwd {host} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -pwd" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_file_readable {host file} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -freadable $file" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_file_writeable {host file} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -fwritable $file" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_file_exists {host file} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -fexists $file" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_dir_exists {host dir} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -dexists $dir" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}


proc Remote_file_open {host file {access r}} {
    global REMSH Remote_Error FileIO_Root
    
    switch -exact $access {
	r {
	    set open_cmd [format "| %s %s -n cat %s" $REMSH $host $file]
	}

	w {
	    set open_cmd [format "| %s %s %s/utils/remote-proxy -write %s" \
		    $REMSH $host $FileIO_Root $file]
	}

	a {
	    set open_cmd [format "| %s %s %s/utils/remote-proxy -append %s" \
		    $REMSH $host $FileIO_Root $file]
	}

	default {
	    set Remote_Error "Invalid access flag - use r, w, or a"
	    return REMOTE_ERROR
	}
    }

    set Remote_Error ""
    return [open $open_cmd $access]
}


proc Remote_file_close {file_desc} {
    global REMSH Remote_Error FileIO_Root

    if {[set errcode [catch "close $file_desc" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return REMOTE_OKAY
    }
}

proc Remote_exec {host script} {
    global REMSH Remote_Error FileIO_Root

    set in [open \
	    "| $REMSH $host -n $FileIO_Root/utils/remote-proxy -execute $script" \
	    r]
    set result [read $in]

    if {[set errcode [catch "close $in" errmsg]] != 0} {
	set Remote_Error $errmsg
	return REMOTE_ERROR
    } else {
	set Remote_Error ""
	return [string trim $result]
    }
}

# End of low-level operations    
###########################################################################
