#!/usr/bin/env perl

########################################################################
# This script reads an EBS trace and converts the PC's and TAU callpaths
########################################################################

# Trim leading and trailing whitespace from a string
sub trim($) {
    my $string = shift;
    $string =~ s/^\s+//;
    $string =~ s/\s+$//;
    return $string;
}


# Read event definitions
my %eventmap;
open (DEF, "<ebstracedef.0.0.0");
while ($line = <DEF>) {
    if ($line =~ /\#.*/) {
	next;
    }
    ($id, $name) = split('\|',$line);
    $id = trim($id);
    $name = trim($name);
    $eventmap{$id} = $name;
}


my ($lastCallpath);
my ($useDeltaStart);
my ($useDeltaStop);

# Read the trace
my ($exe);
open (TRACE, "tac ebstrace.0.0.0 |");
open (OUTPUT, "| tac > ebstrace.processed.0.0.0");
while ($line = <TRACE>) {
    if ($line =~ /\#.*/) {
	print (OUTPUT "$line");
	if ($line =~ /\# exe:.*/) {
	    ($junk, $exe) = split("exe:",$line);
	    $exe = trim($exe);
	}
	next;
    }
    # parse a line
    ($timestamp,$deltaStart,$deltaStop,$pc,$metrics,$callpath) = split('\|',$line);
    $timestamp = trim($timestamp);
    $pc = trim($pc);
    $metrics = trim($metrics);
    $callpath = trim($callpath);
    
    # Process the callpath
    if ($callpath eq "-1") {
	$newCallpath = $lastCallpath;
    } else {
	$callpath = reverse($callpath);
	@events = split(" ",$callpath);
	$newCallpath = "";
	my (@processedEvents);
	foreach my $e (@events) {
	    $newCallpath = "$newCallpath => $eventmap{$e}";
	    push (@processedEvents, $eventmap{$e});
	}
	$newCallpath = join(" => ", @processedEvents);
	$lastCallpath = $newCallpath;
	$useDeltaStart = $deltaStart;
	$useDeltaStop = $deltaStop;
    }


    $deltaStart = $timestamp - $useDeltaStart;
    $deltaStop = $useDeltaStop - $timestamp;

    # Process the PC
    $out = `echo $pc | addr2line -e $exe`;
    chomp($out);
    $newpc = $out;

    # Output the processed data
    print OUTPUT "$timestamp | $deltaStart | $deltaStop | $newpc | $metrics | $newCallpath\n";
}





