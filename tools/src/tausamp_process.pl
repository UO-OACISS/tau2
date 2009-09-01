#!/usr/bin/env perl

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


# Read the trace
my ($exe);
open (TRACE, "<ebstrace.0.0.0");
while ($line = <TRACE>) {
    if ($line =~ /\#.*/) {
	print "$line";
	if ($line =~ /\# exe:.*/) {
	    ($junk, $exe) = split("exe:",$line);
	    $exe = trim($exe);
	}
	next;
    }
    ($timestamp,$pc,$metrics,$callpath) = split('\|',$line);
    $timestamp = trim($timestamp);
    $pc = trim($pc);
    $metrics = trim($metrics);
    $callpath = trim($callpath);
    
    # Process the callpath
    $callpath = reverse($callpath);
    @events = split(" ",$callpath);
    $newCallpath = "";
    my (@processedEvents);
    foreach my $e (@events) {
	$newCallpath = "$newCallpath => $eventmap{$e}";
	push (@processedEvents, $eventmap{$e});
    }
    $newCallpath = join(" => ", @processedEvents);

    # Process the PC
    $out = `echo $pc | addr2line -e $exe`;
    chomp($out);
    $newpc = $out;

    # Output the processed data
    print "$timestamp | $newpc | $metrics | $newCallpath\n";
}
