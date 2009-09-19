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

my %startmap;
my %stopmap;

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
    } elsif ($line =~ /\%.*/) {
	# process stop lines

	($type,$start,$stop,$callpath) = split('\|',$line);
	$start = trim($start);
	$stop = trim($stop);
	$callpath = trim($callpath);
	$callpath = reverse($callpath);
	$startmap{$callpath} = $start;
	$stopmap{$callpath} = $stop;
    } else  {
	# process sample lines

	($type,$timestamp,$deltaStart,$deltaStop,$pc,$metrics,$callpath) = split('\|',$line);
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
	$lastCallpath = $newCallpath;


	$check = $deltaStart;
	$deltaStart = $timestamp - $startmap{$callpath};
	$deltaStop = $stopmap{$callpath} - $timestamp;

	if ($check != $startmap{$callpath}) {
	    die "inconsistent file, $check != $startmap{$callpath}\n";
	}

	# Process the PC
	$out = `echo $pc | addr2line -e $exe`;
	chomp($out);
	$newpc = $out;

	# Output the processed data
	print OUTPUT "$timestamp | $deltaStart | $deltaStop | $newpc | $metrics | $newCallpath\n";
    }
}





