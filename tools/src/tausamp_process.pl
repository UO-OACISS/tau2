#!/usr/bin/env perl


sub trim($) {
    my $string = shift;
    $string =~ s/^\s+//;
    $string =~ s/\s+$//;
    return $string;
}



my %eventmap;

# Read event definitions
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

open (TRACE, "<ebstrace.0.0.0");

my ($exe);

while ($line = <TRACE>) {
    if ($line =~ /\#.*/) {
	#print "$line";
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

#    print "pc = $pc";

    # Process the PC
    $out = `echo $pc | addr2line -e $exe`;
    chomp($out);
    $newpc = $out;
    print "$timestamp | $newpc | $metrics | $newCallpath\n";
}
