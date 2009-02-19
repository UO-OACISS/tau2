#!/usr/bin/env perl

print "Testing tau_instrumentor\n\n";
$which=`which tau_instrumentor`;
print "Using $which\n";


$numpass=0;
$numfail=0;

# Read each line of the list
open (LIST, "list");
while ($line = <LIST>) {
    chomp($line);
    my ($pdb,$source,$check,$select) = split(" ", $line);

# Invoke the tau_instrumentor
    if ($select eq "none") {
	@output = `tau_instrumentor $pdb $source -o $source.testinst`;
    } else {
	@output = `tau_instrumentor $pdb $source -o $source.testinst -f $select`;
    }

# Check the output against the known output
    $ret = system("diff $source.testinst $check &>/dev/null");
    if ($ret == 0) {
	$numpass++;
	print "$source : pass\n";
    } else {
	$numfail++;
	print "$source : fail\n";
    }
}

# Report pass/fail
print "\n";
print "Tests passed: $numpass\n";
print "Tests failed: $numfail\n";
