#!/usr/bin/env perl
use strict;
use File::Basename;
use Cwd qw(realpath);

my $break = 250;

if (@ARGV >= 1) {
    if ($ARGV[0] ne "-n" || @ARGV != 2) {
	die "usage: $0 [-n <break amount> ]\n";
    }

    $break = $ARGV[1];

    if ($break < 2) {
	die "nice try, can't use break < 2\n";
    }
    print "Using breakpoint of $break traces per merge\n";
}

my @traces; # trace files for the next merge
my @edfs;   # edf files for the next merge
my $numtraces = 0;  # how many we have in the two arrays so far

my $pattern = "tautrace.*.*.*";

my $level = 0; # current tree level
my $done = 0;  # done or not
my @out;       # output from tau_merge

my $tau_merge = "tau_merge";

my $check_ex = realpath(dirname($0))."/$tau_merge";

if(-x $check_ex) {
     $tau_merge = $check_ex;
}

system("/bin/rm -f intermediate.*");

while ($done == 0) {

    my $count = 0;
    my $neednext = 0;

    while ( defined(my $filename = glob($pattern)) ) {
#	printf "filename = $filename\n";
	@traces[$numtraces] = $filename;
	
	if ($filename =~ /tautrace/) {
	    my $node = $filename;
	    $node =~ s/tautrace\.([0-9]*)\.([0-9]*)\.([0-9]*)\.trc/$1/;
	    @edfs[$numtraces] = "events.$node.edf";
	} else {
	    my ($a, $b);
	    $a = $filename;
	    $b = $filename;
	    $a =~ s/intermediate\.([0-9]*)\.([0-9]*)\.trc/$1/;
	    $b =~ s/intermediate\.([0-9]*)\.([0-9]*)\.trc/$2/;
	    @edfs[$numtraces] = "intermediate.$a.$b.edf";
	}
	$numtraces++;
	
	
	if ($numtraces >= $break) {
	    $neednext = 1;
	    my $basename = "intermediate.$level.$count";

	    print "$tau_merge -m $basename.edf -e @edfs @traces $basename.trc\n";
	    @out = `$tau_merge -m $basename.edf -e @edfs @traces $basename.trc`;
	    
	    $numtraces = 0;
	    @traces = ();
	    @edfs = ();
	    $count++;
	}
    }

    if ($neednext == 0) { 
	print "$tau_merge -m tau.edf -e @edfs @traces tau.trc\n";
	@out = `$tau_merge -m tau.edf -e @edfs @traces tau.trc`;
	$done = 1;
    }

    # set the next patten and move to the next level
    $pattern = "intermediate.$level.*.trc";
    $level++;
}

# clean up
system("/bin/rm -f intermediate.*");

