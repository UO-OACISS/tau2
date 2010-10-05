#!/usr/bin/env perl

###########################################################
#
# This file is used to replace the #include lines 
# for header instrumentation
#
###########################################################

use strict;
use File::Basename;
use Cwd qw(realpath);

if (@ARGV != 3) {
    die "usage: $0 <pdb file> <original file> <source file>\n";
}

my $header_list = realpath(dirname($0))."/tau_header_list";

my ($pdbfile);
$pdbfile = $ARGV[0];

my ($origfile);
$origfile = $ARGV[1];

my ($file);
$file = $ARGV[2];

my (@headers);
@headers = `$header_list --show $origfile $pdbfile`;
my ($header);
foreach $header (@headers) {
    chomp($header);
}

my (@ids);
@ids = `$header_list --showids $origfile $pdbfile`;
my ($id);
foreach $id (@ids) {
    chomp($id);
}

my ($i);

my (%idhash);

for ($i=0; $i<=$#headers; $i++) {
    $idhash{$headers[$i]} = $ids[$i];
}


my ($newlocation);
$newlocation = $ARGV[3];
open (SOURCE, "<$file") || die "Cannot open file: $!";

my ($line);
while ($line = <SOURCE>) {
    if ($line =~ /^\s*\#\s*include/) { # match <spaces>#<spaces>include
        chomp($line);


        # first, attempt without removing path
	my ($matched);
	$matched = "false";
	foreach $header (@headers) {
	    $id = $idhash{$header};

	    my ($checkheader);
	    $checkheader = $header;
	    if ($line =~ /\W\Q$checkheader\E\W/) {
		$checkheader =~ s/.*\///; # remove path
		# replace with $id_tau_hr_<header>
		$line = "#include <${id}_tau_hr_$checkheader>";
		$matched = "true";
	    }
	}

        # next, loop through again, this time removing the path
	if ($matched eq "false") {
	    foreach $header (@headers) {
		$id = $idhash{$header};
		
		my ($checkheader);
		$checkheader = $header;
		$checkheader =~ s/.*\///; # remove path
		if ($line =~ /\W\Q$checkheader\E\W/) {
		    # replace with $id_tau_hr_<header>
		    $line = "#include <${id}_tau_hr_$checkheader>";
		}
	    }
	}

        print "$line\n";
    } else {
        print $line;
    }
}
