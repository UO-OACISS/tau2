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

if (@ARGV != 2) {
    die "usage: $0 <pdb file> <source file>\n";
}

my $header_list = realpath(dirname($0))."/tau_header_list";

my (@headers);
@headers = `$header_list $ARGV[0]`;

my ($file);
$file = $ARGV[1];

my ($newlocation);
$newlocation = $ARGV[2];
open (SOURCE, "<$file") || die "Cannot open file: $!";

my ($line);
while ($line = <SOURCE>) {
    if ($line =~ /^\s*\#\s*include/) { # match <spaces>#<spaces>include
        chomp($line);

	my ($header);
	foreach $header (@headers) {
	    chomp($header);
	    $header =~ s/.*\///g; # remove path
	    if ($line =~ /$header/) {
		# replace with tau_hr_<header>
		$line = "#include <tau_hr_$header>";
	    }
	}
        print "$line\n";
    } else {
        print $line;
    }
}
