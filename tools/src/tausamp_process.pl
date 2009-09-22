#!/usr/bin/env perl

########################################################################
# This script reads an EBS trace and converts the PC's and TAU callpaths
########################################################################

use strict;

# Trim leading and trailing whitespace from a string
sub trim($) {
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

sub process_trace {
  my($def_file, $trace_file, $out_file) = @_;

  # Read event definitions
  my %eventmap;
  open (DEF, "<$def_file");
  my ($line);
  while ($line = <DEF>) {
    if ($line =~ /\#.*/) {
      next;
    }
    my ($id, $name) = split('\|',$line);
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
  my ($junk, $exe, $node);
  open (TRACE, "tac $trace_file |");
  open (OUTPUT, "| tac > $out_file");
  while ($line = <TRACE>) {
    if ($line =~ /\#.*/) {
      if ($line =~ /\# exe:.*/) {
	($junk, $exe) = split("exe:",$line);
	$exe = trim($exe);
      } elsif ($line =~ /\# node:.*/) {
	($junk, $node) = split("node:",$line);
	$node = trim($node);
      } else {
	print (OUTPUT "$line");
      }
      next;
    } elsif ($line =~ /\%.*/) {
      # process stop lines

      my ($type,$start,$stop,$callpath) = split('\|',$line);
      $start = trim($start);
      $stop = trim($stop);
      $callpath = trim($callpath);
      $startmap{$callpath} = $start;
      $stopmap{$callpath} = $stop;
    } else {
      # process sample lines

      my ($type,$timestamp,$deltaStart,$deltaStop,$pc,$metrics,$callpath) = split('\|',$line);
      $timestamp = trim($timestamp);
      $pc = trim($pc);
      $metrics = trim($metrics);
      $callpath = trim($callpath);

      # Process the callpath
      my @events = split(" ",$callpath);
      @events = reverse (@events);

      my $newCallpath = "";
      my (@processedEvents);
      foreach my $e (@events) {
	$newCallpath = "$newCallpath => $eventmap{$e}";
	push (@processedEvents, $eventmap{$e});
      }
      $newCallpath = join(" => ", @processedEvents);
      $lastCallpath = $newCallpath;

      my $check = $deltaStart;
      $deltaStart = $timestamp - $startmap{$callpath};
      $deltaStop = $stopmap{$callpath} - $timestamp;

#       if ($check != $startmap{$callpath}) {
# 	die "inconsistent file, $check != $startmap{$callpath}\n";
#       }

      # Process the PC
      my $out = `echo $pc | addr2line -e $exe`;
      chomp($out);
      my $newpc = $out;

      # Output the processed data
      print OUTPUT "$timestamp | $deltaStart | $deltaStop | $newpc | $metrics | $newCallpath\n";
    }
  }
  print OUTPUT "# node: $node\n";
}



sub main {

  my $pattern = "ebstrace.raw.*.*.*.*";
  while (defined(my $filename = glob($pattern))) {
    my ($trace_file, $def_file, $out_file);
    $trace_file = $filename;
    my ($junk1, $junk2);
    my ($junk1,$junk2,$pid,$nid,$cid,$tid) = split('\.',$filename);
    $def_file = "ebstrace.def.$pid.$nid.$cid.$tid";
    $out_file = "ebstrace.processed.$pid.$nid.$cid.$tid";
    print "processing $filename ...\n";
    process_trace($def_file, $trace_file, $out_file);
  }
}


main
