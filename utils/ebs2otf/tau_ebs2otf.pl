#!/usr/bin/env perl

########################################################################
# This script reads an EBS trace and converts the PC's and TAU callpaths
########################################################################

use warnings;
use strict;
use IO::Handle;




use ebs2otf;
my $manager = ebs2otf::OTF_FileManager_open ( 100 );
#print $manager;

my $streams = 0;
#my $streams = 8;
my $processes = 4;
my $counterid = 5;

my $writer = ebs2otf::OTF_Writer_open ("ebstrace.otf", $streams, $manager);

my $firstTimestamp = -1;

ebs2otf::OTF_Writer_setBufferSizes( $writer, 10*1024 );

ebs2otf::OTF_Writer_writeDefCounterGroup( $writer, 0, 63, "the counters");
#ebs2otf::OTF_Writer_writeDefTimerResolution( $writer, 0, 1e6);
ebs2otf::OTF_Writer_writeDefTimerResolution( $writer, 0, 1e6);


my $aggregate_to_routine = 0;
my %counters;
my %otf_function_map;
my $otf_num_functions = 0;

# Read only version 0.2
my $reader_version = 0.2;

my $metricsSet = 0;
my $forked = 0;
my $cpuid = 1;

##### create pipes to handle communication
pipe (FROM_PERL, TO_PROGRAM);
pipe (FROM_PROGRAM, TO_PERL);
TO_PROGRAM->autoflush(1);
TO_PERL->autoflush(1);



sub getEventId {
  my ($event) = @_;

  my $val = $otf_function_map{$event};
  if (defined $val) {
    return $val;
  }

  $otf_function_map{$event} = $otf_num_functions;
  $otf_num_functions++;
  return $otf_num_functions-1;
}


sub compare_arrays {
  my ($first, $second) = @_;
  no warnings;  # silence spurious -w undef complaints
  return 0 unless @$first == @$second;
  for (my $i = 0; $i < @$first; $i++) {
    return 0 if $first->[$i] ne $second->[$i];
  }
  return 1;
}



# Trim leading and trailing whitespace from a string
sub trim($) {
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

my %pcmap;

# Translate a PC value
sub translate_pc {
  my ($exe, $pc) = @_;

  if (defined $pcmap{$pc}) {
      return "$pcmap{$pc}";
  }

  if ($forked == 0) {
    $forked = 1;
    my $pid = fork;

    ##### child process becomes the program
    if ($pid == 0) {
      ##### attach standard input/output/error to the pipes
      close  STDIN;
      open  (STDIN,  '<&FROM_PERL') || die ("open: $!");

      close  STDOUT;
      open  (STDOUT, '>&TO_PERL')   || die ("open: $!");

      # close  STDERR;
      # open  (STDERR, '>&STDOUT')    || die;

      ##### close unused parts of pipes
      close FROM_PROGRAM;
      close TO_PROGRAM;

      ##### unbuffer the outputs
      select STDERR; $| = 1;
      select STDOUT; $| = 1;

      if (defined $ENV{"TAU_EBS_EXE"}) {
        $exe = $ENV{"TAU_EBS_EXE"};
      }

      ##### execute the program
      exec "addr2line -C -f -e $exe";

      ##### shouldn't get here!!!
      die;
    } else {
      close FROM_PERL;
      close TO_PERL;
    }
  }

  # write the pc to addr2line
  print TO_PROGRAM "$pc\n";

 # read the result
  my ($func, $fileline);
  $func = <FROM_PROGRAM>;
  if ($func =~ m/^BFD: Dwarf Error/) {
    $func = "unknown";
    $fileline = "unknown:??";
  } else {
    $fileline = <FROM_PROGRAM>;
  }
  chomp($func);
  chomp($fileline);

#  print "Got fileline: $fileline\n";

  # strip the path
  my ($file,$line) = split (":", $fileline);
  $file =~ s!^.*/([^/]*)$!$1!;
  $fileline = "$file:$line";

  my $string = "$func:$fileline";

  if ($aggregate_to_routine == 1) {
    $string = "$func\@\@\@$file";
  }

  $pcmap{$pc} = $string;
  return $string;
}

# process an EBS trace file
sub process_trace {
  my($def_file, $trace_file, $out_file) = @_;

  my $rc;

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
  my @startTokens;
  my @stopTokens;

  my $totalSamples = 0;
  my $negativeSamples = 0;


  my ($junk, $exe, $node, $thread, $version);

  my (@otf_callstack);

  # Read the footer
  my @footer = `tail -3 $trace_file`;
  ($junk, $exe) = split("exe:",$footer[0]);
  $exe = trim($exe);
  ($junk, $node) = split("node:",$footer[1]);
  $node = trim($node);
  ($junk, $thread) = split("thread:",$footer[2]);
  $thread = trim($thread);

  print "exe = $exe\n";
  print "node = $node\n";
  print "thread = $thread\n";
  $cpuid = $cpuid + 1;

  my $process_name = "Node $node, thread $thread";
  ebs2otf::OTF_Writer_writeDefProcess( $writer, 0, $cpuid, $process_name, 0);

  my $lasttimestamp;

  # Read the trace
  open (TRACE, "$trace_file");
  open (OUTPUT, ">$out_file");
  while ($line = <TRACE>) {
#    print "$line";
    if ($line =~ /\#.*/) {
      if ($line =~ /\# Format.*/) {
        ($junk, $version) = split("version:",$line);
	$version = trim($version);
	if ($version != $reader_version) {
	  die "This reader is only for version $reader_version files, you have $version, sorry\n";
	}
        print (OUTPUT "$line");
      } elsif ($line =~ /\# Metrics.*/) {
        my ($junk, $metricString) = split("Metrics:",$line);
	$metricString = trim($metricString);
	if ($metricsSet == 0) {
	  $metricsSet = 1;
	  my (@metrics) = split(" ", $metricString);

	  my $counterid = 0;
	  foreach my $metric (@metrics) {
	    ebs2otf::OTF_Writer_writeDefCounter( $writer, 0, $counterid, $metric, $ebs2otf::OTF_COUNTER_TYPE_ACC | $ebs2otf::OTF_COUNTER_SCOPE_START, 63, "#");
	    $counterid++;
	  }
	}

      } elsif ($line =~ /\# \$.*/) {
	## ignore the format line
      } elsif ($line =~ /\# \%.*/) {
	## output the true format line
	print (OUTPUT "# <timestamp> | <delta-begin> | <delta-end> | <delta-begin metric 1> <delta-end metric 1> ... <delta-begin metric N> <delta-end metric N> | <tau callpath> | <pc callstack>\n");
      } else {
        print (OUTPUT "$line");
      }
      next;
    } elsif ($line =~ /\%.*/) {
      # process stop lines
      my ($type,$metricsStart,$metricsStop,$callpath) = split('\|',$line);
      $callpath = trim($callpath);
      # there is a start value for each metric
      $startmap{$callpath} = $metricsStart;
      # there is a stop value for each metric
      $stopmap{$callpath} = $metricsStop;
    } else {
      # process sample lines

      my ($type,$timestamp,$deltaStart,$deltaStop,$metrics,$callpath,$callstack) = split('\|',$line);
      $timestamp = trim($timestamp);
      $metrics = trim($metrics);
      $callpath = trim($callpath);
      $callstack = trim($callstack);

#      print "timestamp was $timestamp\n";

      if ($firstTimestamp == -1) {
	$firstTimestamp = $timestamp;
      }

      $timestamp = $timestamp - $firstTimestamp + 90000000;
      $lasttimestamp = $timestamp;

      if ($timestamp <= 0) {
	die "timestamp is $timestamp, first = $firstTimestamp\n";
      }

#      print "timestamp is $timestamp\n";

      my @metricValues = split(" ",$metrics);

      for (my $metric = 0; $metric <= $#metricValues; $metric++) {
	$rc = ebs2otf::OTF_Writer_writeCounter( $writer, $timestamp, $cpuid, $metric, $metricValues[$metric] );
	if ($rc == 0) {
	  die "error in OTF_Writer_writeCounter\n";
	}
      }

      # Process the callpath
      my $callpathString = $eventmap{$callpath};
      my @events = split(" => ", $callpathString);

      my @resolvedPath;
      foreach my $e (@events) {
        push (@resolvedPath, getEventId($e));
      }


      my @callstackEntries = split(" ",$callstack);

      # need to drop the top if more than one
      if ($#callstackEntries > 0) {
	pop(@callstackEntries);
      }
      @callstackEntries = reverse (@callstackEntries);

      # Process the callstack
      my $newCallstack = "";
      foreach my $cs (@callstackEntries) {
	my $loc = translate_pc($exe, $cs);
	$newCallstack = "$newCallstack$loc\@";
	push (@resolvedPath, getEventId($loc));
      }
      chop ($newCallstack); # remove the last @

       # print "Old: *** @otf_callstack ***\n";
       # print "New: *** @resolvedPath ***\n";

      if (compare_arrays(\@otf_callstack, \@resolvedPath) == 0) {
	my $i = 0;

	for ($i = 0; $i <= $#resolvedPath; $i++) {
	  if ($i > $#otf_callstack) {
	    last;
	  }
	  if ($otf_callstack[$i] ne $resolvedPath[$i]) {
	    last;
	  }
	}

#	for my $j ($i..$#otf_callstack) {
#	  print "remainder = $j, $otf_callstack[$j]\n";
#	}

	my @tostop = reverse(@otf_callstack[$i..$#otf_callstack]);
	my @tostart = @resolvedPath[$i..$#resolvedPath];

	# print "tostop : @tostop\n";
	# print "tostart: @tostart\n";


	foreach my $e (@tostop) {
#	  print "\nstopping $e\n";
#	  my $event_id = getEventId($e);
	  ebs2otf::OTF_Writer_writeLeave( $writer, $timestamp, $e, $cpuid, 0);
	}

	foreach my $e (@tostart) {
#	  print "\nstarting $e\n";
#	  my $event_id = getEventId($e);
	  ebs2otf::OTF_Writer_writeEnter( $writer, $timestamp, $e, $cpuid, 0);
	}

      }


      @otf_callstack = @resolvedPath;

#      print "time = $timestamp, path = @resolvedPath\n";

      # Output the processed data
#      print OUTPUT "$timestamp | $deltaStart | $deltaStop |";
#      print OUTPUT " | $newCallpath | $newCallstack\n";

    }
  }


  foreach my $e (@otf_callstack) {
    my $event_id = getEventId($e);
    ebs2otf::OTF_Writer_writeLeave( $writer, $lasttimestamp, $event_id, $cpuid, 0);
  }


  print OUTPUT "# node: $node\n";
  print OUTPUT "# thread: $thread\n";
  if ($negativeSamples > 0) {
    print "$negativeSamples negative runtime deltas ignored out of $totalSamples total samples\n"
  }
}

sub main {
  if (defined $ARGV[0] && ($ARGV[0] eq "--aggregate" || $ARGV[0] eq "-a")) {
    $aggregate_to_routine = 1;
    print "Aggregating samples to routine level...\n";
  }
  my $pattern = "ebstrace.raw.*.*.*.*";
  while (defined(my $filename = glob($pattern))) {
    my ($trace_file, $def_file, $out_file);
    $trace_file = $filename;
    my ($junk1,$junk2,$pid,$nid,$cid,$tid) = split('\.',$filename);
    $def_file = "ebstrace.def.$pid.$nid.$cid.$tid";
    $out_file = "ebstrace.processed.$pid.$nid.$cid.$tid";
    print "processing $filename ...\n";
    process_trace($def_file, $trace_file, $out_file);
  }
  print "...done.\n";




  my $groupidx = 20;
  my %grouphash;

  ebs2otf::OTF_Writer_writeDefFunctionGroup( $writer, 0, 16, "standard functions");

  my ($name, $gid, $fid);
  while (($name, $fid) = each(%otf_function_map)) {



    ## all in "standard functions"
    #ebs2otf::OTF_Writer_writeDefFunction( $writer, 0, $fid, $name, 16, 0);

    ## each in its own group
    # if ($name =~ m/\@\@\@/) {
    #   my ($func,$file) = split ("@@@", $name);
    #   $name = $func;
    # }
    # $gid = $fid + 100;
    # ebs2otf::OTF_Writer_writeDefFunctionGroup( $writer, 0, $gid, $name);
    # ebs2otf::OTF_Writer_writeDefFunction( $writer, 0, $fid, $name, $gid, 0);

    ## per file
    $gid = 16;
    if ($name =~ m/\@\@\@/) {
      my ($func,$file) = split ("@@@", $name);

      $name = $func;

      if (defined $grouphash{$file}) {
    	$gid = $grouphash{$file};
      } else {
    	$gid = $groupidx;
    	$groupidx = $groupidx+1;
    	$grouphash{$file} = $gid;

    	ebs2otf::OTF_Writer_writeDefFunctionGroup( $writer, 0, $gid, $file);
      }
    }

    ebs2otf::OTF_Writer_writeDefFunction( $writer, 0, $fid, $name, $gid, 0);

  }




  ebs2otf::OTF_Writer_close( $writer );
}

main

