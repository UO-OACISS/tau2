#!/usr/bin/env perl

########################################################################
# This script reads an EBS trace and converts the PC's and TAU callpaths
########################################################################

use strict;
use IO::Handle;


# Read only version 0.2
my $reader_version = 0.2;

my ($forked);
$forked = 0;

##### create pipes to handle communication
pipe (FROM_PERL, TO_PROGRAM);
pipe (FROM_PROGRAM, TO_PERL);
TO_PROGRAM->autoflush(1);
TO_PERL->autoflush(1);

# Trim leading and trailing whitespace from a string
sub trim($) {
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

my %pcmap;

my @address_maps;

# Translate a PC value
sub old_translate_pc {
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

      close  STDERR;
      open  (STDERR, '>&STDOUT')    || die;

      ##### close unused parts of pipes
      close FROM_PROGRAM;
      close TO_PROGRAM;

      ##### unbuffer the outputs
      select STDERR; $| = 1;
      select STDOUT; $| = 1;


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
  my $func = <FROM_PROGRAM>;
  my $fileline = <FROM_PROGRAM>;

  chomp($func);
  chomp($fileline);
  $pcmap{$pc} = "$func:$fileline";
  return "$func:$fileline";
}

# Translate a PC value
sub translate_pc {
  my ($exe, $pc, $strip) = @_;
  if (defined $pcmap{$pc}) {
      return "$pcmap{$pc}";
  }
#  print "Looking to translate $pc\n";

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

      close  STDERR;
      open  (STDERR, '>&STDOUT')    || die;

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
  my $func = <FROM_PROGRAM>;
  if ($func =~ m/^BFD: Dwarf Error/) {
    $func = "unknown";
    $fileline = "unknown:??";
  } else {
    $fileline = <FROM_PROGRAM>;
  }

  chomp($func);
  chomp($fileline);

  if ($func eq "??") {
    foreach my $rec (@address_maps) {
      #print "between $rec->{ADDR_START} and $rec->{ADDR_END} ???\n";
      if (hex $pc > hex $rec->{ADDR_START} &&  hex $pc < hex $rec->{ADDR_END}) {
	#print "yes!\n";
	#print "$rec->{BINARY}\n";

	my $newpc = (hex $pc) - (hex $rec->{ADDR_START});
	#print "newpc = $newpc\n";


	#print "Hexadecimal number: ", uc(sprintf("%x\n", $newpc)), "\n";

	$newpc = uc(sprintf("%x", $newpc));


	# write the pc to addr2line
	my $outhandle = $rec->{TO_CONVERTER};

	print $outhandle "$newpc\n";

	# read the result
	my $handle = $rec->{FROM_CONVERTER};


	my $func = <$handle>;
	my $fileline = <$handle>;

	chomp($func);
	chomp($fileline);

	$pcmap{$pc} = "$func:$fileline";
	#print "returning $pcmap{$pc}\n";
	return "$func:$fileline";

      } else {
#	print "no!\n";
      }
    }
  }


  if ($strip == 1) {
    # strip the path
    my ($file,$line) = split (":", $fileline);
    $file =~ s!^.*/([^/]*)$!\1!;
    $fileline = "$file:$line";
  }

  $pcmap{$pc} = "$func:$fileline";
#  print "returning $pcmap{$pc}\n";

#  print "got func: $func\n";
#  print "got fileline: $fileline\n";
  return "$func:$fileline";
}


sub read_maps {
  my ($exe, $map_file) = @_;
#  print "reading maps\n";
  my @lines = `cat $map_file`;
  foreach my $line (@lines) {
    chomp($line);
    my ($binary, $start, $end, $offset) = split (" ", $line);
#    print "got line: $line\n";

    if ($binary eq "[stack]") {
      next;
    }


    pipe (SUB_FROM_PERL, SUB_TO_PROGRAM);
    pipe (SUB_FROM_PROGRAM, SUB_TO_PERL);
    SUB_TO_PROGRAM->autoflush(1);
    SUB_TO_PERL->autoflush(1);

#    print "addr2line -C -f -e $binary\n";
    my $pid = fork;


    if ($pid == 0) {
      ##### attach standard input/output/error to the pipes
      close  STDIN;
      open  (STDIN,  '<&SUB_FROM_PERL') || die ("open: $!");

      close  STDOUT;
      open  (STDOUT, '>&SUB_TO_PERL')   || die ("open: $!");

      close  STDERR;
      open  (STDERR, '>&STDOUT')    || die;

      ##### close unused parts of pipes
      close SUB_FROM_PROGRAM;
      close SUB_TO_PROGRAM;

      ##### unbuffer the outputs
      select STDERR; $| = 1;
      select STDOUT; $| = 1;


      ##### execute the program
      exec "addr2line -C -f -e $binary";

      ##### shouldn't get here!!!
      die;
    }

    close SUB_FROM_PERL;
    close SUB_TO_PERL;


    my $rec = {
	    BINARY => $binary,
	    ADDR_START => $start,
	    ADDR_END => $end,
	    OFFSET => $offset,
	    TO_CONVERTER => \*SUB_TO_PROGRAM,
	    FROM_CONVERTER => \*SUB_FROM_PROGRAM,
	   };
    push (@address_maps,$rec);


  }
}


# process an EBS trace file
sub process_trace {
  my($def_file, $trace_file, $map_file, $out_file, $strip) = @_;

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


  # Read the trace
  open (TRACE, "<$trace_file");
  open (OUTPUT, ">$out_file");

  print OUTPUT "# thread: $thread\n";
  print OUTPUT "# node: $node\n";

  while ($line = <TRACE>) {
    #print "$line";
    if ($line =~ /\#.*/) {
      if ($line =~ /\# exe:.*/) {
        ($junk, $exe) = split("exe:",$line);
        $exe = trim($exe);
	read_maps($exe, $map_file);
      } elsif ($line =~ /\# node:.*/) {
        ($junk, $node) = split("node:",$line);
        $node = trim($node);
      } elsif ($line =~ /\# Format.*/) {
        ($junk, $version) = split("version:",$line);
	$version = trim($version);
	if ($version != $reader_version) {
	  die "This reader is only for version $reader_version files, you have $version, sorry\n";
	}
        print (OUTPUT "$line");
      } elsif ($line =~ /\# thread:.*/) {
        ($junk, $thread) = split("thread:",$line);
        $thread = trim($thread);
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

      # Process the callpath
      my @events = split(" ",$callpath);
      my @tmpCallpath = @events;
      @events = reverse (@events);

      my @callstackEntries = split(" ",$callstack);


	# build a key into the map
	$callpath = "";
        foreach my $t (@tmpCallpath) {
          $callpath = "$callpath $t";
        }
	$callpath = trim($callpath);

        my $newCallpath = "";
        my (@processedEvents);
        foreach my $e (@events) {
          $newCallpath = "$newCallpath => $eventmap{$e}";
          push (@processedEvents, $eventmap{$e});
        }
        $newCallpath = join(" => ", @processedEvents);
        $lastCallpath = $newCallpath;

        my $check = $deltaStart;

        @startTokens = split(" ", $startmap{$callpath});
        @stopTokens = split(" ", $stopmap{$callpath});
        $deltaStart = $timestamp - @startTokens[0];
        $deltaStop = @stopTokens[0] - $timestamp;
	#       if ($check != $deltaStart) {
	#         print "$line\n";
	#         die "inconsistent file $callpath, $check != $$deltaStart\n";
	#       }

	# Process the callstack
	my $newCallstack = "";
        foreach my $cs (@callstackEntries) {
	  my $loc = translate_pc($exe, $cs, $strip);
          $newCallstack = "$newCallstack$loc\@";
        }
	chop ($newCallstack); # remove the last @


        # Output the processed data
        print OUTPUT "$timestamp | 0 | 0 |";

        # split the metrics, and their start/stops, and handle them all
        my @metricTokens = split(" ", $metrics);
        my $i = 0;
        for (0..$#metricTokens) {
          my $deltaMetS = $metricTokens[$_] - $startTokens[$_];
          my $deltaMetE = $stopTokens[$_] - $metricTokens[$_];
          print OUTPUT " 0 0";
        }

        print OUTPUT " | $newCallpath | $newCallstack\n";
      $totalSamples = $totalSamples + 1;
    }
  }
  if ($negativeSamples > 0) {
    print "$negativeSamples negative runtime deltas ignored out of $totalSamples total samples\n"
  }
}

sub main {
  my $strip = 1;
  if (defined $ARGV[0] && ($ARGV[0] == "--nostrip" || $ARGV[0] == "-nos")) {
    $strip = 0;
    print "Processing samples... (path strip = $strip)\n";
  }
  my $pattern = "ebstrace.raw.*.*.*.*";
  while (defined(my $filename = glob($pattern))) {
    my ($trace_file, $def_file, $map_file, $out_file);
    $trace_file = $filename;
    my ($junk1, $junk2);
    my ($junk1,$junk2,$pid,$nid,$cid,$tid) = split('\.',$filename);
    $def_file = "ebstrace.def.$pid.$nid.$cid.$tid";
    $map_file = "ebstrace.map.$pid.$nid.$cid.$tid";
    $out_file = "ebstrace.processed.$pid.$nid.$cid.$tid";
    print "processing $filename ...\n";
    process_trace($def_file, $trace_file, $map_file, $out_file, $strip);
  }
  print "...done.\n";
}

main
