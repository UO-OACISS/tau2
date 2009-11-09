#!/usr/bin/env perl

########################################################################
# This script reads an EBS trace and converts the PC's and TAU callpaths
########################################################################

use strict;
use IO::Handle;


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

# Translate a PC value
sub translate_pc {
  my ($exe, $pc) = @_;

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
  return "$func:$fileline"
}

# process an EBS trace file
sub process_trace {
  my($def_file, $trace_file, $out_file, $inclusive) = @_;

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

  # Read the trace
  my ($junk, $exe, $node);
  open (TRACE, "tac $trace_file |");
  open (OUTPUT, "| tac > $out_file");
  while ($line = <TRACE>) {
    #print "$line";
    if ($line =~ /\#.*/) {
      if ($line =~ /\# exe:.*/) {
        ($junk, $exe) = split("exe:",$line);
        $exe = trim($exe);
      } elsif ($line =~ /\# node:.*/) {
        ($junk, $node) = split("node:",$line);
        $node = trim($node);
      } elsif ($line =~ /\# \$.*/) {
	    ## ignore the format line
      } elsif ($line =~ /\# \%.*/) {
	    ## output the true format line
		print (OUTPUT "# <timestamp> | <delta-begin> | <delta-end> | <location> | <delta-begin metric 1> <delta-end metric 1> ... <delta-begin metric N> <delta-end metric N> | <tau callpath>\n");
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

      my ($type,$timestamp,$deltaStart,$deltaStop,$pc,$metrics,$callpath) = split('\|',$line);
      $timestamp = trim($timestamp);
      $pc = trim($pc);
      $metrics = trim($metrics);
      $callpath = trim($callpath);

      # Process the callpath
      my @events = split(" ",$callpath);
      my @tmpCallpath = @events;
      @events = reverse (@events);

      for (0..$#events) {

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
  
        # Process the PC

        my $newpc = translate_pc($exe, $pc);

        if (($deltaStop < 0) || ($deltaStart < 0)) {
          #print "ignoring negative sample, location: $newpc, callpath: $newCallpath\n";
		  $negativeSamples = $negativeSamples + 1;
		  if ($inclusive) {
		    last;
		  }
		  pop(@events);
		  shift(@tmpCallpath);
        }

        # Output the processed data
        print OUTPUT "$timestamp | $deltaStart | $deltaStop | $newpc |";

        # split the metrics, and their start/stops, and handle them all
        my @metricTokens = split(" ", $metrics);
        my $i = 0;
        for (0..$#metricTokens) {
          my $deltaMetS = $metricTokens[$_] - $startTokens[$_];
          my $deltaMetE = $stopTokens[$_] - $metricTokens[$_];
          print OUTPUT " $deltaMetS $deltaMetE";
        }

        print OUTPUT " | $newCallpath\n";

		if (!$inclusive) {
		  last;
		}
		pop(@events);
		shift(@tmpCallpath);
      }
	  $totalSamples = $totalSamples + 1;
    }
  }
  print OUTPUT "# node: $node\n";
  if ($negativeSamples > 0) {
    print "$negativeSamples negative runtime deltas ignored out of $totalSamples total samples\n"
  }
}

sub main {
  my $inclusive = 0;
  if (defined $ARGV[0] && ($ARGV[0] == "--inclusive" || $ARGV[0] == "-i")) {
    $inclusive = 1;
    print "Processing inclusive samples...\n";
  }
  my $pattern = "ebstrace.raw.*.*.*.*";
  while (defined(my $filename = glob($pattern))) {
    my ($trace_file, $def_file, $out_file);
    $trace_file = $filename;
    my ($junk1, $junk2);
    my ($junk1,$junk2,$pid,$nid,$cid,$tid) = split('\.',$filename);
    $def_file = "ebstrace.def.$pid.$nid.$cid.$tid";
    $out_file = "ebstrace.processed.$pid.$nid.$cid.$tid";
    print "processing $filename ...\n";
    process_trace($def_file, $trace_file, $out_file, $inclusive);
  }
  print "...done.\n";
}

main
