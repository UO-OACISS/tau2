#!/usr/bin/env perl

# keys are all complete and partial callpaths
# values are the beginnings of the (partial) callpaths
%callpaths = ();

# list of all functions that call 
# MPI routines directly or indirectly
@mpifuncs = ();


open(PPROF, "pprof |")  or die "Could not start pprof!\n";
while (<PPROF>) 
{
    if( /([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(.*)\s=>\s(.*)$/ )
    {
	@callers=split(/=>/,$7);
	$callee=$8;

	# trim whitespace at beginning and end
	$callee =~ s/^\s+//;
	$callee =~ s/\s+$//;
	foreach $caller (@callers)
	{
	    # trim whitespace at beginning and end
	    $caller =~ s/^\s+//;
	    $caller =~ s/\s+$//;
	    
	    #print "$caller --> $callee\n";
	    $callpaths{$caller." => ".$callee} = $caller;
	    if( $callee =~ /^MPI_/ )
	    {
		push( @mpifuncs, $caller );
	    }
	    
	    foreach $path (keys %callpaths)
	    {
		# see if it is a path that ends with $caller
		if( $path =~ /\Q$caller\E$/ )
		{
		    # add extended callpath with same start
		    $callpaths{$path." => ".$callee} = $callpaths{$path};
		    
		    if( $callee =~ /^MPI_/ )
		    {
			push( @mpifuncs, $callpaths{$path." => ".$callee} );
		    }
		}
	    }
	}
    }
}

my @uniq = keys %{{ map { $_ => 1 } @mpifuncs }};

print "BEGIN_INCLUDE_LIST\n";
foreach $func (@uniq)
{
    print "$func\n";
}
print "END_INCLUDE_LIST\n";
