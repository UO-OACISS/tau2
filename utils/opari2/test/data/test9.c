/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013, 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 * Testfile for automated testing of OPARI2
 *
 * @brief Tests proper treatment of offload regions.
 */

__declspec( target( mic ) ) int global_1 = 5;

__declspec( target( mic ) ) int bar();

__declspec( target( mic ) ) int foo()
{
	int i = 0;
	#pragma omp parallel
	{
		i++;
	}

	return ++global_1;
}

__attribute__( ( target( mic ) ) ) int global_2 = 0;
__attribute__( ( target( mic ) ) ) int f()
{
	int i = 0;
	#pragma omp atomic
	global_2 += 1;
}

__attribute__( ( target( mic ) ) ) void g();


#pragma offload_attribute( push, target( mic ) )
void test();
#pragma offload_attribute( pop )


void test()
{
	int i;

	#pragma omp sections
	{
		i++;
		#pragma omp section
		i++;
		#pragma omp section
		i++;
	}
}


int
main( int argc, char** argv )
{
	int i, j, k;

	#pragma omp parallel for
	for ( i = 0; i < 10; i++ )
	{
	 	j++;
	}

	#pragma offload target( mic ) in( global ) out( i, global )
	{
		i = foo();
		#pragma omp for
		for ( j = 0; j < 10; j++ )
		{
			k ++;
		}
	}

	#pragma offload target( mic ) in( global ) out( i, global )
	{
		i = bar();
	}

	#pragma offload_attribute( push, target( mic ) )

	#pragma omp parallel
	{
		i = f();
		g();
		test();
	}

	#pragma offload_attribute( pop )


	#pragma omp barrier

	printf( "Hello world!\n" );

	return 0;
}

int bar()
{
	#pragma omp single
		global_1++;

	return global_1;
}

void g()
{
	#pragma omp master
	global_2 = 0;
}
