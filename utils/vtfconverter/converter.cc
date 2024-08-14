/************************************************************************
************************************************************************/

/************************************************************************
	converter.cc
	Author: Ryan K. Harris
	
	A driver for the VTF format conversion program.
************************************************************************/

#include <iostream>
#include <string>
#include <stdlib.h>
#include "readVTF.h"
using namespace std;

void displayHelp(void);
void displayCommandLine(void);
void readFile(string * n_fileName, string * n_destPath, double n_startTime, double n_endTime);

int main(int argc, char ** argv)
{
	char * destPath = "./";
	char * fileName = "none";
	char * intervalStart = 0;
	char * intervalEnd = 0;
	int help = 0;
	int commandLine = 0;
	int file = 0;
	int path = 0;
	int interval = 0;
	int numParams = argc - 1;
	double startTime = -1;
	double endTime = -1;

	/* If no args provided, tell user. */
	if(argc == 1)
	{
		cout <<"Usage: "<<argv[0]<<" [options] "<<endl;
		displayHelp();
		return(0);
	}

	/* Only two of the args can work together. */
	if(argc > 8)
	{
		cout << "Too many arguments provided at once\n";
		return(0);
	}

	/* Parse the command line options. */
	while( (argc > 1) && (argv[1][0] == '-'))
	{
		//cerr << "#numParams: " << numParams << "\n";

		if(argv[1][1] == 'h' || argv[1][1] == 'H')
		{
			help = 1;
			--numParams;
		}

		else if(argv[1][1] == 'c' || argv[1][1] == 'C')
		{
			commandLine = 1;
			--numParams;
		}

		else if(argv[1][1] == 'f' || argv[1][1] == 'F')
		{
			file = 1;

			if(numParams > 0)
			{
				fileName = &argv[1][2];
				--numParams;
			}

			if((*fileName) == 0)
			{
				if(numParams <= 0)
				{
					cout << "No file name provided with '-f' file option\n";
					return(1);
				}

				/* Check to see if next arg is the parameter
				associated to this option. */
				if(argv[2][0] != '-')
				{
					fileName = &argv[2][0];
					argc -= 2;
					argv += 2;
					--numParams;
					continue;
				}
				else
				{
					cout << "No file name provided with '-f' file option\n";
					return(1);
				}//else
			}//if
		}

		else if(argv[1][1] == 'p' || argv[1][1] == 'P')
		{
			path = 1;

			if(numParams > 0)
			{
				destPath = &argv[1][2];
				--numParams;
			}

			if((*destPath) == 0)
			{
				if(numParams <= 0)
				{
					cout << "No path provided with '-p' path option\n";
					return(1);
				}

				/* Check to see if next arg is the parameter
				associated to this option. */
				if(argv[2][0] != '-')
				{
					destPath = &argv[2][0];
					argc -= 2;
					argv += 2;
					--numParams;
					continue;
				}
				else
				{
					cout << "No path provided with '-p' path option\n";
					return(1);
				}//else
			}//if
		}

		else if(argv[1][1] == 'i' || argv[1][1] == 'I')
		{
			interval = 1;

			if(numParams > 0)
			{
				intervalStart = &argv[1][2];
				--numParams;
			}

			if((*intervalStart) == 0)
			{
				if(numParams <= 0)
				{
					cout << "No times provided with '-i' interval option\n";
					return(1);
				}

				/* Check to see if next arg is the parameter
				associated to this option. */
				if((argv[2][0] != '-') && (argv[3][0] != '-'))
				{
					intervalStart = &argv[2][0];
					intervalEnd = &argv[3][0];
					argc -= 3;
					argv += 3;
					numParams -= 2;

					/* Need to convert to double. */
					const char * constIntStart = intervalStart;
					const char * constIntEnd = intervalEnd;
					startTime = atof(constIntStart);
					endTime = atof(constIntEnd);

					/* Must verify that startTime < endTime. */
					if(endTime < startTime)
					{
						cout << "Interval ending time ("
						<< endTime << ") is less than starting time("
						<< startTime << "). Exiting.\n";
						return(-1);
					}//if

					continue;
				}
				else
				{
					cout << "No times provided with '-i' interval option\n";
					return(1);
				}//else
			}//if
			else
			{
				/* Assume we acquired intervalStart, get intervalEnd. */
				if(argv[2][0] != '-')
				{
					intervalEnd = &argv[2][0];
					argc -= 2;
					argv += 2;
					--numParams;

					/* Need to convert to double. */
					const char * constIntStart = intervalStart;
					const char * constIntEnd = intervalEnd;
					startTime = atof(constIntStart);
					endTime = atof(constIntEnd);

					/* Must verify that startTime < endTime. */
					if(endTime < startTime)
					{
						cout << "Interval ending time ("
						<< endTime << ") is less than starting time("
						<< startTime << "). Exiting.\n";
						return(-1);
					}//if

					continue;
				}//if
				else
				{
					cout << "Ending time not provided with '-i' interval option,\n"
					<< "or it was unreadable make sure options are seperated "
					<< "by a space\n";
					return(-1);
				}//else
			}//else
		}//else if( '-i')

		/* Take into account that some bum might have passed us a null
			where we expected an option letter, thus we --numParams. */

		--argc;
		++argv;
		--numParams;
	}//while

	/* Print the ones we received so that we can see
		what the program is seeing. */

	/*
	cout << "help on: " << help << "\n";
	cout << "file on: " << file << "\n";
	cout << "path on: " << path << "\n";
	*/

	/*////////////COMMENTING OUT THE FOLLOWING DEBUGGING
	cout << "interval on: " << interval << "\n";

	cout << "filename: " << fileName << "\n";
	cout << "path: " << destPath << "\n";


	cout << "interval: " << intervalStart << " - "
		<< intervalEnd << "\n";

	cout << "interval after conversion: "
		<< startTime << " - " << endTime << "\n";

	*//////////////////////

	/* Process the commandline Options. */
	if(help)
	{
		displayHelp();
		return(0);
	}
	if(commandLine)
	{
		displayCommandLine();
		return(0);
	}
	if(file)
	{
		if(fileName == 0)
		{
			cout << "No file name provided.\n";
			return(0);
		}

		string the_fileName(fileName);
		string the_destPath(destPath);

		readFile(&the_fileName, &the_destPath, startTime, endTime);
		return(0);
	}

	return(0);
}//main

/************************************************************************
	void displayHelp(void)
		--Displays the the help text on screen.
		
	Parameters: none
************************************************************************/
void displayHelp(void)
{
	cout 	<< "***************************HELP***************************\n"
	<< "* '-h' --display this help text.                         *\n"
	<< "* '-c' --open command line interface.                    *\n"
	<< "* '-f' --used as -f <VTF File> where                     *\n"
	<< "*        VTF File is the name of the trace file          *\n"
	<< "*        to be converted to TAU profiles.                *\n"
	<< "* '-p' --used as -p <path> where 'path' is the relative  *\n"
	<< "*        path to the directory where profiles are to     *\n"
	<< "*        stored.                                         *\n"
	<< "* '-i' --used as -i <from> <to> where 'from' and 'to' are*\n"
	<< "*        integers to mark the desired profiling interval.*\n"
	<< "**********************************************************\n";
	return;
}//displayHelp

/************************************************************************
************************************************************************/
void displayCommandLine(void)
{
	char command;
	char destPath[100];
	char fileName[100];
	char * intervalStart;
	char * intervalEnd;
	double startTime = -1;
	double endTime = -1;

	strncpy(destPath,  "./", sizeof(destPath)); 

	while(1)
	{
		cout	<< "***********************COMMANDS***********************\n"
		<< "* 'H' --display help text.                           *\n"
		<< "* 'R' --read a file.                                 *\n"
		<< "* 'S' --set the destination path for profiles.       *\n"
		<< "* 'P' --view current destination path.               *\n"
		<< "* 'I' --set an desired time interval for profiles.   *\n"
		<< "* 'Q' --quit.                                        *\n"
		<< "******************************************************\n";

		cin >> command;

		if(command == 'h' || command == 'H')
		{
			cout 	<< "***************************HELP***************************\n"
			<< "* 'R' --read a file.                                     *\n"
			<< "*       This will prompt user with:                      *\n"
			<< "*           'type file name: '                           *\n"
			<< "*       At the prompt, type the complete name of         *\n"
			<< "*       any single VTF you would like the converter      *\n"
			<< "*       to translate into TAU Profile format.            *\n"
			<< "* 'S' --set the destination path for profiles.           *\n"
			<< "*       This will prompt user with:                      *\n"
			<< "*           'type path: '                                *\n"
			<< "*       At the prompt, type the relative path of the     *\n"
			<< "*       directory where the converter should place the   *\n"
			<< "*       generated profiles.                              *\n"
			<< "* 'P' --view current destination path.                   *\n"
			<< "*       Displays the path that the converter will use as *\n"
			<< "*       it is currently set.                             *\n"
			<< "* 'I' --set a time interval (t1 - t2). Only events       *\n"
			<< "*       occurring within this interval will be profiled. *\n"
			<< "*       User will be prompted by:                        *\n"
			<< "*           'set intervalStart:'                         *\n"
			<< "*       then,                                            *\n"
			<< "*           'set intervalEnd:'                           *\n"
			<< "**********************************************************\n";
			continue;
		}

		if(command == 'r' || command == 'R')
		{
			cout << "type file name:\n";
			cin >> fileName;
			cout << "file name entered: " << fileName << "\n";

			string the_fileName(fileName);
			string the_destPath(destPath);

			readFile(&the_fileName, &the_destPath, startTime, endTime);
			continue;
		}

		if(command == 's' || command == 'S')
		{
			cout << "type path:\n";
			cin >> destPath;
			cout << "path entered: " << destPath << "\n";
			continue;
		}

		if(command == 'p' || command == 'P')
		{
			cout << "destination path: " << destPath << "\n";
			continue;
		}

		if(command == 'i' || command == 'I')
		{
			cout << "set intervalStart: \n";
			cin >> intervalStart;
			cout << "set intervalEnd: \n";
			cin >> intervalEnd;
			cout << "intervalStart: " << intervalStart << "\n";
			cout << "intervalEnd: " << intervalEnd << "\n";

			const char * n_intervalStart = intervalStart;
			const char * n_intervalEnd = intervalEnd;
			startTime = atof(n_intervalStart);
			endTime = atof(n_intervalEnd);

			continue;
		}

		if(command == 'q' || command == 'Q')
		{
			cout << "Done\n";
			return;
		}

		else
		{
			continue;
		}
	}//while

	return;
}//displayCommandLine

/************************************************************************
************************************************************************/
void readFile(string * n_fileName, string * n_destPath, double n_startTime, double n_endTime)
{
	string fileName;
	string destPath;
	fileName = *n_fileName;
	destPath = *n_destPath;

	readVTF * reader = new readVTF(&fileName, &destPath, n_startTime, n_endTime);
	(*reader).readfile();
	delete reader;

	return;
}//readFile(fileName, destPath)

