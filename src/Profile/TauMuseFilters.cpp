/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.acl.lanl.gov/tau                        **
 * *****************************************************************************
 * **    Copyright 2003                                                       **
 * **    Department of Computer and Information Science, University of Oregon **
 * **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauMuseFilters.cpp                             **
 * **      Description     : TAU MUSE/MAGNET Interface                      **
 * **      Author          : Suravee Suthikulpanit                          **
 * **      Contact         : Suravee@cs.uoregon.edu                         **
 * **      Flags           : Compile with                                   **
 * **                        -DTAU_MUSE                                     **
 * ****************************************************************************/
// NOTE: This file contains the filters for TAU/MUSE. 
#include <Profile/Profiler.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>


/*********************
 * Description	: Encode binary code for command addfilter 
 * 		  which wil be sent to Muse server
 * From		: translator.c <modified>
 * Return Value	: the number of bytes in binary_command
 * NOTE		: To be used with process_filter filter.
 *********************/
int AddFilterProcessFilterEncode(char *ascii_command, int size, char *binary_command)
{
	char *arg, *val, temp[MAX_ARGLEN];
	unsigned int event, id, pid;
	int *intptr, *options_size;
	unsigned char *byteptr;
	char *option_flags;
	strncpy(temp, ascii_command, MAX_ARGLEN);
	arg = strtok(temp, " ");
	if (arg)
	{
		if (strcasecmp(arg, "addfilter")==0)
		{
			binary_command[0] = 8;
			arg = strtok(NULL, " ");
			if (arg)
			{
				// handler_ID
				byteptr = (unsigned char *)&binary_command[sizeof(unsigned char)];
				*byteptr = (unsigned char)atoi(arg);
				arg = strtok(NULL, " ");
				if ((!arg)||(strcasecmp(arg, "process_filter")!=0))
				{
					printf("Internal error");
					return 0;
				}
				else
				{
					binary_command[2*sizeof(unsigned char)] = strlen(arg);
					strncpy(&binary_command[3*sizeof(unsigned char)], arg, 
							binary_command[2*sizeof(unsigned char)]+1);
					arg = strtok(NULL, "=");
					val = strtok(NULL, " ");
					if (arg)
					{
						intptr=(int *) &binary_command[binary_command[2*sizeof(unsigned char)]
								+ 4*sizeof(unsigned char)];
						options_size = intptr++;
						*options_size = sizeof(int);
						option_flags = (char *)intptr;
						intptr++;
						*option_flags = 0;
						while (arg && val) {
							if(strcasecmp(arg, "event")==0) {
								*option_flags = *option_flags|0x80;
								event = strtoul(val, NULL, 10);
							} else { 
								if (strcasecmp(arg, "id")==0) {
									*option_flags = *option_flags|0x40;
									id = strtoul(val, NULL, 10);
									
								} else {
									if (strcasecmp(arg, "pid")==0) {
										*option_flags = *option_flags|0x20;
										pid = strtoul(val, NULL, 10);
									} else {
										printf("Unknown option: \"%s\"\n", arg);
										return 0;
									}
								}
							}
							*options_size += sizeof(int);
							
							arg = strtok(NULL, "=");
							val = strtok(NULL, " ");
						}
						if (arg) {
							printf("Could not parse option.  Ignoring option %s", arg);
						
							if (*option_flags == 0) {
								printf("No options specified");
								return 0;
							}
						}
						
						*options_size = htonl(*options_size);
						if (*option_flags&0x80)
							*intptr++ = htonl(event);
						if (*option_flags&0x40)
							*intptr++ = htonl(id);
						if (*option_flags&0x20)
							*intptr++ = htonl(pid);
						return (htonl(*options_size) + binary_command[1+sizeof(int)]
								+ 3 + 2*sizeof(int));
					} else {
						printf("No options specified");
					}
				}
			} else {
				printf("No handler specified");
			}
		} else {
			printf("Invalid command for process_filter: %s\n"
				  "process_filter is a filter, and should be used only with the ADDFILTER command\n"
				  "\tUsage: ADDFILTER <handler id> process_filter [pid=<pid> ]"
				  " [id=<id>] [event=<event type>]", arg);
		}
	} else {
		printf("Internal error.  No command received in encode_command");
	}
	return 0;
}

/* EOF */
