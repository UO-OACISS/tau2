/*
This file is part of the phiprof library

Copyright 2012, 2013, 2014, 2015 Finnish Meteorological Institute
Copyright 2015, 2016 CSC - IT Center for Science 

Phiprof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PHIPROF_HPP
#define PHIPROF_HPP

#include "string"
#include "vector"
#include "mpi.h"

   

/* This files contains the C++ interface */

namespace phiprof
{
   /**
    * Initialize phiprof
    *
    * This function should be called before any other calls to phiprof.
    * @return
    *   Returns true if phiprof started successfully.
    */
   bool initialize();

   /**
    * Initialize a timer, with a particular label   
    *
    * Initialize a timer. This enables one to define groups, and to use
    * the return id value for more efficient starts/stops in tight
    * loops. If this function is called for an existing timer it will
    * simply just return the id.
    *
    *
    * @param label
    *   Name for the timer to be created. This will not yet start the timer, that has to be done separately with a start. 
    * @param groups
    *   The groups to which this timer belongs. Groups can be used combine times for different timers to logical groups, e.g., MPI, io, compute...
    * @return
    *   The id of the timer
    */
   int initializeTimer(const std::string &label,const std::vector<std::string> &groups);
   /**
    * \overload int phiprof::initializeTimer(const std::string &label,const std::vector<std::string> &groups)
    */
   int initializeTimer(const std::string &label);
   /**
    * \overload int phiprof::initializeTimer(const std::string &label,const std::vector<std::string> &groups)
    */
   int initializeTimer(const std::string &label,const std::string &group1);
   /**
    * \overload int phiprof::initializeTimer(const std::string &label,const std::vector<std::string> &groups)
    */
   int initializeTimer(const std::string &label,const std::string &group1,const std::string &group2);
   /**
    * \overload int phiprof::initializeTimer(const std::string &label,const std::vector<std::string> &groups)
    */
   int initializeTimer(const std::string &label,const std::string &group1,const std::string &group2,const std::string &group3);


   /**
    * Get id number of an existing timer that is a child of the currently
    * active one
    *
    * @return
    *  The id of the timer. -1 if it does not exist.
    */
   int getChildId(const std::string &label);
   
   /**
    * Start a profiling timer.
    *
    * This function starts a timer with a certain label (name). If
    * timer does not exist, then it is created. The timer is
    * automatically started in the current active location in the tree
    * of timers. Thus the same start command in the code can start
    * different timers, if the current active timer is different.
    *
    * @param label
    *   Name for the timer to be start. 
    * @return
    *   Returns true if timer started successfully.
    */
   bool start(const std::string &label);
   
   /*
    * \overload bool phiprof::start(const std::string &label)
    */
   bool start(int id);
   
   /**
    * Stop a profiling timer.
    *
    * This function stops a timer with a certain label (name). The
    * label has to match the currently last opened timer. One can also
    * (optionally) report how many workunits was done during this
    * start-stop timed segment, e.g. GB for IO routines, Cells for
    * grid-based solvers. Note, all stops for a particular timer has to
    * report workunits, otherwise the workunits will not be reported.
    *
    * @param label 
    *   Name for the timer to be stopped.     
    * @param workunits 
    *   (optional) Default is for no workunits to be
    *   collected.Amount of workunits that was done during this timer
    *   segment. If value is negative, then no workunit statistics will
    *   be collected.
    * @param workUnitLabel
    *   (optional) Name describing the unit of the workunits, e.g. "GB", "Flop", "Cells",...
    * @return
    *   Returns true if timer stopped successfully.
    */
   bool stop (const std::string &label, double workUnits=-1.0, const std::string &workUnitLabel="");

   /**
    * \overload  bool phiprof::stop(const std::string &label,double workUnits=-1.0,const std::string &workUnitLabel="")
    */
   bool stop (int id, double workUnits, const std::string &workUnitLabel);
   
   /**
    * Fastest stop routine for cases when no workunits are defined.
    */
   bool stop (int id);


   /**
    * Print the  current timer state in a human readable file
    *
    * This function will print the timer statistics in a text based
    * hierarchical form into file(s), each unique set of hierarchical
    * profiles (labels, hierarchy, workunits) will be written out to a
    * separate file. This function will print the times since the
    * ininitalization of phiprof in the first start call. It can be
    * called multiple times, and will not close currently active
    * timers. The time spent in active timers uptill the print call is
    * taken into account, and the time spent in the print function will
    * be corrected for in them.
    *
    *
    * @param comm
    *   Communicator for processes that print their profile.
    * @param fileNamePrefix
    *   (optional) Default value is "profile"
    *   The first part of the filename where the profile is printed. Each
    *   unique set of timers (label, hierarchy, workunits) will be
    *   assigned a unique hash number and the profile will be written
    *   out into a file called fileprefix_hash.txt
    * @return
    *   Returns true if pofile printed successfully.
    */
   bool print(MPI_Comm comm, std::string fileNamePrefix="profile");

}


#endif
