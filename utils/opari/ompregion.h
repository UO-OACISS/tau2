/*************************************************************************/
/* OPARI Version 1.1                                                     */
/* Copyright (c) 2001-2005                                                    */
/* Forschungszentrum Juelich, Zentralinstitut fuer Angewandte Mathematik */
/*************************************************************************/

#ifndef OMPREGION_H
#define OMPREGION_H

#include <iostream>
  using std::ostream;
#include <vector>
  using std::vector;
#include <string>
  using std::string;

class OMPRegion {
public:
  OMPRegion(const string& n, const string& file, int i, int bfl, int bll,
            bool outr = false);

  static void generate_header(ostream& os);

  void generate_descr(ostream& os);

  void finish();

  string name;
  string file_name;
  string sub_name;
  int    id;
  int    begin_first_line;
  int    begin_last_line;
  int    end_first_line;
  int    end_last_line;
  int    num_sections;
  bool   noWaitAdded;
  bool   outer_reg;
  vector<int> descrs;
  OMPRegion* enclosing_reg;

  static OMPRegion* outer_ptr;
};

#endif
