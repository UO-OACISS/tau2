/*************************************************************************/
/* TAU OPARI Disabled Layer    		                                 */
/*************************************************************************/

extern "C" {
  void pomp_finalize() {}
  void pomp_finalize_() {}
  void pomp_finalize__() {}
  void POMP_FINALIZE() {}

  void pomp_init() {}
  void pomp_init_() {}
  void pomp_init__() {}
  void POMP_INIT() {}

  void pomp_off() {}
  void pomp_off_() {}
  void pomp_off__() {}
  void POMP_OFF() {}

  void pomp_on() {}
  void pomp_on_() {}
  void pomp_on__() {}
  void POMP_ON() {}

  void pomp_atomic_enter() {}
  void pomp_atomic_enter_() {}
  void pomp_atomic_enter__() {}
  void POMP_ATOMIC_ENTER() {}

  void pomp_atomic_exit() {}
  void pomp_atomic_exit_() {}
  void pomp_atomic_exit__() {}
  void POMP_ATOMIC_EXIT() {}

  void pomp_barrier_enter() {}
  void pomp_barrier_enter_() {}
  void pomp_barrier_enter__() {}
  void POMP_BARRIER_ENTER() {}

  void pomp_barrier_exit() {}
  void pomp_barrier_exit_() {}
  void pomp_barrier_exit__() {}
  void POMP_BARRIER_EXIT() {}

  void pomp_critical_begin() {}
  void pomp_critical_begin_() {}
  void pomp_critical_begin__() {}
  void POMP_CRITICAL_BEGIN() {}

  void pomp_critical_end() {}
  void pomp_critical_end_() {}
  void pomp_critical_end__() {}
  void POMP_CRITICAL_END() {}

  void pomp_critical_enter() {}
  void pomp_critical_enter_() {}
  void pomp_critical_enter__() {}
  void POMP_CRITICAL_ENTER() {}

  void pomp_critical_exit() {}
  void pomp_critical_exit_() {}
  void pomp_critical_exit__() {}
  void POMP_CRITICAL_EXIT() {}

  void pomp_do_enter() {}
  void pomp_do_enter_() {}
  void pomp_do_enter__() {}
  void POMP_DO_ENTER() {}

  void pomp_do_exit() {}
  void pomp_do_exit_() {}
  void pomp_do_exit__() {}
  void POMP_DO_EXIT() {}

  void pomp_master_begin() {}
  void pomp_master_begin_() {}
  void pomp_master_begin__() {}
  void POMP_MASTER_BEGIN() {}

  void pomp_master_end() {}
  void pomp_master_end_() {}
  void pomp_master_end__() {}
  void POMP_MASTER_END() {}

  void pomp_parallel_begin() {}
  void pomp_parallel_begin_() {}
  void pomp_parallel_begin__() {}
  void POMP_PARALLEL_BEGIN() {}

  void pomp_parallel_end() {}
  void pomp_parallel_end_() {}
  void pomp_parallel_end__() {}
  void POMP_PARALLEL_END() {}

  void pomp_parallel_fork() {}
  void pomp_parallel_fork_() {}
  void pomp_parallel_fork__() {}
  void POMP_PARALLEL_FORK() {}

  void pomp_parallel_join() {}
  void pomp_parallel_join_() {}
  void pomp_parallel_join__() {}
  void POMP_PARALLEL_JOIN() {}

  void pomp_section_begin() {}
  void pomp_section_begin_() {}
  void pomp_section_begin__() {}
  void POMP_SECTION_BEGIN() {}

  void pomp_section_end() {}
  void pomp_section_end_() {}
  void pomp_section_end__() {}
  void POMP_SECTION_END() {}

  void pomp_sections_enter() {}
  void pomp_sections_enter_() {}
  void pomp_sections_enter__() {}
  void POMP_SECTIONS_ENTER() {}

  void pomp_sections_exit() {}
  void pomp_sections_exit_() {}
  void pomp_sections_exit__() {}
  void POMP_SECTIONS_EXIT() {}

  void pomp_single_begin() {}
  void pomp_single_begin_() {}
  void pomp_single_begin__() {}
  void POMP_SINGLE_BEGIN() {}

  void pomp_single_end() {}
  void pomp_single_end_() {}
  void pomp_single_end__() {}
  void POMP_SINGLE_END() {}

  void pomp_single_enter() {}
  void pomp_single_enter_() {}
  void pomp_single_enter__() {}
  void POMP_SINGLE_ENTER() {}

  void pomp_single_exit() {}
  void pomp_single_exit_() {}
  void pomp_single_exit__() {}
  void POMP_SINGLE_EXIT() {}

  void pomp_workshare_enter() {}
  void pomp_workshare_enter_() {}
  void pomp_workshare_enter__() {}
  void POMP_WORKSHARE_ENTER() {}

  void pomp_workshare_exit() {}
  void pomp_workshare_exit_() {}
  void pomp_workshare_exit__() {}
  void POMP_WORKSHARE_EXIT() {}

  void pomp_begin() {}
  void pomp_begin_() {}
  void pomp_begin__() {}
  void POMP_BEGIN() {}

  void pomp_end() {}
  void pomp_end_() {}
  void pomp_end__() {}
  void POMP_END() {}

  void pomp_flush_enter() {}
  void pomp_flush_enter_() {}
  void pomp_flush_enter__() {}
  void POMP_FLUSH_ENTER() {}

  void pomp_flush_exit() {}
  void pomp_flush_exit_() {}
  void pomp_flush_exit__() {}
  void POMP_FLUSH_EXIT() {}

  int tau_openmp_init() {}
  void TauStartOpenMPRegionTimer() {}
  void TauStopOpenMPRegionTimer() {}

  void POMP_Finalize() {}
  void POMP_Init() {}
  void POMP_Off() {}
  void POMP_On() {}
  void POMP_Atomic_enter() {}
  void POMP_Atomic_exit() {}
  void POMP_Barrier_enter() {}
  void POMP_Barrier_exit() {}
  void POMP_Critical_begin() {}
  void POMP_Critical_end() {}
  void POMP_Critical_enter() {}
  void POMP_Critical_exit() {}
  void POMP_For_enter() {}
  void POMP_For_exit() {}
  void POMP_Master_begin() {}
  void POMP_Master_end() {}
  void POMP_Parallel_begin() {}
  void POMP_Parallel_end() {}
  void POMP_Parallel_fork() {}
  void POMP_Parallel_join() {}
  void POMP_Section_begin() {}
  void POMP_Section_end() {}
  void POMP_Sections_enter() {}
  void POMP_Sections_exit() {}
  void POMP_Single_begin() {}
  void POMP_Single_end() {}
  void POMP_Single_enter() {}
  void POMP_Single_exit() {}
  void POMP_Workshare_enter() {}
  void POMP_Workshare_exit() {}
  void POMP_Begin() {}
  void POMP_End() {}
  void POMP_Flush_enter() {}
  void POMP_Flush_exit() {}
  void POMP_Init_lock() {}
  void POMP_Destroy_lock() {}
  void POMP_Set_lock() {}
  void POMP_Unset_lock() {}
  int  POMP_Test_lock() {}
  void POMP_Init_nest_lock() {}
  void POMP_Destroy_nest_lock() {}
  void POMP_Set_nest_lock() {}
  void POMP_Unset_nest_lock() {}
  int  POMP_Test_nest_lock() {}
}

/***************************************************************************
 * $RCSfile: TauKojakOpariDisable.cpp,v $   $Author: amorris $
 * $Revision: 1.1 $   $Date: 2008/06/06 18:39:06 $
 * POOMA_VERSION_ID: $Id: TauKojakOpariDisable.cpp,v 1.1 2008/06/06 18:39:06 amorris Exp $
 ***************************************************************************/


