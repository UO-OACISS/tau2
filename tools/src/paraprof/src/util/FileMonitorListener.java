package edu.uoregon.tau.paraprof.util;

import java.io.File;


public interface FileMonitorListener
{
  /**
   * Called when one of the monitored files are created, deleted
   * or modified.
   * 
   * @param file  File which has been changed.
   */
  void fileChanged (File file);
}