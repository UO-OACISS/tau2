package edu.uoregon.tau.common;

/**
 * Adapted from "Effective Java" by Joshua Bloch
 * 
 * Provides a simple interface for telling a thread to stop
 *
 */
public class StoppableThread extends Thread {
    private boolean stopRequested = false;

    public synchronized void requestStop() {
        stopRequested = true;
    }

    protected synchronized boolean stopRequested() {
        return stopRequested;
    }
}
