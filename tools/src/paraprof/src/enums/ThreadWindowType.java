package edu.uoregon.tau.paraprof.enums;

public class ThreadWindowType {

    
    private final String name;
    
    private ThreadWindowType(String name) { this.name = name; }
    
    public String toString() { return name; }
    
    public static final ThreadWindowType THREAD_WINDOW = new ThreadWindowType("Thread Window");
    
}
