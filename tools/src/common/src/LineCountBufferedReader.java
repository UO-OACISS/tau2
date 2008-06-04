package edu.uoregon.tau.common;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;

public class LineCountBufferedReader extends BufferedReader {

    int currentLine = 0;
    
    public LineCountBufferedReader(Reader in) {
        super(in);
    }

    public String readLine() throws IOException {
        currentLine++;
        return super.readLine();
    }
    
    public int getCurrentLine() {
        return currentLine;
    }

}
