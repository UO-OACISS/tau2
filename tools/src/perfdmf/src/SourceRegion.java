package edu.uoregon.tau.perfdmf;

import java.util.ArrayList;
import java.util.List;

public class SourceRegion {

    private String filename;
    private int startLine, endLine, startColumn, endColumn;
    private List<Parameter> parameters = new ArrayList<Parameter>();
    private String shortName;

    public String getShortName() {
		return shortName;
	}

	public void setShortName(String shortName) {
		this.shortName = shortName;
	}

	public SourceRegion() {}

    public String toString() {
        return filename + " " + startLine + " " + startColumn + " " + endLine + " " + endColumn;
    }

    public int getEndColumn() {
        return endColumn;
    }

    public void setEndColumn(int endColumn) {
        this.endColumn = endColumn;
    }

    public int getEndLine() {
        return endLine;
    }

    public void setEndLine(int endLine) {
        this.endLine = endLine;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public int getStartColumn() {
        return startColumn;
    }

    public void setStartColumn(int startColumn) {
        this.startColumn = startColumn;
    }

    public int getStartLine() {
        return startLine;
    }

    public void setStartLine(int startLine) {
        this.startLine = startLine;
    }

	public List<Parameter> getParameters() {
		return parameters;
	}

	public void setParameters(List<Parameter> parameters) {
		this.parameters = parameters;
	}

}
