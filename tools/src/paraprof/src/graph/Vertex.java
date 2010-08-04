package edu.uoregon.tau.paraprof.graph;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.paraprof.CallGraphWindow.GraphCell;

public class Vertex implements Comparable {

    // A simple structure to hold pairs of vertices
    public static class BackEdge {
        public Vertex a, b;

        public BackEdge(Vertex a, Vertex b) {
            this.a = a;
            this.b = b;
        }
    }

    private List<Vertex> children = new ArrayList<Vertex>();
    private List<Vertex> parents = new ArrayList<Vertex>();
    private Object userObject;
    private boolean visited;

    private int downPriority;
    private int upPriority;

    private int level = -1; // which level this vertex resides on
    private int levelIndex; // the index within the level

    private double baryCenter;
    private double gridBaryCenter;

    private GraphCell graphCell;
    private Object graphObject;
    private int position = -1;
    private int width;
    private int height;
    private float colorRatio;

    private boolean pathHighlight = false;

    public Object getUserObject() {
        return userObject;
    }

    public Vertex(Object userObject, int width, int height) {
        this.userObject = userObject;

        this.setWidth(width);
        this.setHeight(height);

        if (userObject != null && width < 5) {
            this.setWidth(5);
        }
    }

    public int compareTo(Object compare) {
        if (this.getBaryCenter() < ((Vertex) compare).getBaryCenter())
            return -1;
        if (this.getBaryCenter() > ((Vertex) compare).getBaryCenter())
            return 1;
        return 0;
    }

    public int getPriority(boolean down) {
        if (down) {
            return getDownPriority();
        } else {
            return getUpPriority();
        }
    }

    public void setPathHighlight(boolean pathHighlight) {
        this.pathHighlight = pathHighlight;
    }

    public boolean getPathHighlight() {
        return pathHighlight;
    }

    public void setChildren(List<Vertex> children) {
        this.children = children;
    }

    public List<Vertex> getChildren() {
        return children;
    }

    public void setParents(List<Vertex> parents) {
        this.parents = parents;
    }

    public List<Vertex> getParents() {
        return parents;
    }

    public void setVisited(boolean visited) {
        this.visited = visited;
    }

    public boolean getVisited() {
        return visited;
    }

    public void setDownPriority(int downPriority) {
        this.downPriority = downPriority;
    }

    public int getDownPriority() {
        return downPriority;
    }

    public void setUpPriority(int upPriority) {
        this.upPriority = upPriority;
    }

    public int getUpPriority() {
        return upPriority;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public int getLevel() {
        return level;
    }

    public void setLevelIndex(int levelIndex) {
        this.levelIndex = levelIndex;
    }

    public int getLevelIndex() {
        return levelIndex;
    }

    public void setBaryCenter(double baryCenter) {
        this.baryCenter = baryCenter;
    }

    public double getBaryCenter() {
        return baryCenter;
    }

    public void setGridBaryCenter(double gridBaryCenter) {
        this.gridBaryCenter = gridBaryCenter;
    }

    public double getGridBaryCenter() {
        return gridBaryCenter;
    }

    public void setGraphCell(GraphCell graphCell) {
        this.graphCell = graphCell;
    }

    public GraphCell getGraphCell() {
        return graphCell;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public int getPosition() {
        return position;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getWidth() {
        return width;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getHeight() {
        return height;
    }

    public void setColorRatio(float colorRatio) {
        this.colorRatio = colorRatio;
    }

    public float getColorRatio() {
        return colorRatio;
    }

    public void setGraphObject(Object graphObject) {
        this.graphObject = graphObject;
    }

    public Object getGraphObject() {
        return graphObject;
    }

}
