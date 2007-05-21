package edu.uoregon.tau.paraprof.graph;

import java.util.*;

public class Layout {

    private static final int HORIZONTAL_SPACING = 10;
    private static final int MARGIN = 20;

    // part one of Sugiyama's graph layout algorithm 
    public static void runSugiyama(List levels) {
        runPhaseOne(levels);
        //runPhaseTwo(levels);
    }

    private static void runPhaseOne(List levels) {

        int numIterations = 100;

        while (numIterations > 0) {
            for (int i = 0; i < levels.size() - 1; i++) {
                assignBaryCenters((List) levels.get(i), (List) levels.get(i + 1), true);
                Collections.sort((List) levels.get(i));
            }

            for (int i = levels.size() - 1; i > 0; i--) {
                assignBaryCenters((List) levels.get(i), (List) levels.get(i - 1), false);
                Collections.sort((List) levels.get(i));
            }

            numIterations--;
        }
    }

    private static void assignBaryCenters(List level, List level2, boolean down) {

        for (int j = 0; j < level2.size(); j++) {
            Vertex v = (Vertex) level2.get(j);
            v.setLevelIndex(j);
        }

        for (int i = 0; i < level.size(); i++) {
            Vertex v = (Vertex) level.get(i);
            if (down) {
                int sum = 0;
                for (int j = 0; j < v.getChildren().size(); j++) {
                    sum += ((Vertex) v.getChildren().get(j)).getLevelIndex();
                }

                // don't re-assign baryCenter if no children (keep old value, based on parents)
                if (v.getChildren().size() != 0) {
                    v.setBaryCenter(sum / v.getChildren().size());
                }

            } else {
                int sum = 0;
                for (int j = 0; j < v.getParents().size(); j++) {
                    sum += ((Vertex) v.getParents().get(j)).getLevelIndex();
                }

                // don't re-assign baryCenter if no parents (keep old value, based on children)
                if (v.getParents().size() != 0) {
                    v.setBaryCenter(sum / v.getParents().size());
                }

            }
        }

    }

    private static void assignGridBaryCenters(List level, boolean down, boolean finalPass) {

        //        boolean combined = false;
        //
        //        if (!combined) {
        for (int i = 0; i < level.size(); i++) {
            Vertex v = (Vertex) level.get(i);

            //if (finalPass && v.children.size() == 0)
            //    down = false;

            if (down) {
                // don't re-assign baryCenter if no children (keep old value, based on parent)
                if (v.getChildren().size() == 0)
                    continue;

                float sum = 0;
                for (int j = 0; j < v.getChildren().size(); j++) {
                    sum += ((Vertex) v.getChildren().get(j)).getPosition();
                }

                v.setGridBaryCenter(sum / v.getChildren().size());

            } else {
                // don't re-assign baryCenter if no parents (keep old value, based on children)
                if (v.getParents().size() == 0)
                    continue;

                float sum = 0;
                for (int j = 0; j < v.getParents().size(); j++) {
                    sum += ((Vertex) v.getParents().get(j)).getPosition();
                }

                v.setGridBaryCenter(sum / v.getParents().size());

            }
        }
        //        } else {
        //            for (int i = 0; i < level.size(); i++) {
        //                Vertex v = (Vertex) level.get(i);
        //
        //                float sum = 0;
        //                for (int j = 0; j < v.children.size(); j++) {
        //                    sum += ((Vertex) v.children.get(j)).position;
        //                }
        //
        //                for (int j = 0; j < v.parents.size(); j++) {
        //                    sum += ((Vertex) v.parents.get(j)).position;
        //                }
        //
        //                v.gridBaryCenter = sum / (v.parents.size() + v.children.size());
        //
        //            }
        //        }
    }

    private static void improvePositions(List levels, int index, boolean down, boolean finalPass) {
        List level = (List) levels.get(index);

        assignGridBaryCenters(level, down, finalPass);

        for (int i = 0; i < level.size(); i++) {
            Vertex v = (Vertex) level.get(i);

            int desiredPosition = (int) v.getGridBaryCenter();

            if (down && v.getChildren().size() == 0) {
                continue;
            }

            if (desiredPosition > v.getPosition()) {
                int amountMoved = moveRight(level, i, desiredPosition - v.getPosition(), down, v.getPriority(down));
                for (int j = i - 1; j >= 0 && finalPass; j--) {
                    moveRight(level, j, amountMoved, down, v.getPriority(down));
                }
            } else {
                int amountMoved = moveLeft(level, i, v.getPosition() - desiredPosition, down, v.getPriority(down));
                for (int j = i + 1; j < level.size() && finalPass; j++) {
                    moveLeft(level, j, amountMoved, down, v.getPriority(down));
                }
            }
        }

    }

    private static int moveRight(List level, int index, int amount, boolean down, int priority) {

        Vertex v = (Vertex) level.get(index);

        int j = index + 1;

        if (j >= level.size()) {
            v.setPosition(v.getPosition() + amount);
            return amount;
        }

        Vertex u = (Vertex) level.get(j);

        int myRightSide = v.getPosition() + (v.getWidth() / 2);
        int neighborLeftSide = u.getPosition() - (u.getWidth() / 2);

        if (myRightSide + amount + HORIZONTAL_SPACING < neighborLeftSide) {
            v.setPosition(v.getPosition() + amount);
            return amount;
        }

        // not enough room between this box and the one to the right

        if (u.getPriority(down) > priority) {
            // we're lower priority, can't move him, place ourselves as far right as possible
            int newPosition = u.getPosition() - ((v.getWidth() + u.getWidth()) / 2) - HORIZONTAL_SPACING;
            int amountMoved = newPosition - v.getPosition();
            v.setPosition(v.getPosition() + amountMoved);
            return amountMoved;
        }

        int positionNeighborNeedsToBeAt = (v.getPosition() + amount) + ((u.getWidth() + v.getWidth()) / 2) + HORIZONTAL_SPACING;

        // we can move him, so ask to move '' and add whatever he can (he could be blocked by higher priority)
        moveRight(level, j, positionNeighborNeedsToBeAt - u.getPosition(), down, priority);

        int newPosition = u.getPosition() - ((v.getWidth() + u.getWidth()) / 2) - HORIZONTAL_SPACING;
        int amountMoved = newPosition - v.getPosition();
        v.setPosition(v.getPosition() + amountMoved);
        return amountMoved;
    }

    private static int moveLeft(List level, int index, int amount, boolean down, int priority) {
        Vertex v = (Vertex) level.get(index);

        int j = index - 1;

        if (j < 0) {
            v.setPosition(v.getPosition() - amount);
            return amount;
        }

        Vertex u = (Vertex) level.get(j);

        int myLeftSide = v.getPosition() - (v.getWidth() / 2);
        int neighborRightSide = u.getPosition() + (u.getWidth() / 2);

        if (myLeftSide - amount - HORIZONTAL_SPACING > neighborRightSide) {
            v.setPosition(v.getPosition() - amount);
            return amount;
        }

        if (u.getPriority(down) > priority) {
            int newPosition = u.getPosition() + ((u.getWidth() + v.getWidth()) / 2) + HORIZONTAL_SPACING;
            int amountMoved = v.getPosition() - newPosition;
            v.setPosition(v.getPosition() - amountMoved);
            return amountMoved;
        }

        int positionNeighborNeedsToBeAt = (v.getPosition() - amount) - (v.getWidth() / 2) - HORIZONTAL_SPACING
                - (u.getWidth() / 2);

        // we can move him, so ask to move '' and add whatever he can (he could be blocked by higher priority)

        moveLeft(level, j, u.getPosition() - positionNeighborNeedsToBeAt, down, priority);

        int newPosition = u.getPosition() + ((u.getWidth() + v.getWidth()) / 2) + HORIZONTAL_SPACING;
        int amountMoved = v.getPosition() - newPosition;
        v.setPosition(v.getPosition() - amountMoved);
        return amountMoved;
    }

    public static void assignPositions(List levels) {

        // assign initial positions
        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            int lastPosition = 0;
            ((Vertex) level.get(0)).setPosition(0);
            for (int j = 1; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                v.setPosition(lastPosition + HORIZONTAL_SPACING + (((Vertex) level.get(j - 1)).getWidth() + v.getWidth()) / 2);
                lastPosition = v.getPosition();

                v.setDownPriority(v.getChildren().size());
                if (v.getUserObject() == null) {
                    //v.downPriority = Integer.MAX_VALUE;
                    v.setDownPriority(2);
                }

                v.setUpPriority(v.getParents().size());
                if (v.getUserObject() == null) {
                    //v.upPriority = Integer.MAX_VALUE;
                    v.setUpPriority(2);
                }
            }

            // now center everything around zero
            int middle;

            if (level.size() % 2 == 0) {
                int left = ((Vertex) level.get((level.size() - 2) / 2)).getPosition();
                int right = ((Vertex) level.get(level.size() / 2)).getPosition();
                middle = (left + right) / 2;
            } else {
                middle = ((Vertex) level.get((level.size() - 1) / 2)).getPosition();
            }

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                v.setPosition(v.getPosition() - middle);
            }

        }

        for (int i = 1; i < levels.size(); i++) {
            improvePositions(levels, i, false, false);
        }

        for (int i = levels.size() - 2; i >= 0; i--) {
            improvePositions(levels, i, true, false);
        }

        for (int i = 1; i < levels.size(); i++) {
            improvePositions(levels, i, false, false);
        }

        for (int i = levels.size() - 2; i >= 0; i--) {
            improvePositions(levels, i, true, true);
        }

        for (int i = levels.size() - 2; i >= 0; i--) {
            improvePositions(levels, i, true, false);
        }

        // move everything right (since some of our numbers are negative)

        int minValue = 0;
        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                if (v.getPosition() - (v.getWidth() / 2) < minValue) {
                    minValue = v.getPosition() - (v.getWidth() / 2);
                }
            }
        }

        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                v.setPosition(v.getPosition() + (-minValue + MARGIN));
            }
        }
    }

    public static void assignLevel(Vertex v) {

        int maxLevel = 0;
        for (int i = 0; i < v.getParents().size(); i++) {
            Vertex parent = (Vertex) v.getParents().get(i);
            if (parent.getLevel() == -1)
                assignLevel(parent);
            if (parent.getLevel() > maxLevel)
                maxLevel = parent.getLevel();
        }
        v.setLevel(maxLevel + 1);
    }

    public static void fillLevels(Vertex v, List levels, int level) {

        if (v.getVisited() == true)
            return;

        v.setVisited(true);

        if (levels.size() == level) {
            levels.add(new ArrayList());
        }

        v.setLevel(level);

        List currentLevel = (List) levels.get(level);
        currentLevel.add(v);

        for (int i = 0; i < v.getChildren().size(); i++) {
            Vertex child = (Vertex) v.getChildren().get(i);
            fillLevels(child, levels, level + 1);
        }
    }

    public static void insertDummies(Vertex v) {

        for (int i = 0; i < v.getChildren().size(); i++) {
            Vertex child = (Vertex) v.getChildren().get(i);
            if (child.getLevel() - v.getLevel() > 1) {

                // break both edges
                v.getChildren().remove(i);
                child.getParents().remove(v);

                // create dummy and connect to child
                Vertex dummy = new Vertex(null, 1, 1);
                dummy.setLevel(v.getLevel() + 1);
                dummy.getChildren().add(child);
                child.getParents().add(dummy);

                // connect dummy to parrent
                v.getChildren().add(i, dummy);
                dummy.getParents().add(v);
                insertDummies(dummy);
            }
        }
    }
    
    public static List findRoots(Map vertexMap) {
        List roots = new ArrayList();

        for (Iterator it = vertexMap.values().iterator(); it.hasNext();) {
            Vertex v = (Vertex) it.next();
            v.setVisited(false);
        }

        for (Iterator it = vertexMap.values().iterator(); it.hasNext();) {
            Vertex v = (Vertex) it.next();
            for (int i = 0; i < v.getChildren().size(); i++) {
                Vertex child = (Vertex) v.getChildren().get(i);
                child.setVisited(true);
            }
        }

        for (Iterator it = vertexMap.values().iterator(); it.hasNext();) {
            Vertex v = (Vertex) it.next();
            if (v.getVisited() == false)
                roots.add(v);
        }
        return roots;
    }
}
