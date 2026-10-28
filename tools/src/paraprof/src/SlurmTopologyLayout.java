package edu.uoregon.tau.paraprof;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

public final class SlurmTopologyLayout {
    public static final String SOURCE_NAME = "Slurm hierarchy (derived)";
    private static final int GRID_GAP = 1;

    private SlurmTopologyLayout() {
    }

    public static final class Participant {
        private final String address;
        private final String pattern;
        private final long localOrder;
        private final boolean hasLocalOrder;
        private final int processRank;
        private final int threadId;
        private final int inputOrder;

        public Participant(String address, String pattern, String localId,
                int processRank, int threadId, int inputOrder) {
            this.address = address;
            this.pattern = pattern;
            this.processRank = processRank;
            this.threadId = threadId;
            this.inputOrder = inputOrder;
            long parsedLocalOrder = 0;
            boolean parsed = false;
            if (localId != null && !localId.isEmpty()) {
                try {
                    parsedLocalOrder = Long.parseLong(localId);
                    parsed = true;
                } catch (NumberFormatException ignored) {
                    // Fall back to the TAU process rank when local ID is malformed.
                }
            }
            this.localOrder = parsedLocalOrder;
            this.hasLocalOrder = parsed;
        }
    }

    public static final class Layout {
        private final float[][] coordinates;
        private final float[] min;
        private final float[] max;
        private final int resourceCount;

        private Layout(float[][] coordinates, float[] min, float[] max, int resourceCount) {
            this.coordinates = coordinates;
            this.min = min;
            this.max = max;
            this.resourceCount = resourceCount;
        }

        public float[][] getCoordinates() {
            return coordinates;
        }

        public float[] getMin() {
            return min;
        }

        public float[] getMax() {
            return max;
        }

        public int getResourceCount() {
            return resourceCount;
        }
    }

    private static final class Segment implements Comparable<Segment> {
        private final String type;
        private final String name;

        private Segment(String type, String name) {
            this.type = type;
            this.name = name;
        }

        @Override
        public int compareTo(Segment other) {
            int typeOrder = type.compareTo(other.type);
            return typeOrder != 0 ? typeOrder : name.compareTo(other.name);
        }
    }

    private static final class IntList {
        private int[] values = new int[4];
        private int size;

        private void add(int value) {
            if (size == values.length) {
                values = Arrays.copyOf(values, values.length * 2);
            }
            values[size++] = value;
        }
    }

    private static final class Node {
        private final TreeMap<Segment, Node> children = new TreeMap<Segment, Node>();
        private IntList participants;
        private int width;
        private int height;
        private int offsetX;
        private int offsetY;
        private int originX;
        private int originY;
        private int participantOffsetX;
        private int participantOffsetY;
    }

    public static Layout create(List<Participant> participants) {
        if (participants == null || participants.isEmpty()) {
            return null;
        }

        Node root = new Node();
        List<Node> nodes = new ArrayList<Node>();
        nodes.add(root);
        boolean[] validPaths = new boolean[participants.size()];
        int validCount = 0;

        for (int i = 0; i < participants.size(); i++) {
            Participant participant = participants.get(i);
            String[] address = splitPath(participant.address);
            String[] pattern = splitPath(participant.pattern);
            if (address == null || pattern == null || address.length != pattern.length) {
                continue;
            }

            Node node = root;
            for (int component = 0; component < address.length; component++) {
                Segment segment = new Segment(pattern[component], address[component]);
                Node child = node.children.get(segment);
                if (child == null) {
                    child = new Node();
                    node.children.put(segment, child);
                    nodes.add(child);
                }
                node = child;
            }
            if (node.participants == null) {
                node.participants = new IntList();
            }
            node.participants.add(i);
            validPaths[i] = true;
            validCount++;
        }

        if (validCount == 0) {
            return null;
        }

        for (int i = 0; i < participants.size(); i++) {
            if (!validPaths[i]) {
                if (root.participants == null) {
                    root.participants = new IntList();
                }
                root.participants.add(i);
            }
        }

        for (Node node : nodes) {
            if (node.participants != null) {
                sortParticipants(node.participants, participants);
            }
        }

        for (int i = nodes.size() - 1; i >= 0; i--) {
            measure(nodes.get(i));
        }

        float[][] coordinates = new float[participants.size()][3];
        assignCoordinates(nodes, participants, coordinates);
        float[] min = {0.0f, 0.0f, 0.0f};
        float[] max = {root.width - 1.0f, root.height - 1.0f, 0.0f};
        return new Layout(coordinates, min, max, nodes.size() - 1);
    }

    private static String[] splitPath(String value) {
        if (value == null || value.isEmpty()) {
            return null;
        }
        String[] parts = value.split("\\.", -1);
        for (String part : parts) {
            if (part.isEmpty()) {
                return null;
            }
        }
        return parts;
    }

    private static int compareParticipants(Participant left, Participant right) {
        long leftOrder = left.hasLocalOrder ? left.localOrder : left.processRank;
        long rightOrder = right.hasLocalOrder ? right.localOrder : right.processRank;
        int order = Long.compare(leftOrder, rightOrder);
        if (order != 0) {
            return order;
        }
        order = Integer.compare(left.processRank, right.processRank);
        if (order != 0) {
            return order;
        }
        order = Integer.compare(left.threadId, right.threadId);
        return order != 0 ? order : Integer.compare(left.inputOrder, right.inputOrder);
    }

    private static void sortParticipants(IntList indices, List<Participant> participants) {
        int[] source = Arrays.copyOf(indices.values, indices.size);
        int[] target = new int[indices.size];
        int width = 1;
        while (width < indices.size) {
            for (int start = 0; start < indices.size; start += width * 2) {
                int middle = Math.min(start + width, indices.size);
                int end = Math.min(start + width * 2, indices.size);
                int left = start;
                int right = middle;
                for (int output = start; output < end; output++) {
                    if (left < middle && (right >= end
                            || compareParticipants(participants.get(source[left]),
                                    participants.get(source[right])) <= 0)) {
                        target[output] = source[left++];
                    } else {
                        target[output] = source[right++];
                    }
                }
            }
            int[] swap = source;
            source = target;
            target = swap;
            if (width > indices.size / 2) {
                break;
            }
            width *= 2;
        }
        System.arraycopy(source, 0, indices.values, 0, indices.size);
    }

    private static void measure(Node node) {
        int childCount = node.children.size();
        int ownCount = node.participants == null ? 0 : node.participants.size;
        int itemCount = childCount + (ownCount == 0 ? 0 : 1);
        if (itemCount == 0) {
            node.width = 0;
            node.height = 0;
            return;
        }

        int columns = (int) Math.ceil(Math.sqrt(itemCount));
        int rows = (itemCount + columns - 1) / columns;
        int[] columnWidths = new int[columns];
        int[] rowHeights = new int[rows];
        int item = 0;

        if (ownCount > 0) {
            int ownColumns = (int) Math.ceil(Math.sqrt(ownCount));
            int ownRows = (ownCount + ownColumns - 1) / ownColumns;
            columnWidths[0] = ownColumns;
            rowHeights[0] = ownRows;
            item++;
        }
        for (Node child : node.children.values()) {
            int column = item % columns;
            int row = item / columns;
            columnWidths[column] = Math.max(columnWidths[column], child.width);
            rowHeights[row] = Math.max(rowHeights[row], child.height);
            item++;
        }

        int[] columnOffsets = offsets(columnWidths);
        int[] rowOffsets = offsets(rowHeights);
        item = 0;
        if (ownCount > 0) {
            node.participantOffsetX = columnOffsets[0];
            node.participantOffsetY = rowOffsets[0];
            item++;
        }
        for (Node child : node.children.values()) {
            int column = item % columns;
            int row = item / columns;
            child.offsetX = columnOffsets[column];
            child.offsetY = rowOffsets[row];
            item++;
        }
        node.width = extent(columnWidths);
        node.height = extent(rowHeights);
    }

    private static int[] offsets(int[] sizes) {
        int[] offsets = new int[sizes.length];
        for (int i = 1; i < sizes.length; i++) {
            offsets[i] = offsets[i - 1] + sizes[i - 1] + GRID_GAP;
        }
        return offsets;
    }

    private static int extent(int[] sizes) {
        if (sizes.length == 0) {
            return 0;
        }
        return sizes[sizes.length - 1] + offsets(sizes)[sizes.length - 1];
    }

    private static void assignCoordinates(List<Node> nodes, List<Participant> participants,
            float[][] coordinates) {
        for (Node node : nodes) {
            if (node.participants != null) {
                int columns = (int) Math.ceil(Math.sqrt(node.participants.size));
                for (int i = 0; i < node.participants.size; i++) {
                    int participant = node.participants.values[i];
                    coordinates[participant][0] = node.originX + node.participantOffsetX + i % columns;
                    coordinates[participant][1] = node.originY + node.participantOffsetY + i / columns;
                }
            }
            for (Node child : node.children.values()) {
                child.originX = node.originX + child.offsetX;
                child.originY = node.originY + child.offsetY;
            }
        }
    }

}
