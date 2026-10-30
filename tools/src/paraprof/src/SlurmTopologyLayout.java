package edu.uoregon.tau.paraprof;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

/**
 * Generates a hierarchical 2D/3D layout from scheduler topology information.
 *
 * The scheduler topology is represented as a dotted hierarchy where each
 * participant provides:
 *
 *   address = concrete path (e.g. "c0.n1.s2")
 *   pattern = hierarchy labels (e.g. "cabinet.node.socket")
 *
 * The layout process consists of:
 *
 *   1. Build a topology tree from address/pattern pairs.
 *   2. Attach participants to the corresponding leaf nodes.
 *   3. Compute the size required by every subtree.
 *   4. Place subtrees into a hierarchy-preserving grid.
 *   5. Assign final X/Y/Z coordinates.
 *
 * The implementation is intentionally scheduler-agnostic. Slurm hierarchy
 * data is the primary source today, but any scheduler or runtime that can
 * provide compatible address/pattern paths can reuse this layout engine.
 */
public final class SlurmTopologyLayout {
    public static final String SOURCE_NAME = "Slurm hierarchy (Flat)";
    public static final String THREADS_ON_Z_NAME = "Slurm hierarchy (Threads on Z)";
    private static final int GRID_GAP = 1;

    private SlurmTopologyLayout() {
    }

    /**
 * Controls how rank/thread relationships are represented.
 *
 * New 3D layout strategies should generally be implemented by:
 *
 *   1. Extending this enum.
 *   2. Updating placementCount().
 *   3. Updating assignCoordinates().
 *
 * The hierarchy measurement logic remains unchanged as long as the new
 * strategy preserves the same subtree sizing contract.
 */
    public enum ParticipantLayout {
        FLAT,
        THREADS_ON_Z
    }

/**
 * Represents a single profile participant (rank/thread pair) and its
 * scheduler topology metadata.
 *
 * localId is an optional scheduler-provided ordering value used to keep
 * participants from the same locality in scheduler-defined order when
 * available. When missing or malformed we fall back to TAU process rank.
 */
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

    /**
 * Result of topology layout generation.
 *
 * coordinates contains one entry per input participant:
 *
 *   coordinates[i][0] = X
 *   coordinates[i][1] = Y
 *   coordinates[i][2] = Z
 *
 * min/max describe the overall bounding box of the generated layout.
 * resourceCount reports the number of hierarchy resources (tree nodes)
 * excluding the synthetic root.
 */
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

    /**
 * One hierarchy component.
 *
 * Example:
 *
 *   pattern segment = "node"
 *   address segment = "n42"
 *
 * Ordering is first by resource type and then by resource name so layouts
 * remain deterministic regardless of input ordering.
 */
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

    /**
 * Lightweight growable int array used to avoid Integer boxing overhead
 * when storing participant indices.
 */
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

    /**
 * Internal topology tree node.
 *
 * children describe scheduler hierarchy resources.
 * participants contains profile participants attached directly to this node.
 *
 * width/height:
 *     measured footprint of this subtree.
 *
 * offsetX/offsetY:
 *     position relative to parent.
 *
 * originX/originY:
 *     absolute position in layout space.
 *
 * participantOffsetX/participantOffsetY:
 *     position of this node's participant block within the subtree.
 */
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
        return create(participants, ParticipantLayout.FLAT);
    }

    /**
 * Builds a complete topology layout.
 *
 * Participants with valid hierarchy paths are inserted into the topology
 * tree. Participants lacking usable topology information are attached to
 * the synthetic root so that every participant still receives coordinates.
 */
    public static Layout create(List<Participant> participants, ParticipantLayout participantLayout) {
        if (participants == null || participants.isEmpty()) {
            return null;
        }

        Node root = new Node();
        List<Node> nodes = new ArrayList<Node>();
        nodes.add(root);
        boolean[] validPaths = new boolean[participants.size()];
        int validCount = 0;

        // Build a resource hierarchy tree from the scheduler-provided
        // address/pattern paths and attach participants at the corresponding leaf.
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

        // Topology information may be missing or malformed. Place such
        // participants under the synthetic root so they still appear.
        for (int i = 0; i < participants.size(); i++) {
            if (!validPaths[i]) {
                if (root.participants == null) {
                    root.participants = new IntList();
                }
                root.participants.add(i);
            }
        }

        // Ensure deterministic participant ordering within each locality.
        for (Node node : nodes) {
            if (node.participants != null) {
                sortParticipants(node.participants, participants);
            }
        }

        // Measure from leaves upward so parent footprints can incorporate
        // the already-computed size of each child subtree.
        for (int i = nodes.size() - 1; i >= 0; i--) {
            measure(nodes.get(i), participants, participantLayout);
        }

        float[][] coordinates = new float[participants.size()][3];
        int maxZ = assignCoordinates(nodes, participants, coordinates, participantLayout);
        float[] min = {0.0f, 0.0f, 0.0f};
        float[] max = {root.width - 1.0f, root.height - 1.0f, maxZ};
        return new Layout(coordinates, min, max, nodes.size() - 1);
    }

    /**
 * Splits a topology path into hierarchy components.
 *
 * Paths must be fully specified. Empty segments are treated as invalid
 * because they would introduce ambiguous hierarchy levels.
 */
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

    /**
 * Ordering used for participant placement.
 *
 * Priority:
 *   1. Scheduler local ordering (if supplied)
 *   2. Process rank
 *   3. Thread id
 *   4. Original input order
 *
 * This guarantees stable ordering even when inputs contain ties or
 * incomplete scheduler metadata.
 */
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

    /**
 * Stable iterative merge sort on participant indices.
 *
 * Uses integer arrays instead of object collections to minimize allocation
 * overhead when laying out very large profile datasets.
 */
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

    /**
 * Computes the rectangular footprint required by a subtree.
 *
 * Children and any participant block owned by this node are packed into
 * an approximately square grid. The resulting width/height are then used
 * by parent nodes during their own measurement phase.
 *
 * This phase determines only space requirements and relative offsets.
 * Absolute coordinates are assigned later.
 */
    private static void measure(Node node, List<Participant> participants,
            ParticipantLayout participantLayout) {
        int childCount = node.children.size();
        int ownCount = node.participants == null ? 0
                : placementCount(node.participants, participants, participantLayout);
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

    /**
 * Converts element sizes into cumulative starting offsets while preserving
 * GRID_GAP spacing between neighboring items.
 */
    private static int[] offsets(int[] sizes) {
        int[] offsets = new int[sizes.length];
        for (int i = 1; i < sizes.length; i++) {
            offsets[i] = offsets[i - 1] + sizes[i - 1] + GRID_GAP;
        }
        return offsets;
    }

    /**
 * Calculates the total span occupied by a row or column of items including
 * inter-item gaps.
 */
    private static int extent(int[] sizes) {
        if (sizes.length == 0) {
            return 0;
        }
        return sizes[sizes.length - 1] + offsets(sizes)[sizes.length - 1];
    }

    /**
 * Returns the number of XY placements required for this participant group.
 *
 * FLAT:
 *     every participant consumes its own XY slot.
 *
 * THREADS_ON_Z:
 *     participants sharing the same process rank occupy the same XY
 *     position and are stacked along the Z axis.
 */
    private static int placementCount(IntList indices, List<Participant> participants,
            ParticipantLayout participantLayout) {
        if (participantLayout == ParticipantLayout.FLAT) {
            return indices.size;
        }
        TreeMap<Integer, Boolean> ranks = new TreeMap<Integer, Boolean>();
        for (int i = 0; i < indices.size; i++) {
            ranks.put(participants.get(indices.values[i]).processRank, Boolean.TRUE);
        }
        return ranks.size();
    }

    /**
 * Assigns final coordinates to every participant.
 *
 * XY placement follows the footprint computed during the measurement pass.
 *
 * THREADS_ON_Z mode collapses all threads belonging to the same process
 * rank into a shared XY location and assigns increasing Z coordinates to
 * differentiate threads.
 *
 * The traversal also propagates parent origins to children, converting
 * relative subtree offsets into absolute coordinates.
 *
 * Returns the largest assigned Z value for bounding-box calculation.
 */
    private static int assignCoordinates(List<Node> nodes, List<Participant> participants,
            float[][] coordinates, ParticipantLayout participantLayout) {
        int maxZ = 0;
        for (Node node : nodes) {
            if (node.participants != null) {
                int placementCount = placementCount(node.participants, participants, participantLayout);
                int columns = (int) Math.ceil(Math.sqrt(placementCount));
                TreeMap<Integer, Integer> rankPlacements = new TreeMap<Integer, Integer>();
                TreeMap<Integer, Integer> rankDepths = new TreeMap<Integer, Integer>();
                for (int i = 0; i < node.participants.size; i++) {
                    int participantIndex = node.participants.values[i];
                    Participant participant = participants.get(participantIndex);
                    int placement = i;
                    int z = 0;
                    // The first thread encountered for a rank establishes the shared
                    // XY placement. Additional threads reuse that placement and are
                    // stacked upward in Z.
                    if (participantLayout == ParticipantLayout.THREADS_ON_Z) {
                        Integer existingPlacement = rankPlacements.get(participant.processRank);
                        if (existingPlacement == null) {
                            existingPlacement = rankPlacements.size();
                            rankPlacements.put(participant.processRank, existingPlacement);
                        }
                        placement = existingPlacement;
                        Integer existingDepth = rankDepths.get(participant.processRank);
                        z = existingDepth == null ? 0 : existingDepth;
                        rankDepths.put(participant.processRank, z + 1);
                        maxZ = Math.max(maxZ, z);
                    }
                    coordinates[participantIndex][0] = node.originX + node.participantOffsetX
                            + placement % columns;
                    coordinates[participantIndex][1] = node.originY + node.participantOffsetY
                            + placement / columns;
                    coordinates[participantIndex][2] = z;
                }
            }
            for (Node child : node.children.values()) {
                child.originX = node.originX + child.offsetX;
                child.originY = node.originY + child.offsetY;
            }
        }
        return maxZ;
    }

}
