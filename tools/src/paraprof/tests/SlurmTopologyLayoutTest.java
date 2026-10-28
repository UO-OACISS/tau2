package edu.uoregon.tau.paraprof;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class SlurmTopologyLayoutTest {
    private static int inputOrder;

    private SlurmTopologyLayoutTest() {
    }

    public static void main(String[] args) {
        testCanonicalPathsAndOrdering();
        testOpaqueTypesAndRepeatedNames();
        testInvalidAndPartialParticipants();
        testDifferentPathDepthsAndSingleton();
        testDeterminismAndScale();
        System.out.println("Slurm topology parser/layout tests passed.");
    }

    private static void testCanonicalPathsAndOrdering() {
        List<SlurmTopologyLayout.Participant> participants =
                new ArrayList<SlurmTopologyLayout.Participant>();
        participants.add(participant("root.s2610.frontier10247", "switch.switch.node",
                "1", 1, 0));
        participants.add(participant("root.s2610.frontier10247", "switch.switch.node",
                "0", 0, 0));
        participants.add(participant("root.s2610.frontier10248", "switch.switch.node",
                "0", 2, 0));
        participants.add(participant("root.s2611.frontier10248", "switch.switch.node",
                "0", 3, 0));

        SlurmTopologyLayout.Layout layout = requireLayout(participants);
        float[][] coordinates = layout.getCoordinates();
        assertTrue(coordinates[1][0] < coordinates[0][0],
                "same-node ranks must follow local ID order");
        assertTrue(distance(coordinates[1], coordinates[0]) <
                distance(coordinates[1], coordinates[2]),
                "same-node participants must be grouped more tightly");
        assertTrue(!same(coordinates[2], coordinates[3]),
                "nodes under separate switches must occupy distinct regions");
        assertEquals(6, layout.getResourceCount(),
                "shared hierarchy components should be deduplicated");

        List<SlurmTopologyLayout.Participant> noLocalIds =
                new ArrayList<SlurmTopologyLayout.Participant>();
        noLocalIds.add(participant("root.s1.node1", "switch.switch.node", null, 9, 0));
        noLocalIds.add(participant("root.s1.node1", "switch.switch.node", null, 2, 0));
        float[][] rankOrdered = requireLayout(noLocalIds).getCoordinates();
        assertTrue(rankOrdered[1][0] < rankOrdered[0][0],
                "process rank must order participants when local ID is unavailable");
    }

    private static void testOpaqueTypesAndRepeatedNames() {
        List<SlurmTopologyLayout.Participant> participants =
                new ArrayList<SlurmTopologyLayout.Participant>();
        participants.add(participant("root.groupA.node7", "root.unknown.node", null, 0, 0));
        participants.add(participant("root.groupB.node7", "root.unknown.node", null, 1, 0));

        SlurmTopologyLayout.Layout layout = requireLayout(participants);
        assertTrue(layout.getResourceCount() == 5,
                "same-named leaves under separate parents must remain distinct");
        assertTrue(!same(layout.getCoordinates()[0], layout.getCoordinates()[1]),
                "same-named leaves under separate parents must not overlap");
    }

    private static void testInvalidAndPartialParticipants() {
        List<SlurmTopologyLayout.Participant> participants =
                new ArrayList<SlurmTopologyLayout.Participant>();
        participants.add(participant("root.s1.node1", "switch.switch.node", "0", 0, 0));
        participants.add(participant("root.s1", "switch.switch.node", "1", 1, 0));
        participants.add(participant("root..node3", "switch.switch.node", "2", 2, 0));
        participants.add(participant(null, null, null, 3, 0));

        SlurmTopologyLayout.Layout layout = requireLayout(participants);
        Set<String> positions = uniqueCoordinates(layout);
        assertEquals(participants.size(), positions.size(),
                "invalid or absent participants must get separate fallback positions");

        List<SlurmTopologyLayout.Participant> invalid =
                new ArrayList<SlurmTopologyLayout.Participant>();
        invalid.add(participant("root.s1", "switch.switch.node", null, 0, 0));
        invalid.add(participant("root..node1", "switch.switch.node", null, 1, 0));
        assertTrue(SlurmTopologyLayout.create(invalid) == null,
                "profiles with no valid Slurm hierarchy should fall back to existing layout");
    }

    private static void testDifferentPathDepthsAndSingleton() {
        List<SlurmTopologyLayout.Participant> mixedDepth =
                new ArrayList<SlurmTopologyLayout.Participant>();
        mixedDepth.add(participant("root.node1", "switch.node", "0", 0, 0));
        mixedDepth.add(participant("root.group.node2", "switch.switch.node", "0", 1, 0));
        assertEquals(2, uniqueCoordinates(requireLayout(mixedDepth)).size(),
                "paths at different depths must both be represented");

        List<SlurmTopologyLayout.Participant> singleton =
                new ArrayList<SlurmTopologyLayout.Participant>();
        singleton.add(participant("root.node1", "switch.node", null, 0, 0));
        SlurmTopologyLayout.Layout layout = requireLayout(singleton);
        assertEquals(1, layout.getCoordinates().length, "singleton participant count");
        assertEquals(0, (int) layout.getMax()[0], "singleton X extent");
        assertEquals(0, (int) layout.getMax()[1], "singleton Y extent");
        assertFinite(layout.getCoordinates()[0]);
    }

    private static void testDeterminismAndScale() {
        List<SlurmTopologyLayout.Participant> small =
                new ArrayList<SlurmTopologyLayout.Participant>();
        small.add(participant("root.s2.node1", "switch.switch.node", "0", 1, 0));
        small.add(participant("root.s1.node1", "switch.switch.node", "0", 0, 0));
        SlurmTopologyLayout.Layout first = requireLayout(small);
        SlurmTopologyLayout.Layout second = requireLayout(small);
        assertTrue(same(first.getCoordinates()[0], second.getCoordinates()[0])
                && same(first.getCoordinates()[1], second.getCoordinates()[1]),
                "layout must be deterministic across repeated calls");

        List<SlurmTopologyLayout.Participant> large =
                new ArrayList<SlurmTopologyLayout.Participant>(10000);
        for (int switchId = 0; switchId < 10; switchId++) {
            for (int nodeId = 0; nodeId < 10; nodeId++) {
                for (int localId = 0; localId < 100; localId++) {
                    large.add(participant(String.format("root.s%02d.n%02d", switchId, nodeId),
                            "root.switch.node", Integer.toString(localId),
                            switchId * 1000 + nodeId * 100 + localId, 0));
                }
            }
        }
        SlurmTopologyLayout.Layout layout = requireLayout(large);
        assertEquals(10000, uniqueCoordinates(layout).size(),
                "10,000-participant layout must not overlap");
        assertEquals(111, layout.getResourceCount(),
                "10,000-participant hierarchy resource count");
        assertTrue(layout.getMax()[0] < 1000 && layout.getMax()[1] < 1000,
                "scale layout should remain bounded and compact");
    }

    private static SlurmTopologyLayout.Participant participant(String address,
            String pattern, String localId, int rank, int threadId) {
        return new SlurmTopologyLayout.Participant(address, pattern, localId,
                rank, threadId, inputOrder++);
    }

    private static SlurmTopologyLayout.Layout requireLayout(
            List<SlurmTopologyLayout.Participant> participants) {
        SlurmTopologyLayout.Layout layout = SlurmTopologyLayout.create(participants);
        assertTrue(layout != null, "expected a valid layout");
        assertEquals(participants.size(), layout.getCoordinates().length,
                "every represented participant needs coordinates");
        for (float[] coordinate : layout.getCoordinates()) {
            assertFinite(coordinate);
        }
        return layout;
    }

    private static Set<String> uniqueCoordinates(SlurmTopologyLayout.Layout layout) {
        Set<String> coordinates = new HashSet<String>();
        for (float[] point : layout.getCoordinates()) {
            coordinates.add(point[0] + ":" + point[1] + ":" + point[2]);
        }
        return coordinates;
    }

    private static float distance(float[] left, float[] right) {
        float x = left[0] - right[0];
        float y = left[1] - right[1];
        float z = left[2] - right[2];
        return x * x + y * y + z * z;
    }

    private static boolean same(float[] left, float[] right) {
        return left[0] == right[0] && left[1] == right[1] && left[2] == right[2];
    }

    private static void assertFinite(float[] coordinate) {
        for (float value : coordinate) {
            assertTrue(!Float.isNaN(value) && !Float.isInfinite(value),
                    "coordinates must be finite");
        }
    }

    private static void assertEquals(int expected, int actual, String message) {
        assertTrue(expected == actual, message + ": expected " + expected + ", got " + actual);
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
