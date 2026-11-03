package edu.uoregon.tau.paraprof;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Expands topology data sources into layouts offered by the 3D visualizer. */
public final class TopologyLayoutOptions {
    private TopologyLayoutOptions() {
    }

    public static List<String> forTrial(ParaProfTrial trial) {
        Set<String> sources = new LinkedHashSet<String>(trial.getTopologyArray());
        List<String> layouts = new ArrayList<String>();

        if (sources.remove(ParaProfTrial.SLURM_TOPOLOGY_SOURCE)) {
            layouts.add(SlurmTopologyLayout.SOURCE_NAME);
            layouts.add(SlurmTopologyLayout.THREADS_ON_Z_NAME);
        }
        layouts.addAll(sources);
        return layouts;
    }
}
