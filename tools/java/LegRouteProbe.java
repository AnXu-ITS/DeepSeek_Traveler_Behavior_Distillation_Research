// Probe: load config+population, print each leg's route class (debug tool).
import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.population.io.PopulationReader;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.PlanElement;

public class LegRouteProbe {
    public static void main(String[] args) {
        Config config = ConfigUtils.loadConfig(args[0]);
        Scenario scenario = ScenarioUtils.createScenario(config);
        new PopulationReader(scenario).readFile(config.plans().getInputFile());
        int generic = 0, network = 0, pt = 0, none = 0, other = 0;
        for (Person p : scenario.getPopulation().getPersons().values()) {
            for (PlanElement pe : p.getSelectedPlan().getPlanElements()) {
                if (pe instanceof Leg leg) {
                    if (leg.getRoute() == null) { none++; }
                    else {
                        String cls = leg.getRoute().getClass().getSimpleName();
                        if (cls.contains("GenericRouteImpl")) { generic++;
                            if (generic <= 3) System.out.println("GENERIC: person=" + p.getId() + " legMode=" + leg.getMode() + " desc=" + leg.getRoute().getRouteDescription());
                        }
                        else if (cls.contains("NetworkRoute")) network++;
                        else if (cls.contains("DefaultTransitPassengerRoute")) pt++;
                        else { other++; if (other <= 3) System.out.println("OTHER: person=" + p.getId() + " mode=" + leg.getMode() + " class=" + cls); }
                    }
                }
            }
        }
        System.out.println("network=" + network + " pt=" + pt + " generic=" + generic + " none=" + none + " other=" + other);
    }
}
