// Transit debug launcher: logs transit boarding/alighting events so stuck
// passengers can be identified before the "vehicle not empty at last stop"
// assertion aborts the run.
import java.util.HashMap;
import java.util.Map;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.events.PersonEntersVehicleEvent;
import org.matsim.api.core.v01.events.PersonLeavesVehicleEvent;
import org.matsim.api.core.v01.events.TransitDriverStartsEvent;
import org.matsim.api.core.v01.events.handler.PersonEntersVehicleEventHandler;
import org.matsim.api.core.v01.events.handler.PersonLeavesVehicleEventHandler;
import org.matsim.api.core.v01.events.handler.TransitDriverStartsEventHandler;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.matsim.core.scenario.ScenarioUtils;

public final class RunMatsimPreloaded {

    public static final class TransitEventLogger implements
            PersonEntersVehicleEventHandler, PersonLeavesVehicleEventHandler, TransitDriverStartsEventHandler {
        final Map<Id<Person>, String> onboard = new HashMap<>();
        int transitStarts = 0;
        int anyEvents = 0;
        java.io.PrintWriter out;

        TransitEventLogger() {
            try {
                out = new java.io.PrintWriter(new java.io.FileWriter("transit_events.log", true), true);
            } catch (java.io.IOException e) {
                throw new RuntimeException(e);
            }
        }

        private void log(String s) {
            out.println(s);
        }

        @Override
        public void handleEvent(PersonEntersVehicleEvent event) {
            anyEvents++;
            if (event.getVehicleId().toString().startsWith("veh_")) {
                onboard.put(event.getPersonId(), event.getVehicleId().toString());
                log("[TRANSIT] ENTER person=" + event.getPersonId() + " vehicle=" + event.getVehicleId());
            }
        }

        @Override
        public void handleEvent(PersonLeavesVehicleEvent event) {
            anyEvents++;
            if (event.getVehicleId().toString().startsWith("veh_")) {
                onboard.remove(event.getPersonId());
                log("[TRANSIT] LEAVE person=" + event.getPersonId() + " vehicle=" + event.getVehicleId());
            }
        }

        @Override
        public void handleEvent(TransitDriverStartsEvent event) {
            anyEvents++;
            transitStarts++;
            if (transitStarts % 200 == 0) {
                log("[TRANSIT] driverStarts=" + transitStarts + " onboard=" + onboard.size() + " anyEvents=" + anyEvents);
            }
        }

        @Override
        public void reset(int iteration) {
        }
    }

    public static void main(String[] args) {
        Config config = ConfigUtils.loadConfig(args[0]);
        Scenario scenario = ScenarioUtils.loadScenario(config);
        Controler controler = new Controler(scenario);
        TransitEventLogger logger = new TransitEventLogger();
        controler.getEvents().addHandler(logger);
        controler.run();
    }
}
