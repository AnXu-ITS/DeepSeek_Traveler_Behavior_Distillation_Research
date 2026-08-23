// RunMatsimPreloaded: MATSim launcher used by the project's run_matsim.ps1.
//
// Background: under Java 25 + MATSim 2026.0, the default RunMatsim entry point
// materializes the Scenario through a Guice provider that runs twice, which
// loads the network a second time and crashes with
// "There exists already a node with id = ...". Preloading the scenario and
// handing it to Controler(Scenario) binds Scenario as an instance, so the
// buggy provider path never executes.
//
// Build (ASCII paths only; the PowerShell->javac argv encoding corrupts the
// non-ASCII workspace path, hence the Temp junction used by build scripts):
//   javac -cp "<matsim.jar>;<matsim>/libs/*" -d <outdir> RunMatsimPreloaded.java
// Run:
//   java -cp "<outdir>;<matsim.jar>;<matsim>/libs/*" RunMatsimPreloaded config.xml

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.matsim.core.scenario.ScenarioUtils;

public final class RunMatsimPreloaded {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("usage: RunMatsimPreloaded <config.xml>");
            System.exit(2);
        }
        Config config = ConfigUtils.loadConfig(args[0]);
        Scenario scenario = ScenarioUtils.loadScenario(config);
        Controler controler = new Controler(scenario);
        controler.run();
    }
}
